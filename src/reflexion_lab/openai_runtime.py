from __future__ import annotations

import json
import time
from typing import Literal

from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


class OpenAIRuntime:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, temperature: float = 0.0) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _chat(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> tuple[str, int, int]:
        started = time.perf_counter()
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = response.usage.total_tokens if response.usage and response.usage.total_tokens else 0
        content = response.choices[0].message.content or ""
        return content.strip(), usage, latency_ms

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: Literal["react", "reflexion"],
        reflection_memory: list[str],
    ) -> tuple[str, int, int]:
        context_blob = "\n\n".join(f"[{idx}] {chunk.title}: {chunk.text}" for idx, chunk in enumerate(example.context, start=1))
        memory_blob = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- (none)"
        user_prompt = (
            f"Agent type: {agent_type}\n"
            f"Attempt: {attempt_id}\n"
            f"Question: {example.question}\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Reflection memory:\n{memory_blob}\n\n"
            "Return only the final short answer string."
        )
        answer, tokens, latency_ms = self._chat(ACTOR_SYSTEM, user_prompt)
        answer = answer.splitlines()[0].strip()
        return answer, tokens, latency_ms

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
        user_prompt = (
            f"Question: {example.question}\n"
            f"Gold answer: {example.gold_answer}\n"
            f"Predicted answer: {answer}\n\n"
            "Evaluate strictly by semantic equivalence after normalization."
        )
        raw, tokens, latency_ms = self._chat(
            EVALUATOR_SYSTEM,
            user_prompt,
            response_format={"type": "json_object"},
        )
        try:
            payload = json.loads(raw)
            judge = JudgeResult.model_validate(payload)
        except Exception:
            score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
            judge = JudgeResult(
                score=score,
                reason="Fallback evaluator used due to invalid JSON from model.",
                missing_evidence=[] if score == 1 else ["Evaluator output could not be parsed."],
                spurious_claims=[] if score == 1 else [answer],
            )
        return judge, tokens, latency_ms

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
        user_prompt = (
            f"Attempt id: {attempt_id}\n"
            f"Question: {example.question}\n"
            f"Failure reason: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
            f"Spurious claims: {judge.spurious_claims}\n\n"
            "Return strict JSON with keys: attempt_id, failure_reason, lesson, next_strategy."
        )
        raw, tokens, latency_ms = self._chat(
            REFLECTOR_SYSTEM,
            user_prompt,
            response_format={"type": "json_object"},
        )
        try:
            payload = json.loads(raw)
            reflection = ReflectionEntry.model_validate(payload)
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Need stronger grounding from context before finalizing the answer.",
                next_strategy="Re-read relevant context and verify second-hop entity before answering.",
            )
        return reflection, tokens, latency_ms