from __future__ import annotations
from typing import Literal
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


class MockRuntime:
    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: Literal["react", "reflexion"],
        reflection_memory: list[str],
    ) -> tuple[str, int, int]:
        if example.qid not in FIRST_ATTEMPT_WRONG:
            answer = example.gold_answer
        elif agent_type == "react":
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        elif attempt_id == 1 and not reflection_memory:
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        else:
            answer = example.gold_answer

        token_estimate = 220 + (attempt_id * 40)
        latency_ms = 90 + (attempt_id * 25)
        return answer, token_estimate, latency_ms

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            judge = JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        elif normalize_answer(answer) == "london":
            judge = JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        else:
            judge = JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
        token_estimate = 70
        latency_ms = 35
        return judge, token_estimate, latency_ms

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        reflection = ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
        token_estimate = 95
        latency_ms = 40
        return reflection, token_estimate, latency_ms

