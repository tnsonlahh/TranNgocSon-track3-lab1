from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import ContextChunk, QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not raw:
        return []

    first = raw[0]
    if "qid" in first and "gold_answer" in first:
        return [QAExample.model_validate(item) for item in raw]

    # Adapt common HotpotQA fields into the scaffold schema.
    if "id" in first and "answer" in first and "context" in first:
        examples: list[QAExample] = []
        for idx, item in enumerate(raw, start=1):
            titles = item.get("context", {}).get("title", [])
            texts = item.get("context", {}).get("sentences", [])
            if not texts:
                texts = item.get("context", {}).get("text", [])
            chunks = [
                ContextChunk(title=str(title), text=" ".join(sentences) if isinstance(sentences, list) else str(sentences))
                for title, sentences in zip(titles, texts)
            ]
            examples.append(
                QAExample(
                    qid=item.get("id", f"hp_{idx}"),
                    difficulty=item.get("level", "medium"),
                    question=item["question"],
                    gold_answer=item["answer"],
                    context=chunks,
                )
            )
        return examples

    raise ValueError("Unsupported dataset format. Expected scaffold or HotpotQA-like JSON.")

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
