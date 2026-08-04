"""Prompt builders for RAG answering and judging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "configs" / "prompts.yaml"


def load_prompts(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PROMPTS
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_evidence(
    contexts: list[str],
    *,
    authoritative_chunk_ids: list[str] | None = None,
) -> str:
    auth = set(authoritative_chunk_ids or [])
    blocks: list[str] = []
    for i, text in enumerate(contexts, start=1):
        chunk_id = f"chunk_{i}"
        header = f"[{chunk_id}]"
        if chunk_id in auth:
            header = f"[{chunk_id} AUTHORITATIVE]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_answer_messages(
    question: str,
    contexts: list[str],
    *,
    authoritative_chunk_ids: list[str] | None = None,
    prompts: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    cfg = prompts or load_prompts()
    evidence = format_evidence(contexts, authoritative_chunk_ids=authoritative_chunk_ids)
    user = cfg["user_template"].format(question=question, evidence=evidence)
    return [
        {"role": "system", "content": cfg["system"].strip()},
        {"role": "user", "content": user.strip()},
    ]


def build_judge_messages(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    citations: list[str] | None,
    contexts: list[str],
    expected_behavior: str,
    answer_available: bool,
    prompts: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    cfg = prompts or load_prompts()
    evidence = format_evidence(contexts)
    user = cfg["judge_user_template"].format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
        citations=citations if citations is not None else [],
        contexts=evidence,
        expected_behavior=expected_behavior,
        answer_available=answer_available,
    )
    return [
        {"role": "system", "content": cfg["judge_system"].strip()},
        {"role": "user", "content": user.strip()},
    ]
