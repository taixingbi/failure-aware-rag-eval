"""Dual LLM judge for accuracy / relevance / faithfulness / citations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.client import InferenceClient
from src.parse import parse_judge_payload
from src.prompts import build_judge_messages, load_prompts


def _agree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if not a.get("parse_ok") or not b.get("parse_ok"):
        return False
    if a.get("accuracy") != b.get("accuracy"):
        return False
    for key in ("relevance", "faithfulness"):
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            return False
        if abs(float(av) - float(bv)) > 0.25:
            return False
    ca, cb = a.get("citation_accuracy"), b.get("citation_accuracy")
    if ca is None and cb is None:
        return True
    if ca is None or cb is None:
        return False
    return abs(float(ca) - float(cb)) <= 0.25


def _merge_agreed(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    cit = a.get("citation_accuracy")
    if cit is None:
        cit = b.get("citation_accuracy")
    else:
        cb = b.get("citation_accuracy")
        if cb is not None:
            cit = (float(cit) + float(cb)) / 2.0
    return {
        "accuracy": a.get("accuracy"),
        "relevance": (float(a["relevance"]) + float(b["relevance"])) / 2.0,
        "faithfulness": (float(a["faithfulness"]) + float(b["faithfulness"])) / 2.0,
        "citation_accuracy": cit,
        "reason": a.get("reason") or b.get("reason"),
        "needs_human_review": False,
        "judge_agreement": True,
    }


def judge_prediction(
    client: InferenceClient | None = None,
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    citations: list[str] | None,
    contexts: list[str],
    expected_behavior: str,
    answer_available: bool,
    primary_model: str = "gpt-oss",
    secondary_model: str = "deepseek",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
) -> dict[str, Any]:
    prompts = load_prompts()
    messages = build_judge_messages(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer or "",
        citations=citations,
        contexts=contexts,
        expected_behavior=expected_behavior,
        answer_available=answer_available,
        prompts=prompts,
    )

    def _call(model_id: str):
        # Fresh client per call for thread safety
        c = InferenceClient()
        return c.chat_completions(
            model_id,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_p = pool.submit(_call, primary_model)
        f_s = pool.submit(_call, secondary_model)
        primary = f_p.result()
        secondary = f_s.result()

    p = parse_judge_payload(primary.content)
    s = parse_judge_payload(secondary.content)

    result: dict[str, Any] = {
        "primary": {**p, "error": primary.error, "latency_ms": primary.latency_ms},
        "secondary": {**s, "error": secondary.error, "latency_ms": secondary.latency_ms},
    }

    if _agree(p, s):
        result.update(_merge_agreed(p, s))
    else:
        base = p if p.get("parse_ok") else s
        result.update(
            {
                "accuracy": base.get("accuracy"),
                "relevance": base.get("relevance"),
                "faithfulness": base.get("faithfulness"),
                "citation_accuracy": base.get("citation_accuracy"),
                "reason": base.get("reason"),
                "needs_human_review": True,
                "judge_agreement": False,
            }
        )
    return result
