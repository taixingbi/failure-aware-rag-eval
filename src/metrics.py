"""Rule-based answer and failure-specific metrics."""

from __future__ import annotations

import re
import string
from typing import Any


_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s


def is_abstention(answer: str | None, abstained: bool | None = None) -> bool:
    if abstained is True:
        return True
    if not answer:
        return False
    norm = normalize_text(answer).replace(" ", "")
    return norm in {
        "insufficientevidence",
        "insufficient_evidence",
        "i cannot answer",
        "cannot answer",
        "no evidence",
        "not enough evidence",
    } or "insufficient evidence" in normalize_text(answer)


def exact_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    return pred.strip() == gold.strip()


def normalized_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    return p == g or g in p or p in g


def rule_accuracy(
    *,
    answer: str | None,
    gold_answer: str,
    expected_behavior: str,
    abstained: bool | None = None,
    alternate_answer: str | None = None,
) -> dict[str, Any]:
    abstain = is_abstention(answer, abstained)
    if expected_behavior == "abstain":
        correct = abstain
        return {
            "accuracy": int(correct),
            "exact_match": False,
            "normalized_match": False,
            "abstained": abstain,
            "rule_confident": True,
            "matched_alternate": False,
        }

    if abstain:
        return {
            "accuracy": 0,
            "exact_match": False,
            "normalized_match": False,
            "abstained": True,
            "rule_confident": True,
            "matched_alternate": False,
        }

    em = exact_match(answer, gold_answer)
    nm = normalized_match(answer, gold_answer)
    matched_alt = bool(alternate_answer) and normalized_match(answer, alternate_answer)
    if em or nm:
        return {
            "accuracy": 1,
            "exact_match": em,
            "normalized_match": nm,
            "abstained": False,
            "rule_confident": True,
            "matched_alternate": matched_alt,
        }
    if matched_alt:
        return {
            "accuracy": 0,
            "exact_match": False,
            "normalized_match": False,
            "abstained": False,
            "rule_confident": True,
            "matched_alternate": True,
        }
    # ambiguous — defer to judge
    return {
        "accuracy": None,
        "exact_match": False,
        "normalized_match": False,
        "abstained": False,
        "rule_confident": False,
        "matched_alternate": False,
    }


def citation_completeness(citations: list[str] | None) -> float:
    if not citations:
        return 0.0
    return 1.0


def citation_support_rate(
    citations: list[str] | None,
    supporting_chunk_ids: list[str] | None,
) -> float | None:
    """Programmatic citation accuracy vs known gold chunk ids. None if no citations."""
    if not citations:
        return None
    if not supporting_chunk_ids:
        return None
    support = set(supporting_chunk_ids)
    hits = sum(1 for c in citations if c in support)
    return hits / len(citations)


def correct_abstention(answer: str | None, abstained: bool | None = None) -> bool:
    return is_abstention(answer, abstained)


def conflict_resolution_ok(
    *,
    answer: str | None,
    gold_answer: str,
    alternate_answer: str | None,
    citations: list[str] | None,
    gold_chunk_ids: list[str] | None,
    abstained: bool | None = None,
) -> bool:
    if is_abstention(answer, abstained):
        return False
    if alternate_answer and normalized_match(answer, alternate_answer) and not normalized_match(
        answer, gold_answer
    ):
        return False
    if not normalized_match(answer, gold_answer):
        return False
    if gold_chunk_ids and citations:
        return any(c in set(gold_chunk_ids) for c in citations)
    return True


def hard_negative_resistant(
    *,
    answer: str | None,
    expected_behavior: str,
    abstained: bool | None = None,
) -> bool:
    # In RAGFailBench hard_negative removes gold evidence → expect abstain
    if expected_behavior == "abstain":
        return is_abstention(answer, abstained)
    return not is_abstention(answer, abstained) and bool(answer)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den
