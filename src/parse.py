"""Robust JSON extraction from model outputs."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    # direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # find outermost braces
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Repair truncated JSON (common when max_tokens cuts mid-reason string)
    if start >= 0:
        candidate = s[start:]
        repaired = _repair_truncated_json(candidate)
        if repaired is not None:
            return repaired

    return None


def _repair_truncated_json(s: str) -> dict[str, Any] | None:
    """Best-effort close of truncated JSON objects/strings."""
    # If odd number of unescaped quotes, close the string
    t = s
    in_str = False
    escape = False
    for ch in t:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
    if in_str:
        t += '"'
    # drop trailing comma
    t = t.rstrip()
    if t.endswith(","):
        t = t[:-1]
    # close open braces/brackets
    opens = t.count("{") - t.count("}")
    opens_b = t.count("[") - t.count("]")
    t += "]" * max(0, opens_b) + "}" * max(0, opens)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def parse_answer_payload(text: str | None) -> dict[str, Any]:
    """Parse model answer JSON into normalized fields."""
    obj = extract_json_object(text)
    if obj is None:
        return {
            "answer": None,
            "citations": [],
            "abstained": None,
            "parse_ok": False,
            "raw_text": text,
        }

    answer = obj.get("answer")
    if answer is not None:
        answer = str(answer).strip()

    citations = obj.get("citations") or []
    if isinstance(citations, str):
        citations = [citations]
    citations = [str(c).strip() for c in citations if str(c).strip()]

    abstained = obj.get("abstained")
    if abstained is None and answer:
        abstained = answer.upper() == "INSUFFICIENT_EVIDENCE"
    if isinstance(abstained, str):
        abstained = abstained.strip().lower() in {"true", "1", "yes"}

    return {
        "answer": answer,
        "citations": citations,
        "abstained": bool(abstained) if abstained is not None else False,
        "parse_ok": True,
        "raw_text": text,
    }


def parse_judge_payload(text: str | None) -> dict[str, Any]:
    obj = extract_json_object(text)
    if obj is None:
        return {
            "accuracy": None,
            "relevance": None,
            "faithfulness": None,
            "citation_accuracy": None,
            "reason": None,
            "parse_ok": False,
            "raw_text": text,
        }

    def _num(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    acc = obj.get("accuracy")
    if acc is not None:
        try:
            acc = int(round(float(acc)))
        except (TypeError, ValueError):
            acc = None

    cit = obj.get("citation_accuracy")
    if cit is None or (isinstance(cit, str) and cit.lower() in {"null", "n/a", "na"}):
        cit_val: float | None = None
    else:
        cit_val = _num(cit)

    return {
        "accuracy": acc,
        "relevance": _num(obj.get("relevance")),
        "faithfulness": _num(obj.get("faithfulness")),
        "citation_accuracy": cit_val,
        "reason": obj.get("reason"),
        "parse_ok": True,
        "raw_text": text,
    }
