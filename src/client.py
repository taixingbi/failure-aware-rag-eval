"""Bedrock OpenAI-compatible chat completions client."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ChatResult:
    content: str | None
    model: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    raw: dict[str, Any]
    error: str | None = None


class InferenceClient:
    def __init__(
        self,
        function_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        url = (function_url or os.environ.get("FUNCTION_URL") or "").strip()
        if not url:
            raise ValueError("FUNCTION_URL is not set")
        if not url.endswith("/"):
            url += "/"
        self.base_url = url
        self.api_key = api_key or os.environ.get("INFERENCE_API_KEY") or "1234"
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def chat_completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
    ) -> ChatResult:
        endpoint = f"{self.base_url}v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self.session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": "non_json_response", "detail": resp.text[:500]}

                if resp.status_code >= 400 or data.get("error"):
                    err = data.get("error") or data.get("detail") or f"http_{resp.status_code}"
                    detail = data.get("detail")
                    err_s = str(err)
                    hay = f"{err_s} {detail}".lower()
                    rate_limited = any(
                        x in hay
                        for x in ("too many requests", "throttl", "rate limit", "429")
                    )
                    # Do not retry permanent access / availability failures
                    permanent = (not rate_limited) and any(
                        x in hay
                        for x in (
                            "not available",
                            "access is denied",
                            "not authorized",
                            "validation",
                            "invalid model",
                        )
                    )
                    if (
                        not permanent
                        and (
                            rate_limited
                            or resp.status_code in (429, 500, 502, 503, 504)
                            or "bedrock request failed" in hay
                        )
                        and attempt < self.max_retries
                    ):
                        # Longer backoff on throttling
                        delay = min((2**attempt) * (3 if rate_limited else 1), 30)
                        time.sleep(delay)
                        last_error = f"http_{resp.status_code}:{err_s}"
                        continue
                    if detail and str(detail) not in err_s:
                        err_s = f"{err_s}: {detail}"
                    return ChatResult(
                        content=None,
                        model=data.get("model") or model,
                        latency_ms=latency_ms,
                        input_tokens=_usage(data, "prompt_tokens"),
                        output_tokens=_usage(data, "completion_tokens"),
                        raw=data,
                        error=err_s,
                    )

                content = None
                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content")

                return ChatResult(
                    content=content,
                    model=data.get("model") or model,
                    latency_ms=latency_ms,
                    input_tokens=_usage(data, "prompt_tokens"),
                    output_tokens=_usage(data, "completion_tokens"),
                    raw=data,
                    error=None,
                )
            except requests.RequestException as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 16))
                    continue
                return ChatResult(
                    content=None,
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=None,
                    output_tokens=None,
                    raw={},
                    error=last_error,
                )

        return ChatResult(
            content=None,
            model=model,
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            raw={},
            error=last_error or "unknown_error",
        )


def _usage(data: dict[str, Any], key: str) -> int | None:
    usage = data.get("usage") or {}
    val = usage.get(key)
    if val is None:
        # some gateways use input_tokens / output_tokens
        alt = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
        }.get(key)
        if alt:
            val = usage.get(alt)
    return int(val) if val is not None else None
