"""Bedrock OpenAI-compatible chat completions client."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ChatResult:
    content: str | None
    model: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    raw: dict[str, Any]
    error: str | None = None
    ttft_ms: float | None = None
    http_status: int | None = None
    retry_count: int = 0
    streamed: bool = False
    chunk_count: int = 0


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
        stream: bool = False,
    ) -> ChatResult:
        if stream:
            return self.chat_completions_stream(
                model,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        return self._chat_completions_nonstream(
            model,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def chat_completions_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
    ) -> ChatResult:
        """Stream chat completions; TTFT = first non-empty delta.content, E2E = stream end."""
        endpoint = f"{self.base_url}v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            ttft_ms: float | None = None
            pieces: list[str] = []
            chunk_count = 0
            usage: dict[str, Any] = {}
            echoed_model = model
            http_status: int | None = None
            try:
                with self.session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,
                ) as resp:
                    http_status = resp.status_code
                    ctype = (resp.headers.get("content-type") or "").lower()

                    # Gateway may return JSON error even when stream was requested.
                    if resp.status_code >= 400 or "application/json" in ctype:
                        try:
                            data = resp.json()
                        except Exception:
                            data = {
                                "error": "non_json_response",
                                "detail": (resp.text or "")[:500],
                            }
                        err_s, retryable = _classify_error(resp.status_code, data)
                        if retryable and attempt < self.max_retries:
                            delay = min(
                                (2**attempt)
                                * (3 if "throttl" in err_s.lower() or "429" in err_s else 1),
                                30,
                            )
                            time.sleep(delay)
                            last_error = err_s
                            continue
                        e2e = (time.perf_counter() - started) * 1000.0
                        return ChatResult(
                            content=None,
                            model=data.get("model") or model,
                            latency_ms=e2e,
                            input_tokens=_usage(data, "prompt_tokens"),
                            output_tokens=_usage(data, "completion_tokens"),
                            raw=data,
                            error=err_s,
                            ttft_ms=None,
                            http_status=http_status,
                            retry_count=attempt,
                            streamed=True,
                        )

                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        chunk_count += 1
                        if chunk.get("model"):
                            echoed_model = chunk["model"]
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        if chunk.get("error"):
                            err_s = str(chunk.get("error"))
                            detail = chunk.get("detail")
                            if detail and str(detail) not in err_s:
                                err_s = f"{err_s}: {detail}"
                            e2e = (time.perf_counter() - started) * 1000.0
                            return ChatResult(
                                content="".join(pieces) or None,
                                model=echoed_model,
                                latency_ms=e2e,
                                input_tokens=_usage({"usage": usage}, "prompt_tokens"),
                                output_tokens=_usage({"usage": usage}, "completion_tokens"),
                                raw={"chunks": chunk_count, "last_error_chunk": chunk},
                                error=err_s,
                                ttft_ms=ttft_ms,
                                http_status=http_status,
                                retry_count=attempt,
                                streamed=True,
                                chunk_count=chunk_count,
                            )
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - started) * 1000.0
                            pieces.append(text)

                e2e = (time.perf_counter() - started) * 1000.0
                content = "".join(pieces) if pieces else None
                return ChatResult(
                    content=content,
                    model=echoed_model,
                    latency_ms=e2e,
                    input_tokens=_usage({"usage": usage}, "prompt_tokens"),
                    output_tokens=_usage({"usage": usage}, "completion_tokens"),
                    raw={
                        "usage": usage,
                        "chunk_count": chunk_count,
                        "ttft_ms": ttft_ms,
                        "e2e_latency_ms": e2e,
                    },
                    error=None,
                    ttft_ms=ttft_ms,
                    http_status=http_status,
                    retry_count=attempt,
                    streamed=True,
                    chunk_count=chunk_count,
                )
            except requests.RequestException as exc:
                e2e = (time.perf_counter() - started) * 1000.0
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 16))
                    continue
                return ChatResult(
                    content=None,
                    model=model,
                    latency_ms=e2e,
                    input_tokens=None,
                    output_tokens=None,
                    raw={},
                    error=last_error,
                    ttft_ms=ttft_ms,
                    http_status=http_status,
                    retry_count=attempt,
                    streamed=True,
                    chunk_count=chunk_count,
                )

        return ChatResult(
            content=None,
            model=model,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            raw={},
            error=last_error or "unknown_error",
            streamed=True,
            retry_count=self.max_retries,
        )

    def _chat_completions_nonstream(
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
                latency_ms = (time.perf_counter() - started) * 1000.0
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": "non_json_response", "detail": resp.text[:500]}

                if resp.status_code >= 400 or data.get("error"):
                    err_s, retryable = _classify_error(resp.status_code, data)
                    if retryable and attempt < self.max_retries:
                        delay = min(
                            (2**attempt)
                            * (3 if "throttl" in err_s.lower() or "429" in err_s else 1),
                            30,
                        )
                        time.sleep(delay)
                        last_error = f"http_{resp.status_code}:{err_s}"
                        continue
                    return ChatResult(
                        content=None,
                        model=data.get("model") or model,
                        latency_ms=latency_ms,
                        input_tokens=_usage(data, "prompt_tokens"),
                        output_tokens=_usage(data, "completion_tokens"),
                        raw=data,
                        error=err_s,
                        http_status=resp.status_code,
                        retry_count=attempt,
                        streamed=False,
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
                    http_status=resp.status_code,
                    retry_count=attempt,
                    streamed=False,
                )
            except requests.RequestException as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
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
                    retry_count=attempt,
                    streamed=False,
                )

        return ChatResult(
            content=None,
            model=model,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            raw={},
            error=last_error or "unknown_error",
            retry_count=self.max_retries,
            streamed=False,
        )


def _classify_error(status_code: int, data: dict[str, Any]) -> tuple[str, bool]:
    err = data.get("error") or data.get("detail") or f"http_{status_code}"
    detail = data.get("detail")
    err_s = str(err)
    if detail and str(detail) not in err_s:
        err_s = f"{err_s}: {detail}"
    hay = f"{err_s} {detail}".lower()
    rate_limited = any(
        x in hay for x in ("too many requests", "throttl", "rate limit", "429")
    )
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
    retryable = (not permanent) and (
        rate_limited
        or status_code in (429, 500, 502, 503, 504)
        or "bedrock request failed" in hay
    )
    return err_s, retryable


def _usage(data: dict[str, Any], key: str) -> int | None:
    usage = data.get("usage") or {}
    val = usage.get(key)
    if val is None:
        alt = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
        }.get(key)
        if alt:
            val = usage.get(alt)
    return int(val) if val is not None else None
