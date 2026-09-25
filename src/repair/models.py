"""Model requests with explicit backend settings and token accounting."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from repair.io import stable_json


class ModelError(RuntimeError):
    pass


class ContentRejected(ModelError):
    pass


@dataclass
class Reply:
    text: str
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None


class UsageLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def record(self, **row: Any) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(stable_json(row) + "\n")


class ChatClient:
    def __init__(self, service: dict, usage: UsageLog | None = None):
        self.service = dict(service)
        self.model = service.get("model", "")
        self.backend = service.get("backend", "")
        if self.backend not in ("vllm", "chat"):
            raise ValueError("Service backend must be 'vllm' or 'chat'")
        endpoint = service.get("base_url", "").rstrip("/")
        if not endpoint or not self.model:
            raise ValueError("Each service needs base_url and model")
        self.endpoint = endpoint + "/chat/completions"
        self.key = os.environ.get(service.get("api_key_env", ""), "")
        if service.get("api_key_env") and not self.key:
            raise ValueError(f"Set credential variable {service['api_key_env']}")
        self.usage = usage
        self.http = httpx.Client(timeout=float(service.get("timeout_seconds", 180)))

    def close(self) -> None:
        self.http.close()

    def generate(
        self,
        messages: list[dict],
        *,
        settings: dict,
        stage: str,
        role: str,
        identity: str,
        seed: int | None = None,
        deadline: float | None = None,
    ) -> Reply:
        payload = self.payload(messages, settings, seed)
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        retries = int(self.service.get("transport_retries", 2))
        for attempt in range(retries + 1):
            remaining = deadline - time.monotonic() if deadline is not None else float("inf")
            if remaining <= 0:
                raise TimeoutError("The job deadline has elapsed")
            timeout = min(float(self.service.get("timeout_seconds", 180)), remaining)
            try:
                response = self.http.post(
                    self.endpoint, json=payload, headers=headers, timeout=timeout
                )
            except httpx.TransportError:
                status = "transport_error"
                response = None
            else:
                status = str(response.status_code)
            if response is not None and response.is_success:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice["message"]
                    usage = data.get("usage") or {}
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    finish = choice.get("finish_reason") or ""
                    if message.get("refusal") or finish == "content_filter":
                        self._record(stage, role, identity, attempt, "refused", usage)
                        raise ContentRejected("The model refused this request")
                    text = content
                    if reasoning and "<think>" not in content:
                        text = f"<think>\n{reasoning}\n</think>\n{content}"
                    self._record(stage, role, identity, attempt, finish, usage)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("The completion arrived after the job deadline")
                    return Reply(
                        text,
                        content,
                        reasoning,
                        finish,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                    )
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                    self._record(stage, role, identity, attempt, "invalid_response", {})
                    raise ModelError(
                        "The service returned an invalid completion response"
                    ) from error
            self._record(stage, role, identity, attempt, status, {})
            if (
                response is not None
                and response.status_code not in (408, 429)
                and response.status_code < 500
            ):
                # Response bodies can contain request text or credentials.
                raise ModelError(f"Model request rejected with HTTP {response.status_code}")
            if attempt == retries:
                raise ModelError(f"Model request failed after {retries + 1} attempts ({status})")
            remaining = deadline - time.monotonic() if deadline is not None else float("inf")
            delay = max(0, min(2**attempt, 8.0, remaining))
            time.sleep(delay)
        raise AssertionError("Unreachable")

    def payload(self, messages: list[dict], settings: dict, seed: int | None) -> dict:
        payload = {"model": self.model, "messages": messages}
        for key in ("temperature", "top_p"):
            if key in settings:
                payload[key] = settings[key]
        if settings.get("max_tokens"):
            payload["max_tokens"] = settings["max_tokens"]
        if seed is not None:
            payload["seed"] = seed
        if self.backend == "vllm":
            payload["chat_template_kwargs"] = {"enable_thinking": settings.get("thinking", True)}
            if "repetition_penalty" in settings:
                payload["repetition_penalty"] = settings["repetition_penalty"]
        else:
            if "repetition_penalty" in settings:
                if not self.service.get("supports_repetition_penalty", False):
                    raise ValueError("The policy service must support repetition_penalty")
                payload["repetition_penalty"] = settings["repetition_penalty"]
            thinking_field = self.service.get("thinking_field", "")
            if thinking_field:
                payload[thinking_field] = settings.get("thinking", True)
            elif "thinking" in settings and not self.service.get("thinking_always_on", False):
                raise ValueError("Configure thinking_field or thinking_always_on for this service")
        extra = self.service.get("request_fields", {})
        if set(extra) & {
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "repetition_penalty",
        }:
            raise ValueError("request_fields cannot override experiment settings")
        payload.update(extra)
        return payload

    def _record(
        self, stage: str, role: str, identity: str, attempt: int, status: str, usage: dict
    ) -> None:
        if self.usage:
            self.usage.record(
                stage=stage,
                role=role,
                identity=identity,
                attempt=attempt,
                status=status,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )


def parse_json_reply(reply: Reply) -> Any:
    if reply.finish_reason == "length":
        raise ValueError("Truncated JSON response")
    text = reply.content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)
