from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT_SECONDS,
    LLM_PROVIDER,
)


class LLMClient:
    """
    OpenAI-compatible structured generation client.

    Defaults to a deterministic local mock. Set LLM_PROVIDER=deepseek and
    DEEPSEEK_API_KEY to use DeepSeek's OpenAI-compatible chat completions API.
    """

    def __init__(self, provider: str = LLM_PROVIDER, api_key: str = DEEPSEEK_API_KEY) -> None:
        self.provider = provider
        self.api_key = api_key

    def generate_structured(self, role: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "deepseek" and self.api_key:
            return self._generate_deepseek(role, prompt, schema)
        del schema
        return {
            "role": role,
            "summary": prompt[:280],
        }

    def _generate_deepseek(self, role: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are a forecasting ReAct sub-agent. Return strict JSON only. "
            "Do not include markdown fences. Follow the requested schema and keep probabilities or strengths in [0, 1]."
        )
        user_prompt = (
            f"Role: {role}\n\n"
            f"Task:\n{prompt}\n\n"
            f"Requested JSON shape:\n{json.dumps(schema, ensure_ascii=False)}"
        )
        response = httpx.post(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=DEEPSEEK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._parse_json_content(content)

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object.")
        return parsed
