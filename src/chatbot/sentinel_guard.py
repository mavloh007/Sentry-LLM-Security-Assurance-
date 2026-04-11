import os
import httpx
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_SENTINEL_GUARDRAILS: Dict[str, Dict[str, Any]] = {
    "lionguard-2-binary": {},
    "off-topic": {},
    "system-prompt-leakage": {},
    "aws/prompt_attack": {},
}


@dataclass
class SentinelResult:
    blocked: bool
    status_code: Optional[int] = None
    response_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    triggering_guardrails: Optional[List[str]] = None


class SentinelGuard:
    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        guardrails: Optional[Dict[str, Dict[str, Any]]] = None,
        timeout: int = 5,
        threshold: float = 0.90,
        fail_closed: bool = False,
    ):
        self.api_key = api_key or os.getenv("SENTINEL_API_KEY")
        self.url = "https://sentinel.stg.aiguardian.gov.sg/api/v1/validate"
        self.guardrails = guardrails or DEFAULT_SENTINEL_GUARDRAILS
        self.timeout = timeout
        self.threshold = threshold
        self.fail_closed = fail_closed

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def validate(
        self, text: str, messages: Optional[List[Dict[str, str]]] = None
    ) -> SentinelResult:
        """Validate text against Sentinel guardrails (async).

        A timeout is required to prevent a slow/unresponsive Sentinel API from
        blocking the worker thread indefinitely.  With ``fail_closed=False``
        (the default) a timeout simply lets the message through — the backend
        continues operating normally.  Set ``fail_closed=True`` if you prefer
        to reject messages when Sentinel is unreachable.
        """
        if not self.enabled:
            return SentinelResult(blocked=False, error="SENTINEL_API_KEY missing")

        payload: Dict[str, Any] = {
            "text": text,
            "guardrails": self.guardrails,
        }
        if messages:
            payload["messages"] = messages

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                response_json = response.json()

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            return SentinelResult(
                blocked=self.fail_closed,
                error=f"Sentinel API request failed: {exc}",
            )

        # Evaluate guardrail scores against threshold
        triggered = []
        results_dict = response_json.get("results", {})

        for guardrail_name, data in results_dict.items():
            score = data.get("score", 0.0)
            if score > self.threshold:
                triggered.append(f"{guardrail_name} ({score})")

        return SentinelResult(
            blocked=len(triggered) > 0,
            status_code=response.status_code,
            response_json=response_json,
            triggering_guardrails=triggered,
        )
