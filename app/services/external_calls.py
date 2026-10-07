"""Shared policy at existing external-call boundaries, never around a graph/write flow."""
from __future__ import annotations

import contextvars
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)
cancel_event = contextvars.ContextVar("outbound_cancel_event", default=None)
_deadline = contextvars.ContextVar("outbound_deadline", default=None)


class RequestCancelled(asyncio.CancelledError):
    """Must bypass broad Exception-based business fallbacks."""


class ExternalCallError(RuntimeError):
    def __init__(self, category, status_code=None):
        self.category = category
        self.status_code = status_code
        super().__init__(category + (f" (HTTP {status_code})" if status_code else ""))


@contextmanager
def request_deadline(seconds):
    """Share one cooperative budget across retrieval, generation and repair calls."""
    parent = _deadline.get()
    deadline = time.monotonic() + float(seconds)
    token = _deadline.set(min(parent, deadline) if parent is not None else deadline)
    try:
        check_deadline()
        yield
    finally:
        _deadline.reset(token)


def check_deadline():
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise ExternalCallError("deadline_exceeded")


def remaining_timeout(maximum):
    deadline = _deadline.get()
    if deadline is None:
        return float(maximum)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExternalCallError("deadline_exceeded")
    return min(float(maximum), remaining)


def bound_timeout(timeout):
    """Clamp all HTTP phases on every attempt; socket timeouts are not a hard wall clock."""
    if _deadline.get() is None:
        return timeout
    check_deadline()
    timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
    remaining = remaining_timeout(float("inf"))
    return httpx.Timeout(**{name: remaining if value is None else min(value, remaining)
                           for name, value in timeout.as_dict().items()})


def check_cancelled():
    event = cancel_event.get()
    if event is not None and event.is_set():
        raise RequestCancelled()
    # Also cooperate with AnyIO worker cancellation. CLI calls have no host scope.
    from anyio import from_thread
    try:
        from_thread.check_cancelled()
    except RuntimeError:
        pass


def classify_error(exc):
    if isinstance(exc, ExternalCallError):
        return exc.category, exc.status_code
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = exc.response.status_code
    if code:
        category = {400: "invalid_request", 401: "authentication", 403: "permission",
                    404: "not_found", 408: "timeout", 429: "rate_limit"}.get(
                        code, "server_error" if code >= 500 else "request_rejected")
        return category, code
    from openai import APIConnectionError, APITimeoutError
    if isinstance(exc, (httpx.TimeoutException, APITimeoutError, TimeoutError)):
        return "timeout", None
    if isinstance(exc, (httpx.NetworkError, APIConnectionError, ConnectionError)):
        return "connection", None
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "invalid_response", None
    return "unexpected_error", None


def timeout_for(settings, read):
    # Client instances may be cached across requests; only attempts inherit a deadline.
    return httpx.Timeout(float(read), connect=min(float(read), float(settings.llm.timeout.connect)))


def usage_of(value):
    if isinstance(value, dict):
        usage = value.get("usage")
    else:
        usage = getattr(value, "usage", None)
    if usage is None:
        return None  # unavailable is not zero tokens
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
    return {k: v for k in keys if isinstance(v := (usage.get(k) if isinstance(usage, dict)
                                                  else getattr(usage, k, None)), int)}


class ExternalCalls:
    def __init__(self, settings, *, wait=None):
        self.policy = settings.llm.retry
        self.wait = wait or self._wait
        self.last_record = None

    @staticmethod
    def _wait(seconds):
        seconds = remaining_timeout(seconds)
        event = cancel_event.get()
        if event is None:
            time.sleep(seconds)
        else:
            event.wait(seconds)
        check_cancelled()
        check_deadline()

    def run(self, operation, fn, *, emitted=lambda: False, usage=usage_of):
        started = time.monotonic()
        attempts = self.policy.max_attempts if self.policy.enabled else 1
        for attempt in range(1, attempts + 1):
            check_cancelled()
            check_deadline()
            try:
                value = fn()
                check_cancelled()
                check_deadline()
            except Exception as exc:
                category, code = classify_error(exc)
                self._record(operation, attempt, started, category, code, None)
                check_cancelled()
                check_deadline()
                retryable = category in {"connection", "timeout", "rate_limit"} or code in {502, 503, 504}
                if not retryable or emitted() or attempt == attempts:
                    raise ExternalCallError(category, code) from None
                self.wait(remaining_timeout(min(self.policy.backoff_max, self.policy.backoff_initial * 2 ** (attempt - 1))))
            else:
                self._record(operation, attempt, started, "success", None, usage(value))
                return value

    def _record(self, operation, attempt, started, category, code, usage):
        self.last_record = {"operation": operation, "attempts": attempt, "retries": attempt - 1,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                            "category": category, "http_status": code, "usage": usage}
        logger.info("external_call %s", json.dumps(self.last_record))

    def post(self, client, url, *, operation, **kwargs):
        # Explicit status handling avoids requiring a Request on injected fake Responses.
        def checked():
            attempt_kwargs = dict(kwargs)
            if "timeout" in attempt_kwargs:
                attempt_kwargs["timeout"] = bound_timeout(attempt_kwargs["timeout"])
            elif _deadline.get() is not None:
                attempt_kwargs["timeout"] = bound_timeout(getattr(client, "timeout", None))
            response = client.post(url, **attempt_kwargs)
            if response.status_code != 200:
                category, code = classify_error(SimpleNamespace(status_code=response.status_code))
                try:
                    if code == 400 and response.json().get("error", {}).get("code") == "AllocationQuota.FreeTierOnly":
                        category = "quota_exhausted"
                except (ValueError, TypeError, AttributeError):
                    pass
                raise ExternalCallError(category, code)
            return response.json()
        return self.run(operation, checked)
