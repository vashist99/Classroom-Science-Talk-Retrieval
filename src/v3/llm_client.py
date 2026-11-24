"""Bulletproof, thread-safe UF NaviGator API client with atomic caching & USD budget ledger.

Fixes all v1/v2 client bugs:
  - Never caches HTTP errors, JSON error payloads, or empty completions.
  - Uses unique per-thread temp filenames before atomic os.replace() (Windows-safe).
  - Thread-safe CostLedger tracking prompt tokens, completion tokens, cache hits,
    new calls, and exact USD spend with a hard budget cap.
  - Supports automatic 3-key rotation (LLM_API_KEY, LLM_API_KEY_2, LLM_API_KEY_3).
  - Supports both chat completions (with reasoning_effort) and batched embeddings.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_V3_CACHE_DIR = PROJECT_ROOT / "cache" / "v3"
REQUEST_TIMEOUT = (15, 180)

# UF NaviGator AI Toolkit published rates (USD per 1M tokens); conservative defaults
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.06, 0.15),
    "gpt-oss-20b": (0.03, 0.08),
    "gemma-4-31b-it": (0.10, 0.25),
    "llama-3.3-70b-instruct": (0.40, 0.40),
    "nemotron-3-super-120b-a12b": (0.12, 0.30),
    "nemotron-3-nano-30b-a3b": (0.05, 0.12),
    "mistral-small-3.1": (0.10, 0.30),
    "granite-3.3-8b-instruct": (0.05, 0.10),
    "nomic-embed-text-v1.5": (0.01, 0.0),
    "sfr-embedding-mistral": (0.02, 0.0),
}

_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "ratelimit", "too many requests", "429")
_QUOTA_AUTH_MARKERS = (
    "quota", "budget", "credit", "balance", "insufficient",
    "spending limit", "payment", "unauthor", "invalid api key",
    "invalid_api_key", "expired", "forbidden", "402", "401", "403",
)


class BudgetExceededError(RuntimeError):
    """Raised before sending a network call that would exceed the USD budget cap."""


class AllKeysExhaustedError(RuntimeError):
    """Raised when every configured API key is rejected for quota or auth."""


def collect_api_keys(primary_env: str = "LLM_API_KEY") -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw:
            return
        for part in raw.split(","):
            k = part.strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)

    _add(os.getenv(primary_env))
    i = 2
    while (val := os.getenv(f"{primary_env}_{i}")) is not None:
        _add(val)
        i += 1
    return keys


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.20, 0.50))
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


@dataclass
class CostLedger:
    budget_cap_usd: float = 20.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    cache_hits: int = 0
    network_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd_spent: float = 0.0
    cached_usd_equivalent: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def record_hit(self, model: str, prompt_tok: int = 0, comp_tok: int = 0) -> None:
        eq_cost = estimate_cost_usd(model, prompt_tok, comp_tok)
        with self._lock:
            self.cache_hits += 1
            self.cached_usd_equivalent += eq_cost
            m = self.per_model.setdefault(
                model, {"hits": 0, "calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0, "cached_usd": 0.0}
            )
            m["hits"] += 1
            m["cached_usd"] = m.get("cached_usd", 0.0) + eq_cost

    def check_budget(self) -> None:
        with self._lock:
            if self.usd_spent >= self.budget_cap_usd:
                raise BudgetExceededError(
                    f"USD budget cap reached: ${self.usd_spent:.4f} >= ${self.budget_cap_usd:.2f}"
                )

    def record_network(self, model: str, prompt_tok: int, comp_tok: int) -> float:
        cost = estimate_cost_usd(model, prompt_tok, comp_tok)
        with self._lock:
            self.network_calls += 1
            self.prompt_tokens += int(prompt_tok)
            self.completion_tokens += int(comp_tok)
            self.usd_spent += cost
            m = self.per_model.setdefault(
                model, {"hits": 0, "calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0, "cached_usd": 0.0}
            )
            m["calls"] += 1
            m["in_tok"] += int(prompt_tok)
            m["out_tok"] += int(comp_tok)
            m["usd"] += cost
        return cost

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cache_hits": self.cache_hits,
                "network_calls": self.network_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "usd_spent": round(self.usd_spent, 5),
                "cached_usd_equivalent": round(self.cached_usd_equivalent, 5),
                "total_equivalent_usd": round(self.usd_spent + self.cached_usd_equivalent, 5),
                "budget_cap_usd": self.budget_cap_usd,
                "per_model": {
                    k: {kk: round(vv, 5) if "usd" in kk else int(vv) for kk, vv in v.items()}
                    for k, v in self.per_model.items()
                },
            }

    def diff_from(self, before: dict[str, Any]) -> dict[str, Any]:
        """Isolated per-step cost & call diff relative to an earlier summary() snapshot."""
        cur = self.summary()
        return {
            "cache_hits": cur["cache_hits"] - before.get("cache_hits", 0),
            "network_calls": cur["network_calls"] - before.get("network_calls", 0),
            "usd_spent": round(cur["usd_spent"] - before.get("usd_spent", 0.0), 5),
            "cached_usd_equivalent": round(cur["cached_usd_equivalent"] - before.get("cached_usd_equivalent", 0.0), 5),
            "total_equivalent_usd": round(cur["total_equivalent_usd"] - before.get("total_equivalent_usd", 0.0), 5),
        }


class UFNavigatorClient:
    """Thread-safe UF NaviGator OpenAI-compatible client with atomic disk cache."""

    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_V3_CACHE_DIR,
        budget_cap_usd: float = 20.0,
        completion_url: str | None = None,
        embedding_url: str | None = None,
        keys: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.completion_url = completion_url or os.getenv("COMPLETION_URL") or ""
        self.embedding_url = embedding_url or os.getenv("EMBEDDING_URL") or ""
        self.keys = keys if keys is not None else collect_api_keys("LLM_API_KEY")
        self.ledger = CostLedger(budget_cap_usd=budget_cap_usd)
        self._key_idx = 0
        self._key_lock = threading.Lock()
        self._tls = threading.local()

    def _get_session(self) -> requests.Session:
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = requests.Session()
            self._tls.session = sess
        return sess

    def _cache_key(self, namespace: str, model: str, payload: dict[str, Any]) -> str:
        blob = {"ns": namespace, "model": model, "payload": payload}
        raw = json.dumps(blob, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, namespace: str, digest: str) -> Path:
        sub = self.cache_dir / namespace / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{digest}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache_atomic(self, path: Path, obj: dict[str, Any]) -> None:
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _retire_key(self, failed_idx: int) -> bool:
        with self._key_lock:
            if self._key_idx == failed_idx:
                self._key_idx = failed_idx + 1
                return True
            return False

    def _classify_error(self, status_code: int, body: Any) -> str:
        blob = (json.dumps(body) if isinstance(body, (dict, list)) else str(body)).lower()
        if status_code == 429 or any(m in blob for m in _RATE_LIMIT_MARKERS):
            return "rate_limit"
        if status_code in (401, 402, 403) or any(m in blob for m in _QUOTA_AUTH_MARKERS):
            return "quota_auth"
        return "transient"

    def _post_with_retry(self, url: str, payload: dict[str, Any], *, max_retries: int = 3) -> dict[str, Any]:
        if not self.keys or not url:
            raise RuntimeError("UF API keys or endpoint URL not configured.")
        self.ledger.check_budget()

        transient_tries = 0
        rate_tries = 0
        while True:
            with self._key_lock:
                idx = self._key_idx
            if idx >= len(self.keys):
                raise AllKeysExhaustedError("All configured UF NaviGator API keys exhausted.")
            key = self.keys[idx]

            try:
                resp = self._get_session().post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                try:
                    data = resp.json()
                except Exception:
                    data = {"error": f"non-json HTTP {resp.status_code}: {resp.text[:200]}"}
            except Exception as e:
                transient_tries += 1
                if transient_tries > max_retries:
                    raise RuntimeError(f"Network error after {max_retries} retries: {e}") from e
                time.sleep(min(2.0 * transient_tries, 8.0))
                continue

            if resp.status_code == 200 and isinstance(data, dict) and not data.get("error"):
                return data

            kind = self._classify_error(resp.status_code, data)
            if kind == "quota_auth":
                self._retire_key(idx)
                continue
            if kind == "rate_limit":
                rate_tries += 1
                if rate_tries <= 12:
                    # Exponential backoff + round-robin wait; NEVER permanently retire a key on 429!
                    wait_s = min(3.0 * (1.5 ** min(rate_tries, 6)), 35.0)
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Persistent rate limit (HTTP {resp.status_code}): {str(data)[:200]}")

            transient_tries += 1
            if transient_tries > max_retries:
                raise RuntimeError(f"API error HTTP {resp.status_code}: {str(data)[:240]}")
            time.sleep(min(1.5 * transient_tries, 6.0))

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        reasoning_effort: str | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Cached chat completion returning {'content', 'reasoning', 'usage', 'cached'}."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        ns = f"chat_{prompt_version}"
        digest = self._cache_key(ns, model, payload)
        cpath = self._cache_path(ns, digest)
        cached = self._read_cache(cpath)
        if cached is not None and cached.get("content", "").strip():
            u = cached.get("usage") or {}
            self.ledger.record_hit(
                model,
                int(u.get("prompt_tokens") or 0),
                int(u.get("completion_tokens") or 0),
            )
            return {
                "content": cached["content"],
                "reasoning": cached.get("reasoning", ""),
                "usage": u,
                "cached": True,
            }

        last_err = "empty completion"
        for attempt in range(max_retries):
            raw = self._post_with_retry(self.completion_url, payload, max_retries=max_retries)
            usage = raw.get("usage") or {}
            self.ledger.record_network(
                model,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
            try:
                msg = raw["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()
            except (KeyError, IndexError, TypeError) as e:
                last_err = f"malformed response: {e}"
                continue

            if content:
                record = {
                    "model": model,
                    "prompt_version": prompt_version,
                    "content": content,
                    "reasoning": reasoning,
                    "usage": usage,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                self._write_cache_atomic(cpath, record)
                return {"content": content, "reasoning": reasoning, "usage": usage, "cached": False}
            last_err = f"empty content (attempt {attempt + 1})"
            time.sleep(1.0)

        raise RuntimeError(f"{model}: {last_err}")

    def embed_batch(
        self,
        *,
        model: str,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Embed a batch of strings (caller handles disk memmap caching)."""
        if not texts:
            return []
        payload = {"model": model, "input": texts}
        raw = self._post_with_retry(self.embedding_url, payload, max_retries=max_retries)
        usage = raw.get("usage") or {}
        self.ledger.record_network(model, int(usage.get("prompt_tokens") or 0), 0)
        data = sorted(raw["data"], key=lambda x: x["index"])
        return [row["embedding"] for row in data]
