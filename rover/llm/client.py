"""Multi-provider LLM client with disk-based response caching for Rover.

Supports:
- Ollama (local or remote)
- OpenAI-compatible APIs (OpenAI, Groq, OpenRouter, etc.)
- Anthropic Claude API

Users provide their own API keys via environment variables.
No Docker required. No cloud proxy. Your keys, your choice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any


class ResponseCache:
    """Disk-based SQLite cache for LLM responses.

    Cache keys are SHA256 hashes of (model + prompt + system + temperature + json_mode).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path.home() / ".rover" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if not exists."""
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prompt_hash ON llm_cache(prompt_hash)"
            )
            conn.commit()

    def _make_key(
        self, model: str, prompt: str, system: str, temperature: float, json_mode: bool
    ) -> str:
        payload = f"{model}::{system}::{prompt}::{temperature}::{json_mode}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self, model: str, prompt: str, system: str = "", temperature: float = 0.3, json_mode: bool = False
    ) -> str | None:
        """Retrieve cached response if exists and not expired."""
        key = self._make_key(model, prompt, system, temperature, json_mode)
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                row = conn.execute(
                    "SELECT response, created_at, ttl_seconds FROM llm_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row:
                    response, created_at, ttl = row
                    if ttl == 0 or (time.time() - created_at) < ttl:
                        return response
        except sqlite3.Error:
            pass
        return None

    def set(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        system: str = "",
        temperature: float = 0.3,
        json_mode: bool = False,
        ttl_seconds: int = 0,
    ) -> None:
        """Store response in cache."""
        key = self._make_key(model, prompt, system, temperature, json_mode)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache
                    (cache_key, provider, model, response, prompt_hash, created_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, provider, model, response, prompt_hash, time.time(), ttl_seconds),
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                total = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
                size = self.db_path.stat().st_size
                return {"entries": total, "size_bytes": size, "db_path": str(self.db_path)}
        except Exception:
            return {"entries": 0, "size_bytes": 0, "db_path": str(self.db_path)}

    def clear(self) -> None:
        """Clear all cached entries."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("DELETE FROM llm_cache")
                conn.commit()
        except sqlite3.Error:
            pass


class BaseLLMClient:
    """Abstract base for LLM clients."""

    PROVIDER = "base"

    def __init__(self, model: str, timeout: int = 120, cache: ResponseCache | None = None) -> None:
        self.model = model
        self.timeout = timeout
        self.cache = cache or ResponseCache()
        self._cache_hits = 0
        self._cache_misses = 0

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(
        self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        raise NotImplementedError

    def chat(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        raise NotImplementedError

    def cache_stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total": total,
        }

    def _try_cache(
        self, prompt: str, system: str, json_mode: bool, temperature: float
    ) -> str | None:
        cached = self.cache.get(self.model, prompt, system, temperature, json_mode)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        return None

    def _store_cache(
        self, prompt: str, system: str, json_mode: bool, temperature: float, response: str
    ) -> None:
        self.cache.set(self.PROVIDER, self.model, prompt, response, system, temperature, json_mode)


class OllamaClient(BaseLLMClient):
    """Ollama API client — local or remote instance."""

    PROVIDER = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        cache: ResponseCache | None = None,
    ) -> None:
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        super().__init__(model=model, timeout=timeout, cache=cache)
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
                self._available = True
                return True
        except Exception:
            self._available = False
            return False

    def generate(
        self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        cached = self._try_cache(prompt, system, json_mode, temperature)
        if cached is not None:
            return cached

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            response = result.get("response", "")
            self._store_cache(prompt, system, json_mode, temperature, response)
            return response

    def chat(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048},
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "")

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class OpenAICompatibleClient(BaseLLMClient):
    """Generic OpenAI-compatible API client.

    Supports OpenAI, Groq, OpenRouter, Together, Anyscale, and any other
    OpenAI-compatible endpoint.

    Environment:
        ROVER_LLM_API_KEY  - API key (required)
        ROVER_LLM_BASE_URL - Base URL, e.g. https://api.openai.com/v1
        ROVER_LLM_MODEL    - Model name, default gpt-4o-mini
    """

    PROVIDER = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        cache: ResponseCache | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ROVER_LLM_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ROVER_LLM_BASE_URL", "")).rstrip("/")
        model = model or os.environ.get("ROVER_LLM_MODEL", "gpt-4o-mini")
        super().__init__(model=model, timeout=timeout, cache=cache)
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key or not self.base_url:
            self._available = False
            return False
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
                self._available = True
                return True
        except Exception:
            self._available = False
            return False

    def generate(
        self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        cached = self._try_cache(prompt, system, json_mode, temperature)
        if cached is not None:
            return cached

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._chat_completion(messages, json_mode=json_mode, temperature=temperature)
        self._store_cache(prompt, system, json_mode, temperature, response)
        return response

    def chat(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        return self._chat_completion(messages, json_mode=json_mode, temperature=temperature)

    def _chat_completion(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    choice = result.get("choices", [{}])[0]
                    return choice.get("message", {}).get("content", "")
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504):
                    wait = 2 ** attempt
                    time.sleep(wait)
                    last_error = e
                    continue
                raise
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM API failed after 3 attempts: {last_error}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client.

    Uses the native Anthropic Messages API.

    Environment:
        ANTHROPIC_API_KEY - API key (required)
        ANTHROPIC_MODEL   - Model name, default claude-3-5-sonnet-20241022
    """

    PROVIDER = "anthropic"
    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        cache: ResponseCache | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        super().__init__(model=model, timeout=timeout, cache=cache)
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key:
            self._available = False
            return False
        try:
            req = urllib.request.Request(
                f"{self.BASE_URL}/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
                self._available = True
                return True
        except Exception:
            self._available = False
            return False

    def generate(
        self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        cached = self._try_cache(prompt, system, json_mode, temperature)
        if cached is not None:
            return cached

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        response = self._messages_api(system=system, messages=messages, temperature=temperature)
        self._store_cache(prompt, system, json_mode, temperature, response)
        return response

    def chat(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        return self._messages_api(system="", messages=messages, temperature=temperature)

    def _messages_api(
        self, system: str, messages: list[dict[str, str]], temperature: float
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    content_blocks = result.get("content", [])
                    return "".join(
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
                    )
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504):
                    wait = 2 ** attempt
                    time.sleep(wait)
                    last_error = e
                    continue
                raise
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Anthropic API failed after 3 attempts: {last_error}")


class LLMClient:
    """Unified LLM client that auto-detects available provider from env vars.

    Priority (user-configurable via ROVER_LLM_PROVIDER):
    1. "anthropic" - ANTHROPIC_API_KEY set
    2. "openai"   - ROVER_LLM_API_KEY + ROVER_LLM_BASE_URL set
    3. "ollama"   - localhost:11434 reachable (or OLLAMA_HOST set)
    4. None       - fall back to rule-based analysis

    Usage:
        # With Anthropic Claude
        export ANTHROPIC_API_KEY="sk-ant-..."
        rover analyze --since 2m

        # With OpenAI
        export ROVER_LLM_API_KEY="sk-..."
        export ROVER_LLM_BASE_URL="https://api.openai.com/v1"
        rover analyze --since 2m

        # With Ollama (local)
        ollama pull llama3.2
        rover analyze --since 2m

        # With Ollama (remote)
        export OLLAMA_HOST="https://your-ollama.com"
        rover analyze --since 2m
    """

    def __init__(self, preferred_provider: str | None = None) -> None:
        cache = ResponseCache()
        self._anthropic: AnthropicClient | None = None
        self._openai: OpenAICompatibleClient | None = None
        self._ollama: OllamaClient | None = None
        self._provider = "none"
        self.cache = cache

        provider = preferred_provider or os.environ.get("ROVER_LLM_PROVIDER", "").lower()

        # Try Anthropic first if key present or explicitly requested
        if provider in ("", "anthropic"):
            anthropic = AnthropicClient(cache=cache)
            if anthropic.is_available():
                self._anthropic = anthropic
                self._provider = "anthropic"
                return

        # Try OpenAI-compatible if key present or explicitly requested
        if provider in ("", "openai") and self._provider == "none":
            openai = OpenAICompatibleClient(cache=cache)
            if openai.is_available():
                self._openai = openai
                self._provider = "openai"
                return

        # Try Ollama last (always available if running)
        if provider in ("", "ollama") and self._provider == "none":
            ollama = OllamaClient(cache=cache)
            if ollama.is_available():
                self._ollama = ollama
                self._provider = "ollama"
                return

    def is_available(self) -> bool:
        return self._provider != "none"

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        if self._anthropic:
            return self._anthropic.model
        if self._openai:
            return self._openai.model
        if self._ollama:
            return self._ollama.model
        return "none"

    def generate(
        self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        if self._anthropic:
            return self._anthropic.generate(prompt, system, json_mode, temperature)
        if self._openai:
            return self._openai.generate(prompt, system, json_mode, temperature)
        if self._ollama:
            return self._ollama.generate(prompt, system, json_mode, temperature)
        return ""

    def chat(
        self, messages: list[dict[str, str]], json_mode: bool = False, temperature: float = 0.3
    ) -> str:
        if self._anthropic:
            return self._anthropic.chat(messages, json_mode, temperature)
        if self._openai:
            return self._openai.chat(messages, json_mode, temperature)
        if self._ollama:
            return self._ollama.chat(messages, json_mode, temperature)
        return ""

    def cache_stats(self) -> dict[str, Any]:
        if self._anthropic:
            return self._anthropic.cache_stats()
        if self._openai:
            return self._openai.cache_stats()
        if self._ollama:
            return self._ollama.cache_stats()
        return {"hits": 0, "misses": 0, "hit_rate": 0.0, "total": 0}

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self._provider,
            "model": self.model,
            "cache": self.cache_stats(),
            "cache_db": self.cache.stats(),
        }

    @classmethod
    def supported_providers(cls) -> dict[str, str]:
        """Return a mapping of provider names to setup instructions."""
        return {
            "anthropic": "export ANTHROPIC_API_KEY='sk-ant-...'  # Claude models",
            "openai": "export ROVER_LLM_API_KEY='sk-...'; export ROVER_LLM_BASE_URL='https://api.openai.com/v1'",
            "ollama": "Install Ollama + ollama pull llama3.2   # Free, local",
        }
