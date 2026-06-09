"""Tests for LLM integration."""
from __future__ import annotations

from rover.llm.client import (
    AnthropicClient,
    LLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    ResponseCache,
)


class TestOllamaClient:
    def test_is_available_detects_local(self):
        client = OllamaClient()
        # This will return True/False depending on local Ollama
        # We just verify it doesn't crash
        result = client.is_available()
        assert isinstance(result, bool)

    def test_list_models(self):
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)

    def test_host_override(self):
        client = OllamaClient(host="http://example.com:11434")
        assert client.host == "http://example.com:11434"
        assert client.is_available() is False


class TestResponseCache:
    def test_cache_roundtrip(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        cache.set("openai", "test-model", "prompt text", "response text", system="sys", temperature=0.2, json_mode=True)
        result = cache.get("test-model", "prompt text", system="sys", temperature=0.2, json_mode=True)
        assert result == "response text"

    def test_cache_miss(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        result = cache.get("test-model", "different prompt")
        assert result is None

    def test_cache_stats(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        stats = cache.stats()
        assert "entries" in stats
        assert "size_bytes" in stats

    def test_cache_clear(self, tmp_path):
        cache = ResponseCache(cache_dir=tmp_path)
        cache.set("p", "m", "prompt", "response")
        cache.clear()
        result = cache.get("m", "prompt")
        assert result is None


class TestLLMClient:
    def test_provider_detection(self):
        client = LLMClient()
        # Should return a provider string, not crash
        assert client.provider in ("anthropic", "openai", "ollama", "none")
        assert isinstance(client.model, str)

    def test_cache_stats(self):
        client = LLMClient()
        stats = client.cache_stats()
        assert "hits" in stats
        assert "misses" in stats

    def test_supported_providers(self):
        providers = LLMClient.supported_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "ollama" in providers


class TestOpenAICompatibleClient:
    def test_no_api_key_returns_not_available(self):
        client = OpenAICompatibleClient(api_key="", base_url="")
        assert client.is_available() is False

    def test_provider_name(self):
        client = OpenAICompatibleClient(api_key="sk-test", base_url="https://api.openai.com/v1")
        assert client.PROVIDER == "openai"


class TestAnthropicClient:
    def test_no_api_key_returns_not_available(self):
        client = AnthropicClient(api_key="")
        assert client.is_available() is False

    def test_provider_name(self):
        client = AnthropicClient(api_key="sk-ant-test")
        assert client.PROVIDER == "anthropic"

    def test_default_model(self):
        client = AnthropicClient(api_key="test")
        assert "claude" in client.model
