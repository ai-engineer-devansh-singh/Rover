"""LLM-powered analysis modules for Rover.

Supports multiple LLM providers via user-provided API keys:
  - Anthropic Claude (ANTHROPIC_API_KEY)
  - OpenAI / OpenRouter / Groq / etc. (ROVER_LLM_API_KEY + ROVER_LLM_BASE_URL)
  - Ollama local or remote (OLLAMA_HOST, no API key needed)
"""
from rover.llm.client import (
    AnthropicClient,
    LLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    ResponseCache,
)
from rover.llm.decisions import DecisionExtractor
from rover.llm.nonce import NonceExtractor, NonceVerifier
from rover.llm.scorer import LLMScorer
from rover.llm.steering import SteeringExtractor, SteeringMetrics
from rover.llm.summarizer import SessionSummarizer

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "OllamaClient",
    "OpenAICompatibleClient",
    "ResponseCache",
    "SessionSummarizer",
    "DecisionExtractor",
    "SteeringExtractor",
    "SteeringMetrics",
    "NonceExtractor",
    "NonceVerifier",
    "LLMScorer",
]
