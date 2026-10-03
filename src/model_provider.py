from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Cấu hình chung cho mọi chat model provider.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.2
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "openai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "openai-compatible": "custom",
    "custom": "custom",
    "openai_compat": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",   # common typo
    "anthropics": "anthropic",  # common typo
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "or": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Chuẩn hoá tên provider và map các alias phổ biến về canonical name."""
    if value is None:
        return ""
    key = value.strip().lower().replace("_", "-").replace(" ", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    # remove non-alphanumeric dashes for fuzzy match
    squashed = key.replace("-", "")
    for alias, canonical in _ALIASES.items():
        if alias.replace("-", "") == squashed:
            return canonical
    return key


def build_chat_model(config: ProviderConfig) -> Any:
    """Khởi tạo chat model theo provider đã chọn.

    Trả về None nếu thiếu SDK hoặc thiếu API key. Việc trả về None giúp
    agent tự fallback về chế độ offline (deterministic) để benchmark/test
    vẫn chạy được mà không cần key thật.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            if not config.api_key:
                return None
            return ChatOpenAI(
                model=config.model_name or "gpt-4o-mini",
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except Exception:
            return None

    if provider == "custom":
        try:
            from langchain_openai import ChatOpenAI

            if not config.base_url:
                return None
            return ChatOpenAI(
                model=config.model_name or "gpt-4o-mini",
                temperature=config.temperature,
                api_key=config.api_key or "sk-no-key-required",
                base_url=config.base_url,
            )
        except Exception:
            return None

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not config.api_key:
                return None
            return ChatGoogleGenerativeAI(
                model=config.model_name or "gemini-1.5-flash",
                temperature=config.temperature,
                google_api_key=config.api_key,
            )
        except Exception:
            return None

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            if not config.api_key:
                return None
            return ChatAnthropic(
                model=config.model_name or "claude-3-5-sonnet-latest",
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except Exception:
            return None

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model_name or "llama3.1",
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434",
            )
        except Exception:
            return None

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI

            if not config.api_key:
                return None
            return ChatOpenAI(
                model=config.model_name or "openrouter/auto",
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )
        except Exception:
            try:
                from langchain_openrouter import ChatOpenRouter

                if not config.api_key:
                    return None
                return ChatOpenRouter(
                    model=config.model_name or "openrouter/auto",
                    temperature=config.temperature,
                    api_key=config.api_key,
                )
            except Exception:
                return None

    return None
