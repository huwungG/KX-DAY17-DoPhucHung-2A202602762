from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Cấu hình chia sẻ cho toàn bộ lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    force_offline: bool = True


_PROVIDER_ENV_KEYS: dict[str, tuple[str, str, str | None]] = {
    # canonical provider -> (api_key_env, base_url_env, default_model)
    "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL", "gpt-4o-mini"),
    "gemini": ("GEMINI_API_KEY", None, "gemini-1.5-flash"),
    "anthropic": ("ANTHROPIC_API_KEY", None, "claude-3-5-sonnet-latest"),
    "ollama": (None, "OLLAMA_BASE_URL", "llama3.1"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter/auto"),
}


def _build_provider_from_env(provider: str, model_name: str | None) -> ProviderConfig:
    canonical = normalize_provider(provider)
    api_key_env, base_url_env, default_model = _PROVIDER_ENV_KEYS.get(
        canonical, (None, None, "gpt-4o-mini")
    )
    api_key = os.environ.get(api_key_env) if api_key_env else None
    base_url = os.environ.get(base_url_env) if base_url_env else None
    return ProviderConfig(
        provider=canonical,
        model_name=model_name or os.environ.get("LLM_MODEL") or default_model,
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        api_key=api_key,
        base_url=base_url,
    )


def _has_any_provider_key() -> bool:
    for canonical, (api_key_env, _, _) in _PROVIDER_ENV_KEYS.items():
        if api_key_env and os.environ.get(api_key_env):
            return True
    return False


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load config từ .env + biến môi trường, fallback offline nếu không có key."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env nếu có (không bắt buộc)
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except Exception:
        # python-dotenv không có sẵn, vẫn chạy được bằng env hiện tại
        pass

    # Tạo state/ nếu chưa có
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    compact_threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.environ.get("COMPACT_KEEP_MESSAGES", "6"))

    provider_name = os.environ.get("LLM_PROVIDER", "openai")
    model_name = os.environ.get("LLM_MODEL")
    primary = _build_provider_from_env(provider_name, model_name)
    judge = _build_provider_from_env(
        os.environ.get("JUDGE_PROVIDER", provider_name),
        os.environ.get("JUDGE_MODEL", model_name),
    )

    # force_offline: mặc định True nếu không phát hiện key của provider nào
    explicit_offline = os.environ.get("FORCE_OFFLINE")
    if explicit_offline is not None:
        force_offline = explicit_offline.strip().lower() in ("1", "true", "yes")
    else:
        force_offline = not _has_any_provider_key()

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=primary,
        judge_model=judge,
        force_offline=force_offline,
    )
