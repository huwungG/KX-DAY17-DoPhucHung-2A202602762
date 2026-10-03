from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates_with_confidence,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


_CONFIDENCE_THRESHOLD = 0.6

# Mapping từ key fact -> câu trả lời tiếng Việt khi gặp recall question
_FACT_LABELS = {
    "name": "tên",
    "location": "nơi ở hiện tại",
    "profession": "nghề nghiệp hiện tại",
    "response_style": "style trả lời",
    "favorite_drink": "đồ uống yêu thích",
    "favorite_food": "món ăn yêu thích",
    "technical_interest": "mối quan tâm kỹ thuật",
}

_RECALL_TRIGGERS = {
    "name": ("tên", "tên mình", "là ai"),
    "location": ("ở đâu", "nơi ở", "ở hiện tại", "đang ở"),
    "profession": ("nghề gì", "làm nghề", "nghề nghiệp", "làm gì"),
    "response_style": ("style trả lời", "cách trả lời", "trả lời mình thích"),
    "favorite_drink": ("đồ uống", "yêu thích uống", "thích uống"),
    "favorite_food": ("món ăn", "yêu thích ăn", "thích ăn"),
    "technical_interest": ("quan tâm", "kỹ thuật chính", "hứng thú"),
}


def _detect_recall_keys(message: str) -> list[str]:
    """Trả về danh sách fact keys mà user đang hỏi tới."""
    msg_lower = message.lower()
    matched: list[str] = []
    # Nếu là câu tổng hợp ("tóm tắt", "là ai", "nhắc lại") → trả về tất cả
    if any(k in msg_lower for k in ("tóm tắt", "là ai", "nhắc lại")):
        return list(_FACT_LABELS.keys())
    for key, cues in _RECALL_TRIGGERS.items():
        if any(cue in msg_lower for cue in cues):
            matched.append(key)
    return matched


class AdvancedAgent:
    """Agent B: 3 lớp memory.

    1. Short-term: thread state trong CompactMemoryManager (giữ N message gần nhất).
    2. Persistent: User.md trên đĩa, sống qua mọi thread.
    3. Compact: tự động nén khi tổng token vượt ngưỡng.

    Bonus:
    - Confidence threshold: chỉ ghi fact khi conf >= 0.6 (bỏ qua câu hỏi, nhiễu).
    - Conflict handling: `upsert_fact` ghi đè key cũ, không giữ đồng thời.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline or self.config.force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    # ----- public API -----

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ----- offline implementation -----

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Bước 1: extract facts với confidence
        facts_with_conf = extract_profile_updates_with_confidence(message)
        for key, (value, conf) in facts_with_conf.items():
            if conf >= _CONFIDENCE_THRESHOLD:
                # upsert_fact đã handle conflict (overwrite key trùng)
                self.profile_store.upsert_fact(user_id, key, value)

        # Bước 2: append user message vào compact memory (có thể trigger compact)
        self.compact_memory.append(thread_id, "user", message)

        # Bước 3: estimate prompt context = User.md + summary + kept messages
        prompt_ctx_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_ctx_tokens
        )

        # Bước 4: sinh reply (đọc từ User.md để cross-thread recall hoạt động)
        reply_text = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # Bước 5: cập nhật agent token counter
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(
            reply_text
        )

        return {
            "content": reply_text,
            "agent_tokens": estimate_tokens(reply_text),
            "prompt_tokens": prompt_ctx_tokens,
            "compaction_count": self.compact_memory.compaction_count(thread_id),
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        facts_with_conf = extract_profile_updates_with_confidence(message)
        for key, (value, conf) in facts_with_conf.items():
            if conf >= _CONFIDENCE_THRESHOLD:
                self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_ctx_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_ctx_tokens
        )

        try:
            profile_text = self.profile_store.read_text(user_id)
            kept = self.compact_memory.context(thread_id).get("messages", [])
            history = "\n".join(f"{m['role']}: {m['content']}" for m in kept)
            prompt = f"{profile_text}\n\nLịch sử gần đây:\n{history}\n\nUser: {message}"
            response = self.langchain_agent.invoke(prompt)
            reply_text = getattr(response, "content", str(response))
        except Exception:
            reply_text = self._offline_response(user_id, thread_id, message)

        self.compact_memory.append(thread_id, "assistant", reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(
            reply_text
        )
        return {
            "content": reply_text,
            "agent_tokens": estimate_tokens(reply_text),
            "prompt_tokens": prompt_ctx_tokens,
            "compaction_count": self.compact_memory.compaction_count(thread_id),
        }

    # ----- helpers -----

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Prompt context = User.md + summary + N message gần nhất (đã compact)."""
        user_md_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        summary_tokens = estimate_tokens(summary if isinstance(summary, str) else "")
        msgs = ctx.get("messages", [])
        kept_tokens = sum(
            estimate_tokens(m.get("content", "")) for m in msgs  # type: ignore[union-attr]
        )
        return user_md_tokens + summary_tokens + kept_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        recall_keys = _detect_recall_keys(message)

        if recall_keys and facts:
            parts: list[str] = []
            for key in recall_keys:
                if key in facts:
                    label = _FACT_LABELS.get(key, key)
                    parts.append(f"{label}={facts[key]}")
            if parts:
                joined = ", ".join(parts)
                return f"Mình nhớ: {joined}."

        if recall_keys:
            return "Mình chưa ghi nhận thông tin này trong hồ sơ, nhắc lại giúp mình nhé."

        # Không phải câu hỏi recall → acknowledgment có tham chiếu memory
        if facts:
            return "Đã ghi nhận, mình cập nhật hồ sơ người dùng."
        return "Đã nhận, mình sẽ ghi nhận thông tin quan trọng vào hồ sơ."

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        try:
            model = build_chat_model(self.config.model)
            if model is None:
                return None
            return model
        except Exception:
            return None
