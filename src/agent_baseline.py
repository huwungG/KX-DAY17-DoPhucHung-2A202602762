from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


_RECALL_CUES = (
    "tên gì",
    "tên mình",
    "nhắc lại tên",
    "ở đâu",
    "nơi ở",
    "nghề gì",
    "làm nghề",
    "nghề nghiệp",
    "yêu thích",
    "thích uống",
    "đồ uống",
    "món ăn",
    "style trả lời",
    "cách trả lời",
    "quan tâm",
    "kỹ thuật chính",
    "là ai",
    "tóm tắt",
    "nhắc lại",
)


class BaselineAgent:
    """Agent A: chỉ có within-session memory. Không có User.md, không compact.

    Mục tiêu: làm mốc so sánh công bằng cho AdvancedAgent.
    - Mỗi thread có session riêng, không nhìn thấy thread khác.
    - `prompt_tokens_processed` tăng tuyến tính theo độ dài thread (đây là
      điểm benchmark muốn phơi bày để chứng minh giá trị của compact memory).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline or self.config.force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    # ----- public API -----

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Live path nếu có model thật
        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                # Fallback về offline nếu live lỗi (ví dụ thiếu key runtime)
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline không có persistent memory
        return 0

    # ----- offline implementation -----

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        # 1. Ghi user message
        session.messages.append({"role": "user", "content": message})

        # 2. Đo prompt tokens processed = TỔNG tất cả message đã từng thấy
        #    (bao gồm user + assistant). Đây là chi phí "mang theo" qua các
        #    lượt, tăng tuyến tính theo thread.
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        # 3. Sinh reply deterministic. Nếu là câu hỏi recall → "không nhớ".
        msg_lower = message.lower()
        if any(cue in msg_lower for cue in _RECALL_CUES) and any(
            q in message for q in ("?", "nhỉ", "vậy", "gì", "gì?", "nào", "sao")
        ):
            reply_text = "Tôi không nhớ thông tin này trong session này."
        else:
            snippet = message.strip()
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            reply_text = f"Đã nhận: {snippet}"

        # 4. Ghi assistant message + cập nhật token usage
        session.messages.append({"role": "assistant", "content": reply_text})
        session.token_usage += estimate_tokens(reply_text)

        return {
            "content": reply_text,
            "agent_tokens": estimate_tokens(reply_text),
            "prompt_tokens": prompt_tokens,
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]
        session.messages.append({"role": "user", "content": message})
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in session.messages)
        prompt_tokens = estimate_tokens(history_text)
        try:
            response = self.langchain_agent.invoke(history_text)
            reply_text = getattr(response, "content", str(response))
        except Exception:
            reply_text = "Tôi không nhớ thông tin này trong session này."
        session.messages.append({"role": "assistant", "content": reply_text})
        session.token_usage += estimate_tokens(reply_text)
        session.prompt_tokens_processed += prompt_tokens
        return {
            "content": reply_text,
            "agent_tokens": estimate_tokens(reply_text),
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        if self.force_offline:
            return None
        try:
            model = build_chat_model(self.config.model)
            if model is None:
                return None
            from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
            return model
        except Exception:
            return None
