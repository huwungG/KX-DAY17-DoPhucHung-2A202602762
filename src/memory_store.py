from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator.

    Không cần chính xác theo tokenizer thật; đủ ổn định để so sánh hai agent.
    Quy tắc: ước lượng ~4 ký tự / token cho văn bản tiếng Việt + tiếng Anh.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# User profile store
# ---------------------------------------------------------------------------


_HEADER = "# User Profile\n\n"
_NOISE_NEAR_LOCATION = (
    "đi họp",
    "ghé thăm",
    "công tác",
    "bay ra",
    "vừa bay",
    "vừa đến",
    "đến thăm",
    "tạm qua",
    "đi ngang",
)
_NOISE_NEAR_PROFESSION = ("đùa", "ví dụ", "lấy ví dụ", "từng nói đùa", "giả sử")


def _slugify(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id.strip())
    return safe.strip("_").lower() or "anonymous"


@dataclass
class UserProfileStore:
    """Lưu trữ bền vững `User.md` cho mỗi user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slugify(user_id) / "User.md"

    def _ensure_parent(self, user_id: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _HEADER
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self._ensure_parent(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists():
            return False
        current = path.read_text(encoding="utf-8")
        if search_text not in current:
            return False
        new_text = current.replace(search_text, replacement, 1)
        path.write_text(new_text, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse các dòng `- key: value` trong User.md thành dict."""
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            payload = stripped[2:].strip()
            if ":" not in payload:
                continue
            key, _, value = payload.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                result[key] = value
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Conflict handling: nếu key đã tồn tại thì overwrite, không giữ cả cũ lẫn mới."""
        if not key or value is None or value == "":
            return
        text = self.read_text(user_id)
        new_line = f"- {key}: {value}"
        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(new_line, text, count=1)
        else:
            if not text.strip().startswith("#"):
                text = _HEADER + text
            notes_match = re.search(r"^## ", text, re.MULTILINE)
            if notes_match:
                head = text[: notes_match.start()].rstrip() + "\n"
                tail = text[notes_match.start():]
                text = head + new_line + "\n\n" + tail
            else:
                # Đảm bảo header kết thúc bằng 1 dòng trống rồi đến fact mới
                text = text.rstrip() + "\n\n" + new_line + "\n"
        self.write_text(user_id, text)


# ---------------------------------------------------------------------------
# Fact extraction (Vietnamese)
# ---------------------------------------------------------------------------


_NAME_PATTERNS = [
    re.compile(
        r"(?i:tên mình)\s+(?:là\s+|làm\s+)?([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:mình tên)\s+(?:là\s+|làm\s+)?([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:mình là)\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
]

_LOCATION_PATTERNS = [
    re.compile(
        r"(?i:mình)\s+(?:đang\s+)?ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:hiện)\s+ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:mình)\s+(?:đang\s+)?sống\s+ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:chuyển)\s+(?:ra|về|sang)\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})\s+ở",
    ),
    re.compile(
        r"(?i:chuyển)\s+(?:ra|về|sang)\s+ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:làm việc)\s+ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
    re.compile(
        r"(?i:vẫn)\s+ở\s+([A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+){0,3})",
    ),
]

_PROFESSION_KEYWORDS = (
    "backend engineer",
    "frontend engineer",
    "fullstack developer",
    "full-stack developer",
    "mlops engineer",
    "ml engineer",
    "data engineer",
    "data scientist",
    "ai engineer",
    "product manager",
    "designer",
    "lập trình viên",
    "kỹ sư",
    "sinh viên",
    "giáo viên",
    "bác sĩ",
    "freelancer",
    "developer",
    "dev",
)
_PROFESSION_RE = re.compile(
    r"(?:mình )?(?:đang |vẫn )?(?:làm|là|chuyển sang|làm việc như là|đảm nhận vai trò) ("
    + "|".join(re.escape(p) for p in _PROFESSION_KEYWORDS)
    + r")",
    re.IGNORECASE,
)

_DRINK_KEYWORDS = (
    "cà phê sữa đá",
    "cà phê đá",
    "cà phê",
    "trà sữa",
    "trà đào",
    "trà",
    "nước ép",
    "bia",
)
_FOOD_KEYWORDS = (
    "mì Quảng",
    "phở",
    "bún chả",
    "bún bò",
    "cơm tấm",
    "bánh mì",
    "bánh xèo",
)
_DRINK_RE = re.compile(
    r"(?:thích |uống |hay uống |đồ uống yêu thích là |uống )("
    + "|".join(re.escape(d) for d in _DRINK_KEYWORDS)
    + r")",
    re.IGNORECASE,
)
_FOOD_RE = re.compile(
    r"(?:thích ăn |món ăn yêu thích là |ăn )("
    + "|".join(re.escape(f) for f in _FOOD_KEYWORDS)
    + r")",
    re.IGNORECASE,
)

_STYLE_PATTERNS = [
    re.compile(
        r"trả lời (?:theo dạng |thành |bằng )?(3 bullet ngắn|3 bullet|bullet ngắn|ngắn gọn|thân thiện|chuyên nghiệp|hài hước|chi tiết)",
        re.IGNORECASE,
    ),
    re.compile(
        r"muốn bạn trả lời (ngắn gọn|chi tiết|thân thiện|hài hước|chuyên nghiệp)",
        re.IGNORECASE,
    ),
    re.compile(
        r"style trả lời (ngắn gọn|3 bullet|bullet ngắn|thân thiện|chuyên nghiệp)",
        re.IGNORECASE,
    ),
]

_TECH_KEYWORDS = (
    "Python",
    "AI",
    "AI agent",
    "AI ứng dụng",
    "RAG",
    "LLM",
    "MLOps",
    "ML",
    "agent",
    "memory system",
    "memory architecture",
    "memory compaction",
    "benchmark",
    "prompt engineering",
    "async Python",
)


_DISCOURSE_PARTICLES = (
    "nhé",
    "nhỉ",
    "nha",
    "vậy",
    "đó",
    "đấy",
    "ạ",
    "thôi",
    "hen",
    "à",
    "ha",
    "hả",
)


def _strip_particles(text: str) -> str:
    """Bỏ các discourse particle tiếng Việt ở cuối text."""
    if not text:
        return text
    out = text.strip()
    changed = True
    while changed:
        changed = False
        for p in _DISCOURSE_PARTICLES:
            if out.lower().endswith(p):
                out = out[: -len(p)].rstrip(" ,.;:")
                changed = True
    return out.strip()


def _is_question(message: str) -> bool:
    stripped = message.strip()
    if stripped.endswith("?") or stripped.endswith("?"):
        return True
    if stripped.endswith("nhỉ") or stripped.endswith("vậy") or stripped.endswith("hả"):
        return True
    return False


def _has_noise(message: str, words: tuple[str, ...]) -> bool:
    lower = message.lower()
    return any(w in lower for w in words)


def _extract_last_valid(message: str, patterns: list[re.Pattern]) -> str | None:
    """Lấy match CUỐI cùng không bị 'không còn' phủ nhận.

    Hữu ích cho các fact dạng correction: 'X chứ không còn Y' hoặc
    'OLD, nhưng thực ra NEW' — match sau cùng (loại trừ negation) là bản mới.
    """
    valid: list[str] = []
    for pattern in patterns:
        for m in pattern.finditer(message):
            preceding = message[max(0, m.start() - 30): m.start()].lower()
            if "không còn" in preceding:
                continue
            value = m.group(1).strip()
            if value:
                valid.append(value)
    return valid[-1] if valid else None


def _extract_style(message: str) -> str | None:
    """Trích response_style.

    Ưu tiên "3 bullet" nếu user có đề cập (vì đó là style cụ thể hơn
    "ngắn gọn"). Nếu không, lấy match cuối cùng từ _STYLE_PATTERNS.
    """
    if re.search(r"\b3\s*bullet(?:\s+ngắn)?\b", message, re.IGNORECASE):
        return "3 bullet ngắn"
    # Nếu có "bullet ngắn" đi kèm "ngắn gọn" thì vẫn ưu tiên
    if re.search(r"bullet\s+ngắn", message, re.IGNORECASE):
        return "bullet ngắn"
    matches: list[str] = []
    for pattern in _STYLE_PATTERNS:
        for m in pattern.finditer(message):
            value = m.group(1).strip()
            if value:
                matches.append(value)
    return matches[-1] if matches else None


def _extract_tech_interests(message: str) -> str | None:
    """Trích tech interest CHỈ khi user dùng cấu trúc 'thích X' / 'quan tâm X'.

    Tránh match nhầm khi user chỉ mention profession chứa 'ML' / 'MLOps'.
    """
    match = re.search(
        r"(?:thích|quan tâm|hứng thú với|yêu thích|thấy thú vị với|interest(?:ed)? in)\s+([^.\n]+?)(?:\.|$)",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    candidate_text = match.group(1)
    # Tách theo "và" / "," / " và "
    chunks = re.split(r"\s*(?:,|và|\+)\s*", candidate_text)
    found: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip().rstrip(".,;:")
        for kw in _TECH_KEYWORDS:
            if kw.lower() in chunk.lower() and kw not in found:
                # Phải có "thích" / "quan tâm" trước đó (đã đảm bảo ở regex trên)
                # Match nguyên kw trong chunk, không phải chỉ là ký tự overlap
                if chunk.lower().startswith(kw.lower()) or f" {kw.lower()}" in f" {chunk.lower()}":
                    found.append(kw)
                    break
    return ", ".join(found) if found else None


def extract_profile_updates_with_confidence(
    message: str,
) -> dict[str, tuple[str, float]]:
    """Trả về dict {key: (value, confidence)}.

    Confidence = 0.0 nghĩa là bỏ qua (câu hỏi hoặc nhiễu).
    Confidence >= 0.6 được phép ghi vào User.md (bonus: confidence threshold).
    """
    facts: dict[str, tuple[str, float]] = {}
    if not message or not message.strip():
        return facts
    if _is_question(message):
        return facts

    is_locational_noise = _has_noise(message, _NOISE_NEAR_LOCATION)
    is_prof_noise = _has_noise(message, _NOISE_NEAR_PROFESSION)

    # name
    name = _extract_last_valid(message, _NAME_PATTERNS)
    if name:
        facts["name"] = (_strip_particles(name), 0.9)

    # location (skip nếu có noise words; lấy match cuối cùng không bị
    # "không còn" phủ nhận để xử lý correction)
    if not is_locational_noise:
        location = _extract_last_valid(message, _LOCATION_PATTERNS)
        if location:
            location = _strip_particles(location).rstrip(".,;:")
            if location:
                facts["location"] = (location, 0.9)

    # profession (skip nếu có noise words; nếu có nhiều match thì lấy match
    # cuối cùng không bị "không còn" phủ nhận — đây là cách tự nhiên để
    # xử lý correction "Mình không còn làm X nữa, giờ chuyển sang Y")
    if not is_prof_noise:
        prof_matches = list(_PROFESSION_RE.finditer(message))
        valid_profs: list[str] = []
        for m in prof_matches:
            preceding = message[max(0, m.start() - 30): m.start()].lower()
            if "không còn" in preceding:
                # user đang phủ nhận profession này
                continue
            valid_profs.append(m.group(1).strip())
        if valid_profs:
            facts["profession"] = (valid_profs[-1], 0.9)

    # drink
    drink_match = _DRINK_RE.search(message)
    if drink_match:
        facts["favorite_drink"] = (drink_match.group(1).strip(), 0.85)

    # food
    food_match = _FOOD_RE.search(message)
    if food_match:
        facts["favorite_food"] = (food_match.group(1).strip(), 0.85)

    # style
    style = _extract_style(message)
    if style:
        facts["response_style"] = (style, 0.85)

    # technical_interest
    tech = _extract_tech_interests(message)
    if tech:
        facts["technical_interest"] = (tech, 0.75)

    return facts


def extract_profile_updates(message: str) -> dict[str, str]:
    """Backward-compatible API: chỉ trả về các fact có confidence >= 0.6."""
    return {
        key: value
        for key, (value, conf) in extract_profile_updates_with_confidence(message).items()
        if conf >= 0.6
    }


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Tóm tắt một đoạn message cũ thành text ngắn.

    Lấy first 3 + last 3 (hoặc tối đa max_items), mỗi message cắt còn ~80 ký tự.
    """
    if not messages:
        return ""
    head = messages[:3]
    tail = messages[-3:] if len(messages) > 3 else []
    chosen = head + ([m for m in tail if m not in head] if tail else [])
    chosen = chosen[:max_items]

    lines: list[str] = []
    for msg in chosen:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if not content:
            continue
        if len(content) > 80:
            content = content[:77] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Quản lý short-term memory có compact khi vượt ngưỡng token."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _total_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        msgs = thread.get("messages", [])  # type: ignore[arg-type]
        summary = thread.get("summary", "")  # type: ignore[arg-type]
        msg_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)  # type: ignore[union-attr]
        return msg_tokens + estimate_tokens(summary if isinstance(summary, str) else "")

    def _compact_once(self, thread_id: str) -> None:
        thread = self._thread(thread_id)
        msgs = list(thread.get("messages", []))  # type: ignore[arg-type]
        if len(msgs) <= self.keep_messages:
            return
        # Phần sẽ được tóm tắt: tất cả trừ keep_messages cuối
        to_summarize = msgs[: -self.keep_messages]
        keep = msgs[-self.keep_messages :]
        if not to_summarize:
            return
        existing_summary = thread.get("summary", "")  # type: ignore[arg-type]
        new_chunk = summarize_messages(to_summarize)
        if existing_summary and new_chunk:
            combined = f"{existing_summary}\n---\n{new_chunk}"
        else:
            combined = existing_summary or new_chunk
        thread["summary"] = combined
        thread["messages"] = keep
        thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        msgs = list(thread.get("messages", []))  # type: ignore[arg-type]
        msgs.append({"role": role, "content": content})
        thread["messages"] = msgs
        # Compact cho đến khi dưới ngưỡng
        safety = 0
        while self._total_tokens(thread_id) > self.threshold_tokens and safety < 50:
            before_compactions = int(thread.get("compactions", 0))
            self._compact_once(thread_id)
            after_compactions = int(thread.get("compactions", 0))
            if after_compactions == before_compactions:
                # Không thể compact thêm nữa (đã giữ tối đa)
                break
            safety += 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        # Trả về shallow copy
        return {
            "messages": list(thread.get("messages", [])),  # type: ignore[arg-type]
            "summary": thread.get("summary", ""),  # type: ignore[arg-type]
            "compactions": int(thread.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id).get("compactions", 0))

    def total_tokens(self, thread_id: str) -> int:
        return self._total_tokens(thread_id)
