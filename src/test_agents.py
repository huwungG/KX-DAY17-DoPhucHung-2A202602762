from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    extract_profile_updates_with_confidence,
)


def make_config(tmp_path: Path) -> LabConfig:
    """Build một LabConfig cô lập cho test, trỏ vào tmp_path.

    Threshold được set cực thấp để compaction trigger NHANH trong unit test.
    """
    root = tmp_path.resolve()
    (root / "state" / "profiles").mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=100,
        compact_keep_messages=3,
        model=load_config().model,
        judge_model=load_config().judge_model,
        force_offline=True,
    )


# ---------------------------------------------------------------------------
# Required test #1: User.md read/write/edit
# ---------------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")

    # Initially no file → read_text returns default header
    initial = store.read_text("alice")
    assert initial.startswith("# User Profile")
    assert store.file_size("alice") == 0

    # Write some content
    path = store.write_text("alice", "# User Profile\n\n- name: Alice\n- location: Hanoi\n")
    assert path.exists()
    assert store.file_size("alice") > 0

    # Edit one fact
    changed = store.edit_text("alice", "Alice", "Alicia")
    assert changed is True
    text = store.read_text("alice")
    assert "Alicia" in text
    assert "Alice\n" not in text  # đã bị thay thế

    # Edit text không tồn tại → False, không crash
    no_change = store.edit_text("alice", "NotInFile", "Whatever")
    assert no_change is False

    # facts() parse được
    facts = store.facts("alice")
    assert facts.get("name") == "Alicia"
    assert facts.get("location") == "Hanoi"


# ---------------------------------------------------------------------------
# Required test #2: Compact trigger
# ---------------------------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    long_message = (
        "Mình vừa đọc xong một bài khá dài về memory architecture cho agent AI, "
        "trong đó tác giả phân tích sự khác biệt giữa short-term memory, persistent memory "
        "và compact memory, kèm theo các ví dụ thực tế từ hệ thống production. "
        "Đây là turn thứ {} để ép compact xảy ra trong thread này."
    )

    for i in range(8):
        agent.reply("u1", "t1", long_message.format(i))

    state = agent.compact_memory.context("t1")
    assert agent.compaction_count("t1") >= 1, (
        f"Expected at least 1 compaction, got {agent.compaction_count('t1')}"
    )
    assert len(state["messages"]) <= cfg.compact_keep_messages + 1, (
        f"After compact, kept messages should be ≤ keep_messages+1, "
        f"got {len(state['messages'])}"
    )
    # Summary phải có nội dung
    assert state["summary"], "Summary should be non-empty after compaction"


# ---------------------------------------------------------------------------
# Required test #3: Cross-session recall
# ---------------------------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)

    advanced = AdvancedAgent(config=cfg, force_offline=True)
    baseline = BaselineAgent(config=cfg, force_offline=True)

    # Feed profile facts in thread "t1"
    advanced.reply("u1", "t1", "Mình tên là DũngCT.")
    advanced.reply("u1", "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    advanced.reply("u1", "t1", "Mình thích cà phê sữa đá.")

    baseline.reply("u1", "t1", "Mình tên là DũngCT.")
    baseline.reply("u1", "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    baseline.reply("u1", "t1", "Mình thích cà phê sữa đá.")

    # Cross-session recall in thread "t2"
    adv_answer = advanced.reply("u1", "t2", "Mình tên gì?").get("content", "")
    base_answer = baseline.reply("u1", "t2", "Mình tên gì?").get("content", "")

    assert "DũngCT" in adv_answer, f"Advanced should recall name, got: {adv_answer}"
    assert "DũngCT" not in base_answer, (
        f"Baseline must NOT recall across sessions, got: {base_answer}"
    )
    assert "không nhớ" in base_answer.lower() or "không" in base_answer.lower(), (
        f"Baseline should explicitly indicate forgetting, got: {base_answer}"
    )

    # Test location recall too
    adv_loc = advanced.reply("u1", "t2", "Mình đang ở đâu?").get("content", "")
    assert "Đà Nẵng" in adv_loc, f"Advanced should recall location, got: {adv_loc}"


# ---------------------------------------------------------------------------
# Required test #4: Compact reduces prompt load on long thread
# ---------------------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    long_msg = (
        "Đây là message dài khoảng 200 ký tự để mô phỏng thread có nhiều token. "
        "Mình đang benchmark sự khác biệt giữa baseline agent và advanced agent "
        "về chi phí prompt context khi hội thoại dài. Compact memory sẽ kéo chi phí xuống."
    )
    assert len(long_msg) >= 150

    for i in range(15):
        baseline.reply("u-bench", "t-bench", long_msg + f" turn {i}")
        advanced.reply("u-bench", "t-bench", long_msg + f" turn {i}")

    base_prompt = baseline.prompt_token_usage("t-bench")
    adv_prompt = advanced.prompt_token_usage("t-bench")

    # Advanced phải compact và giữ prompt context thấp hơn baseline rõ rệt
    assert adv_prompt < base_prompt, (
        f"Advanced prompt tokens ({adv_prompt}) should be < baseline ({base_prompt})"
    )
    # Ít nhất 20% nhỏ hơn
    assert adv_prompt * 5 < base_prompt * 4, (
        f"Advanced should be ≥20% smaller than baseline; "
        f"got base={base_prompt}, adv={adv_prompt}"
    )

    # Và advanced phải thực sự compact
    assert advanced.compaction_count("t-bench") >= 1, (
        "Advanced should have compacted at least once on a 15-turn long thread"
    )


# ---------------------------------------------------------------------------
# Bonus test #1: Confidence threshold rejects incidental mentions
# ---------------------------------------------------------------------------


def test_confidence_threshold_rejects_incidental_mentions(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    # "đi họp ở" là noise → KHÔNG save location
    agent.reply("u-noise", "t-noise", "Hôm qua mình đi họp ở Hà Nội thôi.")
    facts = agent.profile_store.facts("u-noise")
    assert "location" not in facts or "Hà Nội" not in facts.get("location", ""), (
        f"Incidental 'đi họp ở' should not persist, got: {facts}"
    )

    # "Mình ở Hà Nội nhé" là assertion thật → save
    agent.reply("u-noise", "t-noise", "Mình ở Hà Nội nhé.")
    facts = agent.profile_store.facts("u-noise")
    assert facts.get("location") == "Hà Nội", (
        f"Direct assertion 'Mình ở Hà Nội' should persist, got: {facts}"
    )

    # Bonus: văn bản câu hỏi không save gì
    q_facts = extract_profile_updates("Mình tên gì?")
    assert q_facts == {}, f"Questions should not extract facts, got: {q_facts}"

    # Bonus: văn bản có "đùa" gần profession thì không save
    joke_facts = extract_profile_updates("Mình đùa với team là mình chuyển sang product manager")
    assert "profession" not in joke_facts or joke_facts.get("profession") != "product manager", (
        f"Joke should not save profession, got: {joke_facts}"
    )


# ---------------------------------------------------------------------------
# Bonus test #2: Conflict handling overwrites old value
# ---------------------------------------------------------------------------


def test_conflict_handling_overwrites_old_location(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    # Correction: Đà Nẵng → Huế
    agent.reply("u1", "t1", "Mình ở Đà Nẵng.")
    facts = agent.profile_store.facts("u1")
    assert facts.get("location") == "Đà Nẵng", facts

    agent.reply("u1", "t1", "Mình chuyển sang Huế ở.")
    facts = agent.profile_store.facts("u1")
    assert facts.get("location") == "Huế", (
        f"Conflict handling should overwrite to Huế, got: {facts}"
    )

    # File KHÔNG còn chứa "Đà Nẵng" sau khi overwrite
    text = agent.profile_store.read_text("u1")
    assert "Đà Nẵng" not in text, (
        f"Old location 'Đà Nẵng' should be gone from User.md after overwrite, got:\n{text}"
    )

    # Test profession correction: backend engineer → MLOps engineer
    agent.reply("u2", "t2", "Mình đang làm backend engineer.")
    agent.reply(
        "u2",
        "t2",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    )
    facts = agent.profile_store.facts("u2")
    assert facts.get("profession") == "MLOps engineer", (
        f"Profession correction should land on MLOps engineer, got: {facts}"
    )


# ---------------------------------------------------------------------------
# Smoke test: estimators
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_with_confidence_threshold() -> None:
    # Questions return empty
    assert extract_profile_updates_with_confidence("Bạn tên gì?") == {}
    # Direct assertions return facts
    res = extract_profile_updates_with_confidence("Mình tên là Alice.")
    assert "name" in res
    assert res["name"][0] == "Alice"
    assert res["name"][1] >= 0.6
