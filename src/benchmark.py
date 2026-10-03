from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_conversations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Conversations file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def recall_points(answer: str, expected: list[str]) -> float:
    """Đếm bao nhiêu chuỗi trong `expected` xuất hiện trong `answer` (case-insensitive)."""
    if not expected:
        return 1.0
    if not answer:
        return 0.0
    lower_answer = answer.lower()
    hits = sum(1 for token in expected if token.lower() in lower_answer)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Điểm chất lượng heuristic: cân bằng giữa recall và độ dài hợp lý."""
    if not answer:
        return 0.0
    score = 0.0
    length = len(answer)
    # Điểm thưởng cho độ dài hợp lý
    if 20 <= length <= 400:
        score += 0.4
    elif 10 <= length < 20 or 400 < length <= 800:
        score += 0.2
    # Điểm recall
    score += 0.6 * recall_points(answer, expected)
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def _fresh_agent(agent_cls, config: LabConfig, threshold: int | None = None):
    """Tạo agent mới với state sạch để tránh leakage giữa các benchmark run."""
    cfg = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=config.state_dir,
        compact_threshold_tokens=threshold
        if threshold is not None
        else config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
        force_offline=True,
    )
    return agent_cls(config=cfg, force_offline=True)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
) -> BenchmarkRow:
    """Chạy agent qua toàn bộ conversation + recall_questions trong thread mới."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    recall_count = 0
    total_compactions = 0
    final_memory_bytes = 0

    for conv in conversations:
        user_id = conv.get("user_id", "default")
        # Tất cả convs của cùng user dùng chung thread "main" để phản ánh
        # đúng việc user tương tác liên tục qua nhiều phiên.
        thread_id = "main"
        for turn in conv.get("turns", []):
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_compactions = max(
                total_compactions,
                result.get("compaction_count", agent.compaction_count(thread_id)),
            )

        # Hỏi recall_questions trong thread MỚI để test cross-session recall
        for i, rq in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"recall-{conv.get('id', i)}-{i}"
            question = rq.get("question", "")
            expected = rq.get("expected_contains", [])
            result = agent.reply(user_id, recall_thread, question)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            answer = result.get("content", "")
            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            recall_count += 1

        # Track memory file size cho user
        try:
            mem = agent.memory_file_size(user_id)
            final_memory_bytes = max(final_memory_bytes, mem)
        except Exception:
            pass

    avg_recall = total_recall / recall_count if recall_count else 0.0
    avg_quality = total_quality / recall_count if recall_count else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=final_memory_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _try_tabulate(rows: list[list[str]], headers: list[str]) -> str | None:
    try:
        from tabulate import tabulate  # type: ignore

        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        return None


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            f"{r.compactions:,}",
        ]
        for r in rows
    ]
    table = _try_tabulate(data, headers)
    if table is not None:
        return table
    # Hand-rolled fallback
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *data)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    body_lines = [
        " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) for row in data
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def _delta_line(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    return (
        f"[Delta] prompt tokens (Adv - Base) = "
        f"{advanced.prompt_tokens_processed - baseline.prompt_tokens_processed:+,}, "
        f"compactions = {advanced.compactions - baseline.compactions:+,}, "
        f"recall = {advanced.recall_score - baseline.recall_score:+.2f}, "
        f"quality = {advanced.response_quality - baseline.response_quality:+.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_section(
    section_name: str,
    dataset_path: Path,
    config: LabConfig,
    threshold_override: int | None = None,
) -> tuple[list[BenchmarkRow], BenchmarkRow, BenchmarkRow]:
    conversations = load_conversations(dataset_path)

    base = _fresh_agent(BaselineAgent, config, threshold=threshold_override)
    adv = _fresh_agent(AdvancedAgent, config, threshold=threshold_override)

    base_row = run_agent_benchmark("Baseline", base, conversations)
    adv_row = run_agent_benchmark("Advanced", adv, conversations)

    print(f"## {section_name}")
    print()
    print(f"[Dataset] {dataset_path.name} ({len(conversations)} conversation(s))")
    if threshold_override is not None:
        print(f"[Note] compact_threshold_tokens overridden to {threshold_override} for this run.")
    print()
    print(format_rows([base_row, adv_row]))
    print()
    print(_delta_line(base_row, adv_row))
    print()
    return [base_row, adv_row], base_row, adv_row


def main() -> None:
    import sys
    import io
    # Đảm bảo output là UTF-8 (Windows console mặc định là cp1252)
    if hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception:
            pass

    config = load_config(Path(__file__).resolve().parent.parent)
    print("# Day 17 Memory Systems - Benchmark Report")
    print()
    print(
        f"[Config] repo={config.base_dir} | compact_threshold={config.compact_threshold_tokens} "
        f"| keep_messages={config.compact_keep_messages} | force_offline={config.force_offline}"
    )
    print()

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    if not standard_path.exists():
        print(f"!! Missing standard dataset: {standard_path}")
    else:
        _run_section("Standard Benchmark", standard_path, config)

    if not stress_path.exists():
        print(f"!! Missing long-context dataset: {stress_path}")
    else:
        # Dùng threshold thấp hơn cho stress test để chắc chắn compact trigger
        stress_threshold = min(600, config.compact_threshold_tokens)
        _run_section(
            "Long-Context Stress Benchmark",
            stress_path,
            config,
            threshold_override=stress_threshold,
        )


if __name__ == "__main__":
    main()
