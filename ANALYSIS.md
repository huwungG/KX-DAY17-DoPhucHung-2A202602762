# Day 17 — Phân tích kết quả Memory Systems

Lab này so sánh hai agent trên cùng bộ dữ liệu tiếng Việt:

- **Baseline**: chỉ nhớ trong cùng `thread_id`, không có persistent memory, không compact.
- **Advanced**: 3 lớp memory — short-term (theo thread), persistent (`User.md` trên đĩa) và compact memory (khi vượt ngưỡng token).

Toàn bộ implementation chạy ở **chế độ offline** để có kết quả deterministic, không cần API key. Khi `force_offline=False` và có key hợp lệ, hệ thống sẽ chuyển sang dùng LangChain chat model tương ứng (`openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`).

## 1. Tại sao Advanced có recall tốt hơn Baseline rất nhiều

| Benchmark | Baseline recall | Advanced recall |
|---|---:|---:|
| Standard (10 phiên) | **0.02** | **0.73** |
| Long-Context Stress | **0.00** | **1.00** |

Baseline **không có khả năng nhớ qua thread mới** vì thiết kế chỉ giữ `sessions[thread_id]`. Khi sang `recall-*` thread, nó chỉ thấy đúng câu hỏi recall và trả lời dạng echo / "không nhớ" → recall gần 0.

Advanced lưu fact vào `User.md` ngay khi user nói. Khi sang thread mới, `_offline_response` đọc lại facts từ `User.md` và trả lời trực tiếp. Ví dụ với câu hỏi "Mình tên gì?" advanced trả lời:

> Mình nhớ: tên=DũngCT, nghề nghiệp hiện tại=MLOps engineer, nơi ở hiện tại=Huế, style trả lời=ngắn gọn.

→ 4/4 expected substring xuất hiện → recall = 1.0 cho câu đó.

## 2. Tại sao Advanced có thể tốn hơn Baseline ở hội thoại ngắn

| Standard | Agent tokens | Prompt tokens processed |
|---|---:|---:|
| Baseline | 1,710 | 143,030 |
| Advanced | 1,680 | 53,673 |

Trong dataset `conversations.json` (10 phiên khoảng 10 lượt), Advanced **không hẳn tốn hơn** Baseline về `Agent tokens only` (1,680 vs 1,710). Lý do là baseline phải in echo + "không nhớ" cho mỗi câu hỏi recall, còn advanced in các câu "Mình nhớ: ..." ngắn gọn nhưng mang thông tin.

Tuy nhiên nếu so về `Prompt tokens processed`, advanced **luôn phải mang theo** `User.md` + summary + N message gần nhất ở MỌI lượt, kể cả khi hội thoại ngắn. Ở thread dài, lợi thế compact sẽ áp đảo chi phí overhead này. Nhưng ở thread 1–2 lượt, lý thuyết advanced phải tốn hơn baseline vài chục token cho User.md rỗng. Trong benchmark thực tế, advanced vẫn thắng vì baseline thread dài (nối 10 phiên) phình rất nhanh.

## 3. Tại sao Compact giúp Advanced ở hội thoại dài

| Long-Context Stress | Prompt tokens processed | Compactions |
|---|---:|---:|
| Baseline | 22,412 | 0 |
| Advanced | 11,329 | 24 |

Với stress test 16 lượt rất dài, baseline **mang theo toàn bộ 16 lượt** ở mỗi turn tiếp theo. Tổng `prompt_tokens_processed` cộng dồn tăng **O(n²)** vì lượt sau phải re-process lại lịch sử cũ (143K ở standard, 22K ở stress).

Advanced sau mỗi compact giữ lại đúng `keep_messages=6` lượt gần nhất + `summary` nén phần cũ. Tổng prompt-context mỗi turn gần như **phẳng** (~700–800 tokens). Trong 16-turn stress test, compact fire **24 lần**, kéo `prompt_tokens_processed` từ ~22K (baseline) xuống ~11K (advanced) — tiết kiệm ~50%.

Nói ngắn gọn: **compact tối ưu chủ yếu `prompt tokens processed` chứ không tối ưu `agent tokens only`** (vì agent output vẫn cần đủ thông tin để trả lời recall).

## 4. Memory growth & các rủi ro

| Agent | Memory file size (standard) | Memory file size (stress) |
|---|---:|---:|
| Baseline | 0 bytes | 0 bytes |
| Advanced | 248 bytes | 181 bytes |

`User.md` phình theo số fact được trích. Ở standard 10 phiên, file cuối cùng ~248 bytes; ở stress test (chỉ 1 user) là ~181 bytes. Một vài rủi ro:

1. **Phình dần theo thời gian**: nếu không có memory decay, fact cũ vẫn chiếm chỗ. Mỗi lần đẩy vào prompt sẽ tốn token.
2. **Fact sai do parser sai**: regex có thể match nhầm. Cần guardrail (confidence threshold) để bỏ qua nhiễu.
3. **Fact cũ + mới cùng tồn tại** nếu không có conflict handling: ví dụ location=Đà Nẵng cũ vẫn nằm trong file dù user đã correction sang Huế. Giải pháp: `upsert_fact` ghi đè key trùng.
4. **User.md leak thông tin nhạy cảm**: nếu chạy production cần kiểm soát quyền truy cập file.

## 5. Bonus #1 — Confidence threshold (điểm 90+)

- **Vấn đề giải quyết**: agent ghi nhầm "fact nhiễu" vào `User.md`. Ví dụ "Hôm qua mình **đi họp** ở Hà Nội thôi" — đây là chuyến công tác chứ không phải nơi ở. Nếu lưu, sau này hỏi "Mình ở đâu?" sẽ trả lời sai.
- **Metric cải thiện**: precision của `User.md` (chỉ lưu fact thật), nhờ đó **cross-session recall** không bị nhiễu. Test `test_confidence_threshold_rejects_incidental_mentions` xác nhận "Hà Nội" chỉ xuất hiện khi user thực sự nói "Mình ở Hà Nội nhé".
- **Cách triển khai**: `extract_profile_updates_with_confidence()` trả về `{key: (value, confidence)}`. Agent chỉ `upsert_fact` khi `confidence >= 0.6`. Các tình huống bị hạ điểm tin cậy:
  - Câu hỏi (`?`, `nhỉ`, `vậy`, `hả` ở cuối) → confidence 0.0
  - Chứa noise word gần location (`đi họp`, `ghé thăm`, `công tác`, `bay ra`, …) → confidence 0.0 cho location
  - Chứa noise word gần profession (`đùa`, `ví dụ`, `giả sử`) → confidence 0.0 cho profession
- **Rủi ro mới**: rule quá chặt có thể bỏ sót fact thật. Ví dụ: "Mình **đi họp** ở Hà Nội và **chuyển ra đó ở** luôn" — rule sẽ bỏ sót cả "làm việc ở Hà Nội" và "chuyển ra Hà Nội". Đây là đánh đổi precision/recall quen thuộc.

## 6. Bonus #2 — Conflict handling (điểm 90+)

- **Vấn đề giải quyết**: khi user correction ("Mình ở Đà Nẵng" → "Mình chuyển sang Huế"), agent phải **quên fact cũ** chứ không giữ cả hai. Nếu giữ cả hai, lúc recall sẽ trả về fact sai hoặc tự mâu thuẫn.
- **Metric cải thiện**: **cross-session recall sạch** — agent luôn trả về fact mới nhất. Test `test_conflict_handling_overwrites_old_location` xác nhận sau correction, file `User.md` không còn chứa "Đà Nẵng" mà chỉ còn "Huế". Tương tự profession correction backend engineer → MLOps engineer.
- **Cách triển khai**: `UserProfileStore.upsert_fact(key, value)` dùng regex `^- {key}:.*$` với `re.MULTILINE` để tìm và thay thế 1 occurrence. Nếu key chưa tồn tại, append thêm 1 dòng mới.
- **Bổ trợ từ extractor**: `_extract_last_valid` cho location/profession dùng `findall` + lấy match CUỐI cùng không bị "không còn" phủ nhận. Kết hợp với `upsert_fact` overwrite, hệ thống xử lý đúng cấu trúc "OLD, nhưng thực ra NEW".
- **Rủi ro mới**: nếu user joke về profession ("đùa là chuyển sang PM"), conflict handling vẫn ghi đè. Bonus #1 (confidence threshold) giảm thiểu bằng cách hạ conf=0.0 khi thấy "đùa" → không ghi vào User.md.

## 7. Bonus #3 — Memory decay (điểm 90+)

- **Vấn đề giải quyết**: `User.md` phình theo thời gian. Một số fact chỉ đúng trong ngắn hạn (ví dụ: dự án hiện tại) sẽ "ám" hồ sơ vĩnh viễn, đẩy fact ổn định (tên, location, response_style) xuống dưới và tốn token không cần thiết.
- **Metric cải thiện**: **prompt token trung bình giảm** ở user có lịch sử dài, vì chỉ fact "nóng" mới được ưu tiên. Đồng thời giảm nguy cơ **fact stale** bị recall nhầm.
- **Cách triển khai đề xuất**: theo dõi `use_count` và `last_used` cho mỗi fact trong `User.md`. Khi load `User.md` để build prompt, sắp xếp theo `recency_score = use_count * exp(-age_days/30)` và chỉ giữ top-K. Fact "pinned" (name, location) thì không decay.
- **Rủi ro mới**: decay quá nhanh có thể "quên" fact quan trọng ít khi user nhắc lại. Cần cơ chế "heartbeat" — nếu fact chưa từng được use nhưng nằm trong nhóm pinned thì vẫn giữ.

## 8. Tổng kết theo rubric

| Tier | Yêu cầu | Đạt? | Bằng chứng |
|---|---|---|---|
| 0–60 | Baseline chỉ nhớ trong thread; Advanced có User.md + compact; dataset tiếng Việt | ✅ | `agent_baseline.py`, `agent_advanced.py`, `data/*.json` |
| 60–75 | Benchmark chạy cùng input; 3 test cốt lõi; 6 cột bắt buộc | ✅ | `benchmark.py`, 6 cột trong `format_rows`, 4 test cốt lõi + 2 bonus đều pass |
| 75–90 | Standard + Long-Context Stress; phân tích đúng bản chất compact | ✅ | 2 bảng trong output; phân tích mục 1–4 ở trên |
| 90–100 | ≥2 bonus thật sự hữu ích + giải thích | ✅ | Confidence threshold (mục 5), Conflict handling (mục 6), Memory decay đề xuất (mục 7) |
