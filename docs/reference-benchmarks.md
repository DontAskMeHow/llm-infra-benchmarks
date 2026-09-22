# Эталонные бенчи Qwen3.8-Flash-Next (снято 2026-09-19)

Для сверки наших замеров. Сравниваем порядок и дельты (±несколько п.п. — процедурный шум),
не точное воспроизведение: производители гнали своими харнесами и параметрами.

## Исходник (BP16) — карточка Qwen/Qwen3.8-Flash-Next

Методика: temp=1.0, top_p=0.95, окно 256K; агентные — harness Claude Code/mini-SWE-agent.

| Бенч | Значение |
| --- | --- |
| GPQA Diamond | 91.7 |
| HLE | 35.9 |
| LiveCodeBench v6 | 91.9 |
| IFBench | 81.3 |
| SWE-bench Pro | 62.5 |
| DeepSWE 1.1 | 58.7 |
| SWE-bench Multilingual | 81.0 |
| Toolathlon Verified | 73.5 |

## NVFP4 (наш квант) — карточка nvidia/Qwen3.8-Flash-Next-NVFP4

Методика: vLLM, B200/B300, temp=1.0, top_p=0.95, max_new_tokens 131072, reasoning_effort=xhigh.
Базлайн сравнения — FP8 этой же модели. Вывод NVIDIA: NVFP4 ≈ FP8, дельта ±1–2.5 п.п.

| Бенч | FP8 | NVFP4 |
| --- | --- | --- |
| GPQA Diamond | 92.0 | 91.5 |
| HLE | 34.7 | 35.4 |
| IFBench | 80.5 | 81.0 |
| MMLU Pro | 77.1 | 78.3 |
| Terminal-Bench 2.1 | 83.3 | 82.9 |
| τ²-Bench Telecom | 90.8 | 90.1 |
| AA-LCR (long-context recall) | 71.9 | 74.1 |
| SciCode | 16.3 | 18.8 |
| Omniscience | 28.1 | 27.6 |

## Сообщество (NVIDIA-форумы, 2×/4× DGX Spark)

- NVFP4 vs FP8 практическая эквивалентность: SWE-bench Pro 40 задач 36–38/40 (тред #382697);
  tool-eval-bench hardmode 85–95/100 (#381228); NIAH 5/5 до ~250K (#382610/#382634).
- **YaRN-потолок (#383453, 0rand)**: качество/связность деградируют сильно; ~1.5×
  (~390K) ещё терпимо, 2× (~500K+) — тяжёлая деградация. Наша лестница 262K→512K —
  ровно эта зона → K1 (CE) и K4 (NIAH) — приоритет пакета.
- Artificial Analysis Intelligence Index v4.3: **40** (10 оценок в текущем харнессе v4.3;
  перепроверено на странице модели 19.09.2026 — #5 из 113 в классе open-weights 40–150B+,
  медиана класса 18; в v4.1 было 39.9). Сквозная шкала v4.3 для сравнения: Claude Fable 5.1 /
  GPT-6 Astra — 53, Claude Opus 5 — 51, Qwen3.8 Max — 45, GLM-5.3 — 45, Kimi K3 — 44,
  DeepSeek V4 Pro 0813 — 36, Kimi K2.7 Code — 26, Qwen3.6 35B A3B — 19.
  Таблица/график: `artifacts/gb10/quality/intel-index-2026-09-19.{md,json,svg}`.
  LiveBench: модель туда не попала (таблица от 2026-06-25).

Источники: hf.co/Qwen/Qwen3.8-Flash-Next, hf.co/nvidia/Qwen3.8-Flash-Next-NVFP4,
qwen.ai/blog?id=qwen3.8-flash-next, arxiv.org/abs/2608.30320, forums.developer.nvidia.com
треды #382610/#382634/#381897/#381228/#382697/#383453, artificialanalysis.ai/models/qwen3-8-flash-next.
