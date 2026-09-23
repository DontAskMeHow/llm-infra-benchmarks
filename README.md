# llm-infra-benchmarks

Замеры качества и нагрузки LLM-инференса, разработанные на паре
2× NVIDIA DGX Spark (GB10) под vLLM. Модель под тестом: Qwen3.8-Flash-Next
(квант NVFP4), контекст 512K через YaRN×2 поверх нативного окна 262K.

## Что измеряется

- **Деградация длинного контекста** — пробы CE/perplexity на длинах 131K/262K
  (нативные) и 327K/393K/512K (YaRN); резкий рост PPL за 262K означает, что
  расширенный контекст ломает модель.
- **Карта NIAH** — сетка «иголка в стоге сена»: 4 длины (128K–512K) × 3 глубины
  (10/50/90%), каждая ячейка — холодный prefill с уникальным наполнителем.
- **QA-батареи** — точность на воспроизводимых срезах GPQA-Diamond / MMLU-Pro
  со сверкой с эталонными цифрами вендора (`docs/reference-benchmarks.md`).
- **Гигиена вывода** — доля пустых ответов, скор повторов, отслеживание
  обрезок (`finish=length`), битовая идентичность повторных прогонов
  CPU-восстановления KV.
- **Нагрузочный гарнесс** (`scripts/gb10/dspark_tune/`) — когорты с общим
  префиксом, вброс холодных prefill-запросов, фазовое наращивание нагрузки,
  защита по `/metrics` с авто-стопом при ожидании/вытеснениях, сбор
  `nvidia-smi dmon`.

## Топология стенда

Два узла GB10 образуют один vLLM-деплоймент: head (rank 0, OpenAI-совместимый
API на порту 8013) + worker (rank 1, `--headless`). Ключевые флаги:
`--tensor-parallel-size 2 --enable-expert-parallel`, спекулятивное
декодирование MTP (3 черновых токена), `--kv-cache-dtype fp8`,
`--enforce-eager` (CUDA-графы зависают на sm_121 с NVFP4), RDMA между узлами.
Тесты ходят с ноутбука через SSH-проброс порта
(`ssh -L 8013:127.0.0.1:8013 user@head-node`), поэтому везде по умолчанию
`http://127.0.0.1:8013/v1` — при необходимости переопределите `--base-url`.
Python 3.10+; единственная сторонняя зависимость — `openai` (в
`needle_test.py`), см. `requirements.txt`.

## Скрипты

Качество (`scripts/quality/`):

- `make_pools.py` — собирает QA-срезы (GPQA-Diamond 30, MMLU-Pro 25) с
  datasets-server Hugging Face, без токена, фиксированные смещения.
- `perplexity_probe.py --output ce.json` — пробы CE по длинам
  (`echo` + `prompt_logprobs`); `--doc file` — на реальном тексте, иначе
  наполнитель.
- `needle_map.py --output needle.json` — сетка NIAH 4×3, мышление ON, temp 0.
- `qa_battery.py --pool gpqa_diamond_30.jsonl --output results.json` —
  мышление ON, temp 1.0 / top-p 0.95, лимит max-tokens на рассуждение;
  `--workers N` — параллельные короткие вопросы;
  `--retry-truncated prev.json`.
- `report.py --run-dir <dir> --out report.md` — markdown-сводка каталога
  прогона с дельтой против эталона вендора.

Сторона кластера (`scripts/gb10/`):

- `canary_quality.py --base-url ... --model ...` — канарейка в 3 фазы:
  холодная иголка, битовая идентичность CPU-восстановления KV (нужен
  `VLLM_SERVER_DEV_MODE=1` на сервере), связность общего префикса.
- `needle_test.py --base-url ... --mode cold|warm|both` — NIAH, включая
  тёплый режим (общий префикс → путь CPU-восстановления), клиент `openai`.
- `bench_table.py a.json b.json` — сводит JSON-результаты стандарта v2 в
  выровненную таблицу; `--retire OLD NEW REASON` помечает результат
  устаревшим.
- `dspark_tune/` — `genload.py` (генератор нагрузки), `runner.py` (оркестратор
  перебора N), `parse.py` (таймлайн + метрики → `summary.csv`),
  `quick_t1.py` / `peek.py` (быстрые сводки прогонов), `pf_solo_test.py`
  (лесенка скорости prefill), `run_flood2.sh` (перебор холодного наплыва;
  рабочий каталог через `DSTUNE_DIR`), `vmparse.py` / `vmparse_h.py`
  (pretty-принтеры prometheus JSON).

Семантика метрик, грабли, происхождение пулов и операционные заметки для
этой связки железа и движка — в `docs/`.

## Лицензия

MIT — см. `LICENSE`.
