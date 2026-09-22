# llm-infra-benchmarks

Quality and load benchmarks for LLM inference, developed against a
2× NVIDIA DGX Spark (GB10) pair running vLLM. Model under test:
Qwen3.8-Flash-Next (NVFP4 quant), 512K context via YaRN×2 on top of a
native 262K window.

## What is measured

- **Long-context degradation** — CE/perplexity probes at 131K/262K (native)
  and 327K/393K/512K (YaRN) lengths; a sharp PPL rise past 262K means the
  extended context breaks the model.
- **NIAH map** — needle-in-a-haystack grid, 4 lengths (128K–512K) × 3 depths
  (10/50/90%), each cell a cold prefill of unique filler.
- **QA batteries** — accuracy on reproducible GPQA-Diamond / MMLU-Pro slices
  checked against vendor reference numbers (`docs/reference-benchmarks.md`).
- **Output hygiene** — empty-answer ratio, repetition score, truncated
  (`finish=length`) tracking, bit-identity of CPU KV-restore re-hits.
- **Load harness** (`scripts/gb10/dspark_tune/`) — warm shared-prefix
  cohorts, cold prefill injection, ramp phases, `/metrics` guard with
  auto-abort on waiting/preemptions, `nvidia-smi dmon` capture.

## Stand topology

Two GB10 nodes form one vLLM deployment: head (rank 0, OpenAI-compatible
API on port 8013) + worker (rank 1, `--headless`). Key flags:
`--tensor-parallel-size 2 --enable-expert-parallel`, MTP speculative
decoding (3 draft tokens), `--kv-cache-dtype fp8`, `--enforce-eager`
(CUDA graphs wedge sm_121 with NVFP4), RDMA between nodes. Tests run from
a laptop through an SSH port forward
(`ssh -L 8013:127.0.0.1:8013 user@head-node`), so all scripts default to
`http://127.0.0.1:8013/v1` — point `--base-url` anywhere else as needed.
Python 3.10+; the only third-party import is `openai` (in `needle_test.py`),
see `requirements.txt`.

## Scripts

Quality (`scripts/quality/`):

- `make_pools.py` — build QA slices (GPQA-Diamond 30, MMLU-Pro 25) from the
  Hugging Face datasets-server, no token, fixed offsets.
- `perplexity_probe.py --output ce.json` — CE probes by length
  (`echo` + `prompt_logprobs`); `--doc file` for real text, else filler.
- `needle_map.py --output needle.json` — NIAH 4×3 grid, thinking ON, temp 0.
- `qa_battery.py --pool gpqa_diamond_30.jsonl --output results.json` —
  thinking ON, temp 1.0 / top-p 0.95, max-tokens cap on reasoning;
  `--workers N` for parallel short questions; `--retry-truncated prev.json`.
- `report.py --run-dir <dir> --out report.md` — markdown summary of a run
  directory with delta vs the vendor reference.

Cluster side (`scripts/gb10/`):

- `canary_quality.py --base-url ... --model ...` — 3-phase canary: cold
  needle, CPU KV-restore bit-identity (needs `VLLM_SERVER_DEV_MODE=1` on
  the server), shared-prefix coherence.
- `needle_test.py --base-url ... --mode cold|warm|both` — NIAH including a
  warm mode (shared prefix → CPU restore path) via the `openai` client.
- `bench_table.py a.json b.json` — compare standard-v2 benchmark result
  JSONs into an aligned table; `--retire OLD NEW REASON` marks a result
  superseded.
- `dspark_tune/` — `genload.py` (load generator), `runner.py` (per-N sweep
  orchestrator), `parse.py` (timeline + metrics → `summary.csv`),
  `quick_t1.py` / `peek.py` (fast per-run summaries), `pf_solo_test.py`
  (prefill-speed ladder), `run_flood2.sh` (cold-flood sweep; working dir
  via `DSTUNE_DIR`), `vmparse.py` / `vmparse_h.py` (prometheus JSON
  pretty-printers).

Metric semantics, traps, pool provenance, and ops notes for this
hardware/engine combo: see `docs/`.

## License

MIT — see `LICENSE`.
