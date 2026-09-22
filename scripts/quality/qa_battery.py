#!/usr/bin/env python3
"""
QA-батарея качества: прогон пула вопросов через OpenAI-совместимый API vLLM.

Пул — JSONL: {"id": str, "prompt": str, "answer": str, "type": "mcq|free|code"}.
Парсинг thinking-aware: читаем только message.content (vLLM отделяет reasoning);
если content пуст — фолбэк на хвост reasoning_content.

Параметры прогона из канона (зафиксированы 2026-09-19):
  --temperature 1.0 --top-p 0.95 --max-tokens 2048 (кап мышления), thinking ON.

Использование:
  python qa_battery.py --base-url http://127.0.0.1:8013/v1 \
      --model Qwen3.8-Flash-Next-NVFP4 --pool gpqa_diamond_20.jsonl \
      --output results.json

Короткие независимые вопросы можно гонять параллельно (потоковый пул, дефолт 1):
  ... --pool gpqa_diamond_30.jsonl --workers 4
Длинноконтекстные тесты (CE/NIAH) и ведж параллелить нельзя — только по одному.
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.request

CHOICE_RE = re.compile(r"\b([A-J])\b")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def chat(base_url, model, prompt, max_tokens, temperature, top_p, thinking):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip('/') + '/chat/completions', data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    choice = data["choices"][0]
    msg = choice["message"] or {}
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
    usage = data.get("usage", {})
    return {"content": content, "reasoning": reasoning,
            "finish": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(dt, 2)}


def parse_answer(text, qtype):
    if not text:
        return None
    t = text.strip()
    if qtype == "mcq":
        m = BOXED_RE.search(t)
        if m and m.group(1).strip().upper() in "ABCDEFGHIJ":
            return m.group(1).strip().upper()
        m = CHOICE_RE.search(t)
        if m:
            return m.group(1).upper()
        return None
    if qtype in ("free", "code"):
        m = BOXED_RE.search(t)
        if m:
            return m.group(1).strip()
        num = NUM_RE.findall(t)
        if num:
            return num[-1]
        return t.splitlines()[-1].strip() if t.splitlines() else None
    return t[:200]


def normalize(s):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def repetition_score(text):
    """Максимальная доля повторяющихся unigram — грубый детектор зацикливания."""
    if not text:
        return 0.0
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return 0.0
    top = max((words.count(w) for w in set(words)), default=0)
    return round(top / len(words), 3)


def filter_truncated(items, prev_path):
    """Оставляет вопросы, не доведённые до конца в предыдущем прогоне (finish == "length")."""
    with open(prev_path, encoding="utf-8") as f:
        prev = json.load(f)
    trunc_ids = {it.get("id") for it in prev.get("items", [])
                 if it.get("finish") == "length"}
    kept = [it for it in items if it.get("id") in trunc_ids]
    return kept, prev.get("n", 0), len(trunc_ids)


def ask_one(args, it):
    """Один вопрос → строка результата; отдельно флаги empty/truncated для сводки."""
    qtype = it.get("type", "mcq")
    try:
        r = chat(args.base_url, args.model, it["prompt"], args.max_tokens,
                 args.temperature, args.top_p, args.thinking)
    except Exception as e:
        return {"id": it.get("id"), "error": str(e)}, 0, 0
    got = parse_answer(r["content"] or (r["reasoning"][-500:]
                                        if r["reasoning"] else ""), qtype)
    ok = (r["finish"] != "length" and bool(got)
          and normalize(got) == normalize(it["answer"]))
    row = {"id": it.get("id"), "ok": ok, "got": got,
           "answer": it.get("answer"), "finish": r["finish"],
           "latency_s": r["latency_s"],
           "prompt_tokens": r["prompt_tokens"],
           "completion_tokens": r["completion_tokens"],
           "repetition": repetition_score(r["content"])}
    return row, int(not r["content"]), int(r["finish"] == "length")


def run(args):
    items = []
    with open(args.pool, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]
    retry_note = ""
    if args.retry_truncated:
        items, prev_n, trunc_n = filter_truncated(items, args.retry_truncated)
        retry_note = (f" | retry-truncated from {args.retry_truncated} "
                      f"(было {prev_n}, трункейтов {trunc_n}, взято {len(items)})")
    print(f"Pool: {args.pool} | questions: {len(items)} | "
          f"max_tokens={args.max_tokens} temp={args.temperature} "
          f"thinking={args.thinking} workers={args.workers}{retry_note}")
    if not items:
        print("Трункейтов нет — нечего дозапрашивать.")
        return None
    results = []
    correct, empty, truncated, t0_all = 0, 0, 0, time.time()
    if args.workers <= 1:
        for i, it in enumerate(items):
            row, e, t = ask_one(args, it)
            results.append(row)
            correct += int(row.get("ok") or 0)
            empty += e
            truncated += t
            if row.get("error"):
                print(f"[{i+1}/{len(items)}] ERROR {row['error']}")
            else:
                print(f"[{i+1}/{len(items)}] {'PASS' if row['ok'] else 'FAIL'} | "
                      f"ct={row['completion_tokens']} | {row['latency_s']}s | "
                      f"got={row['got']!r}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ask_one, args, it): i for i, it in enumerate(items)}
            results = [None] * len(items)
            empties = [0] * len(items)
            truncs = [0] * len(items)
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                row, empties[i], truncs[i] = fut.result()
                results[i] = row
                if row.get("error"):
                    print(f"[{i+1}/{len(items)}] ERROR {row['error']}")
                else:
                    print(f"[{i+1}/{len(items)}] {'PASS' if row['ok'] else 'FAIL'} | "
                          f"ct={row['completion_tokens']} | {row['latency_s']}s | "
                          f"got={row['got']!r}")
        correct = sum(int(r.get("ok") or 0) for r in results)
        empty = sum(empties)
        truncated = sum(truncs)
    total = time.time() - t0_all
    summ = {
        "pool": args.pool, "n": len(items), "correct": correct,
        "accuracy": round(correct / len(items), 3) if items else None,
        "empty_ratio": round(empty / len(items), 3) if items else None,
        "truncated": truncated,
        "wall_seconds": round(total, 1),
        "params": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                   "top_p": args.top_p, "thinking": args.thinking,
                   "workers": args.workers},
        "items": results,
    }
    if args.retry_truncated:
        summ["retry_from"] = args.retry_truncated
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")
    print(f"SUMMARY: {correct}/{len(items)} correct "
          f"({summ['accuracy']}) | wall {total:.0f}s | empty {empty} | "
          f"truncated {truncated}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="потоковый пул запросов (по умолчанию 1 — последовательно)")
    ap.add_argument("--retry-truncated", default=None,
                    help="дозапросить только вопросы с finish=length из JSON предыдущего прогона")
    ap.add_argument("--thinking", dest="thinking", action="store_true",
                    default=True, help="thinking включён (по умолчанию)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="выключить мышление (классический MCQ-режим)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers должен быть >= 1")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
