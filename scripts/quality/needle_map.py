#!/usr/bin/env python3
"""
NIAH-карта (needle-in-a-haystack) для пилота Qwen3.8-Flash-Next на паре GB10.

Сетка: 3 длины × 3 глубины (10/50/90)% по умолчанию. Каждый запрос — отдельный
холодный префилл уникального филлера (seed от длины и глубины). thinking ON, temp=0.

Запуск — с ноутбука через форвард:
  python needle_map.py --base-url http://127.0.0.1:8013/v1 --output needle.json
"""

import argparse
import json
import random
import re
import string
import sys
import time
import urllib.request

NEEDLE_CODE = "YELLOW-UNICORN-7391"
NEEDLE_SENT = (f"IMPORTANT SECRET: the special code mentioned in this document is "
               f"{NEEDLE_CODE}. Remember this code.")
QUESTION = ("What is the special code mentioned in the document? "
            "Reply with ONLY the code, nothing else.")
CHAR_PER_TOKEN = 2  # калибровка для филлера случайных слов (как в needle_test.py)


def filler(n_tokens, seed):
    rng = random.Random(seed)
    words, chars = [], 0
    while chars < n_tokens * CHAR_PER_TOKEN:
        w = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 12)))
        words.append(w)
        chars += len(w) + 1
        if rng.random() < 0.05:
            words[-1] += '.'
        if rng.random() < 0.02:
            words.append('\n')
    return ' '.join(words)


def build_prompt(length, depth):
    before = int(length * depth)
    after = max(length - before - 60, 100)
    seed = length * 1000 + int(depth * 100)
    return (f"{filler(before, seed)}\n\n{NEEDLE_SENT}\n\n"
            f"{filler(after, seed + 999)}\n\n{QUESTION}")


def chat(base_url, model, prompt, max_tokens=512):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip('/') + '/chat/completions', data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    msg = data["choices"][0]["message"] or {}
    content = (msg.get("content") or "").strip()
    usage = data.get("usage", {})
    return content, dt, usage


def run(args):
    cells = []
    passed = 0
    for length in args.lengths:
        for depth in args.depths:
            prompt = build_prompt(length, depth)
            label = f"{length / 1000:.0f}K@{int(depth * 100)}%"
            try:
                ans, dt, u = chat(args.base_url, args.model, prompt)
            except Exception as e:
                print(f"[{label}] ERROR {e}")
                cells.append({"length": length, "depth": depth,
                              "found": False, "error": str(e)})
                continue
            found = NEEDLE_CODE in ans
            passed += int(found)
            cells.append({"length": length, "depth": depth, "found": found,
                          "answer": ans[:120], "latency_s": round(dt, 1),
                          "prompt_tokens": u.get("prompt_tokens"),
                          "completion_tokens": u.get("completion_tokens")})
            print(f"[{label}] {'PASS' if found else 'FAIL'} | {dt:.0f}s | "
                  f"pt={u.get('prompt_tokens')} | ans={ans[:60]!r}")
    out = {"model": args.model, "needle": NEEDLE_CODE,
           "lengths": args.lengths, "depths": args.depths,
           "passed": passed, "total": len(cells), "cells": cells}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")
    print(f"SUMMARY: {passed}/{len(cells)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--lengths", default="128000,262144,384000,512000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    args.depths = [float(x) for x in args.depths.split(",")]
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
