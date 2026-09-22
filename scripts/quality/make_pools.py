#!/usr/bin/env python3
"""
Сборка пулов QA из открытых источников (datasets-server, без токена).

- GPQA-Diamond (MC-версия): зеркало hendrydong/gpqa_diamond_mc (split test) —
  оригинал Idavidrein/gpqa на HF под гейтом; формат задач \boxed{A-D}.
- MMLU-Pro: TIGER-Lab/MMLU-Pro (split test, 10 вариантов, answer_index).

Все срезки — с фиксированными смещениями (воспроизводимость), метаданные при записи.

Использование:
  python make_pools.py [--out-dir artifacts/gb10/quality/datasets]
"""

import argparse
import json
import os
import re
import sys
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCDEFGHIJ"


def fetch(dataset, config, split, offset, length):
    url = (f"{ROWS}?dataset={urllib.parse.quote(dataset)}&config="
           f"{urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
           f"&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=90) as r:
        data = json.loads(r.read())
    return [row["row"] for row in data["rows"]]


def make_gpqa(out_dir):
    rows = fetch("hendrydong/gpqa_diamond_mc", "default", "test",
                 offset=60, length=30)
    items = []
    for i, row in enumerate(rows):
        sol = row.get("solution") or ""
        m = re.search(r"\\boxed\{([A-J])\}", sol, re.IGNORECASE)
        letter = m.group(1).upper() if m else "?"
        items.append({
            "id": f"gpqa-d-{i:03d}",
            "prompt": row["problem"].strip(),
            "answer": letter,
            "type": "mcq",
            "domain": row.get("domain", ""),
        })
    path = os.path.join(out_dir, "gpqa_diamond_30.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"GPQA-D: {len(items)} вопросов -> {path}")
    print(" ".join(f"{it['id']}:{it['answer']}" for it in items[:5]), "...")
    return path


def make_mmlu(out_dir):
    exemplars = fetch("TIGER-Lab/MMLU-Pro", "default", "test",
                      offset=100, length=5)
    rows = fetch("TIGER-Lab/MMLU-Pro", "default", "test", offset=200, length=25)

    def fmt_options(r):
        return "\n".join(f"({L}) {opt}" for L, opt in zip(LETTERS, r["options"]))

    demo = ""
    for r in exemplars:
        demo += (f"EXAMPLE\nQuestion: {r['question']}\nOptions:\n{fmt_options(r)}\n"
                 f"Answer: {r['answer']}\n\n")
    items = []
    for i, r in enumerate(rows):
        prompt = (f"{demo}Question: {r['question']}\nOptions:\n{fmt_options(r)}\n"
                  f"Please answer with ONLY the letter of the correct option.")
        items.append({
            "id": f"mmlu-pro-{r['question_id']}",
            "prompt": prompt,
            "answer": str(r["answer"]).strip().upper(),
            "type": "mcq",
            "category": r.get("category", ""),
        })
    path = os.path.join(out_dir, "mmlu_pro_25.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"MMLU-Pro: {len(items)} вопросов (5-shot демо вшиты) -> {path}")
    print(" ".join(f"{it['id']}:{it['answer']}" for it in items[:5]), "...")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/gb10/quality/datasets")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_gpqa(args.out_dir)
    make_mmlu(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
