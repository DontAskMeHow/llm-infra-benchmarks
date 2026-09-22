#!/usr/bin/env python3
"""
Сводка прогона model-quality: markdown-таблица результатов + сравнение с эталоном.

  python report.py --run-dir artifacts/gb10/quality/runs/2026-09-19 --out report.md
"""

import argparse
import json
import os
import re
import sys

REF = {
    "gpqa_diamond": {"value": 91.5, "unit": "acc %", "source":
                     "nvidia/Qwen3.8-Flash-Next-NVFP4 (эталон кванта; Qwen 91.7)"},
    "mmlu_pro": {"value": 78.3, "unit": "acc %", "source":
                 "nvidia/Qwen3.8-Flash-Next-NVFP4 (эталон кванта)"},
    "ifbench": {"value": 81.0, "unit": "acc %", "source": "NVFP4-карточка"},
    "hle": {"value": 35.4, "unit": "acc %", "source": "NVFP4-карточка"},
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize(run_dir):
    out = []
    out.append("# Сводка качественного прогона\n")
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(run_dir, name)
        try:
            data = load_json(path)
        except Exception as e:
            out.append(f"- {name}: не прочитался ({e})\n")
            continue
        if "cells" in data:  # needle
            passed = data.get("passed", 0)
            total = data.get("total", 0)
            out.append(f"## NIAH-карта ({name})\n\n"
                       f"pass {passed}/{total}\n")
            for c in data["cells"]:
                icon = "PASS" if c.get("found") else "FAIL"
                out.append(f"- {c.get('length', 0) // 1000}K @ {int(c.get('depth', 0) * 100)}%"
                           f": {icon} ({c.get('latency_s')}s)\n")
        elif "rows" in data and data["rows"] and "ce" in data["rows"][0]:  # CE
            out.append(f"## CE-пробы ({name})\n\n| Длина | CE | PPL | t |\n")
            out.append("| --- | --- | --- | --- |\n")
            for r in data["rows"]:
                if "error" in r:
                    out.append(f"| {r['length']} | ERROR {r['error']} | | |\n")
                else:
                    out.append(f"| {r['length']} | {r['ce']} | {r['ppl']} | "
                               f"{r['latency_s']}s |\n")
        elif "accuracy" in data:  # qa_battery
            acc = (data["accuracy"] or 0) * 100
            wall = data.get("wall_seconds", 0)
            p = data.get("params", {})
            key = re.sub(r"_\d+$", "", os.path.splitext(name)[0])
            ref = REF.get(key)
            out.append(f"## QA: {name}\n\n"
                       f"accuracy **{acc:.1f}%** ({data['correct']}/{data['n']}), "
                       f"wall {wall / 60:.1f} мин, empty {data.get('empty_ratio')}\n")
            if ref:
                delta = acc - ref["value"]
                out.append(f"эталон {ref['value']} ({ref['source']}) — "
                           f"**дельта {delta:+.1f} п.п.**\n")
            out.append(f"\nпараметры: {p}\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = summarize(args.run_dir)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
