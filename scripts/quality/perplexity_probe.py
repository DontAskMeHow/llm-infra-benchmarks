#!/usr/bin/env python3
"""
CE-пробы (перплексия) по длинам — градусник деградации от YaRN×2.

Снимается через /v1/completions с echo+prompt_logprobs: один запрос на длину.
Длины: 131K/262K (нативно) и 327K/393K/512K (YaRN). Резкий рост PPL за 262K =
расширение контекста ломает модель.

Текст документа: --doc <файл> (реальный текст) или без него — детерминированный
филлер (случайные слова). Для кривой деградации хватает филлера (сравниваются длины
между собой, не с чужими PPL).

  python perplexity_probe.py --base-url http://127.0.0.1:8013/v1 --output ce.json
"""

import argparse
import json
import math
import random
import string
import sys
import time
import urllib.request

LENGTHS = [131072, 262144, 327680, 393216, 524288]
CHAR_PER_TOKEN = 2


def filler(tokens, seed):
    rng = random.Random(seed)
    words, chars = [], 0
    while chars < tokens * CHAR_PER_TOKEN:
        w = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 12)))
        words.append(w)
        chars += len(w) + 1
        if rng.random() < 0.05:
            words[-1] += '.'
        if rng.random() < 0.02:
            words.append('\n')
        if rng.random() < 0.005:
            words.append('\n\nParagraph.\n\n')
    return ' '.join(words)


def completions(base_url, model, prompt):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "echo": True,
        "prompt_logprobs": 1,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip('/') + '/completions', data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    return data, dt


def _chosen_logprob(pos):
    """Логпроб выбранного токена из одной позиции prompt_logprobs.

    vLLM отдаёт словарь {token_id: {"logprob": float, "rank": int, ...}, ...},
    где выбранный токен помечен rank == 1. Прочие формы — как fallback."""
    if isinstance(pos, list):
        pos = pos[0] if pos else None
    if pos is None:
        return None
    if not isinstance(pos, dict):
        return pos if isinstance(pos, (int, float)) else None
    if "logprob" in pos:
        return pos.get("logprob")
    ranked = [(i.get("rank"), i.get("logprob")) for i in pos.values()
              if isinstance(i, dict) and isinstance(i.get("logprob"), (int, float))]
    if not ranked:
        return None
    for rank, lp in ranked:
        if rank == 1:
            return lp
    ranked.sort(key=lambda r: (r[0] if isinstance(r[0], int) else 10**9))
    return ranked[0][1]


def ce_from(data):
    """CE по prompt-токенам из echo prompt_logprobs (первый элемент None)."""
    choice = data["choices"][0]
    ppl = choice.get("prompt_logprobs")
    if not ppl:
        raise RuntimeError("prompt_logprobs отсутствуют в ответе (echo не поддержан "
                           "этой версией vLLM) — использовать оконную методику")
    vals = []
    for lp in ppl:
        v = _chosen_logprob(lp)
        if isinstance(v, (int, float)):
            vals.append(v)
    if not vals:
        raise RuntimeError("нет logprob-значений")
    ce = -sum(vals) / len(vals)
    return ce, math.exp(ce), len(vals)


def run(args):
    doc = None
    if args.doc:
        with open(args.doc, encoding="utf-8") as f:
            doc = f.read()
    else:
        print("Нет --doc: беру детерминированный филлер "
              "(кривая деградации валидна, абсолютные PPL — нет)")
    rows = []
    for i, length in enumerate(args.lengths):
        text = doc[: length * CHAR_PER_TOKEN] if doc else filler(length, 777000 + length)
        print(f"[{i+1}/{len(args.lengths)}] length={length:>7} ...", end=" ", flush=True)
        try:
            data, dt = completions(args.base_url, args.model, text)
            ce, ppl, n = ce_from(data)
            usage = data.get("usage", {})
            rows.append({"length": length, "ce": round(ce, 4), "ppl": round(ppl, 2),
                         "tokens": n, "latency_s": round(dt, 1),
                         "prompt_tokens": usage.get("prompt_tokens")})
            print(f"CE {ce:.4f} | PPL {ppl:.2f} | {n} tok | {dt:.0f}s")
        except Exception as e:
            rows.append({"length": length, "error": str(e)})
            print("ERROR", e)
    out = {"model": args.model, "lengths": args.lengths,
           "doc": args.doc or "deterministic-filler", "rows": rows}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.output}")
    print("SUMMARY (length -> PPL):")
    for r in rows:
        if "error" not in r:
            print(f"  {r['length']:>7}: CE {r['ce']:.4f} PPL {r['ppl']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--doc", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",")]
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
