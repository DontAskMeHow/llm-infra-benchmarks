#!/usr/bin/env python3
"""
Canary-тест качества вывода для гибридных Qwen с CPU KV offload.

Три фазы — проверяют, что восстановление из CPU (offload restore) и длинный
контекст не портят вывод:

1. needle (cold)  — иголка в случайном филлере на заданных длинах,
                    каждый промпт уникален (без prefix cache).
2. restore        — один тяжёлый промпт 100K: первый ответ (холодный)
                    попадает в CPU L1; делаем POST /reset_prefix_cache
                    (GPU-кэш стирается, CPU остаётся); повторяем тот же
                    промпт — ответ должен прийти из CPU restore и совпасть
                    с первым. Расхождение/мусор = restore портит состояния.
3. shared-prefix  — два РАЗНЫХ вопроса на общий филлер: ответы должны быть
                    связными и отвечать на свои вопросы (симптом #4247 —
                    «многоязычный салат» на общих префиксах).

Использование:
  python canary_quality.py --base-url http://127.0.0.1:8013/v1 \\
      --model Qwen3.8-27B-native --lengths 32000,100000 --runs 1

Для фазы restore требуется VLLM_SERVER_DEV_MODE=1 на сервере.
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.request


NEEDLE_SENT = ("IMPORTANT SECRET: the special access code is "
               "YELLOW-UNICORN-7391. Remember this code.")
NEEDLE_CODE = "YELLOW-UNICORN-7391"
QUESTION = ("What is the special access code? Reply with ONLY the code, "
            "nothing else.")


def filler(n_tokens, seed, chars_per_token=2):
    rng = random.Random(seed)
    words, chars = [], 0
    while chars < n_tokens * chars_per_token:
        w = ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 12)))
        words.append(w)
        chars += len(w) + 1
        if rng.random() < 0.05:
            words[-1] += '.'
        if rng.random() < 0.02:
            words.append('\n')
    return ' '.join(words)


def chat(base_url, model, content, max_tokens=256):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip('/') + '/chat/completions', data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    msg = data["choices"][0]["message"]
    answer = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    usage = data.get("usage", {})
    return answer, dt, usage


def phase_needle(base_url, model, lengths, runs):
    print("=== Phase 1: needle (cold) ===")
    summary = []
    for L in lengths:
        ok = 0
        for i in range(runs):
            before = filler(int(L * 0.9), seed=L * 1000 + i)
            after = filler(max(int(L * 0.1) - 60, 100), seed=L * 1000 + i + 999)
            prompt = f"{before}\n\n{NEEDLE_SENT}\n\n{after}\n\n{QUESTION}"
            try:
                ans, dt, u = chat(base_url, model, prompt, max_tokens=1024)
            except Exception as e:
                print(f"  [{L}] run {i}: ERROR {e}")
                continue
            found = NEEDLE_CODE in ans
            ok += int(found)
            print(f"  [{L}] run {i}: {'PASS' if found else 'GARBAGE?'}"
                  f" | {dt:.1f}s | pt={u.get('prompt_tokens')} | "
                  f"ans={ans[:60]!r}")
        summary.append((L, ok, runs))
    return summary


def phase_restore(base_url, model):
    print("=== Phase 2: restore (cold -> reset GPU -> re-hit from CPU) ===")
    L = 100000
    before = filler(L - 1000, seed=777001)
    prompt = (f"{before}\n\n{NEEDLE_SENT}\n\n"
              "Требуется: подробно объясни, зачем нужен such access code, "
              "приведи сам код и объясни каждый символ кода."
              "\n\n" + QUESTION)
    ans1, dt1, u1 = chat(base_url, model, prompt, max_tokens=1024)
    print(f"  cold: {dt1:.1f}s pt={u1.get('prompt_tokens')} "
          f"ct={u1.get('completion_tokens')} ans1={ans1[:120]!r}")
    time.sleep(3)

    req = urllib.request.Request(
        base_url.rstrip('/').rstrip('/v1') + '/reset_prefix_cache',
        data=b'{}', method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"  reset_prefix_cache -> HTTP {r.status}")
    except Exception as e:
        print(f"  reset_prefix_cache FAILED: {e}")

    ans2, dt2, u2 = chat(base_url, model, prompt, max_tokens=1024)
    same = ans1 == ans2
    found2 = NEEDLE_CODE in ans2
    print(f"  rehit: {dt2:.1f}s pt={u2.get('prompt_tokens')} "
          f"ans2={ans2[:120]!r}")
    verdict = ("PASS (bit-identical)" if same else
               ("YELLOW (differs)" if found2 else "RED (garbage/missing code)"))
    print(f"  verdict: {verdict}")
    return {"same": same, "found2": found2, "dt_cold": dt1, "dt_rehit": dt2,
            "ans1": ans1, "ans2": ans2}


def phase_shared(base_url, model):
    print("=== Phase 3: shared prefix, two different questions ===")
    L = 90000
    shared = filler(L, seed=888001)
    q1 = f"{shared}\n\nQuestion about documents: who wrote the text above and when?"
    q2 = f"{shared}\n\nВопрос: в каком городе происходит действие описанного текста?"
    a1, dt1, _ = chat(base_url, model, q1, max_tokens=1024)
    a2, dt2, _ = chat(base_url, model, q2, max_tokens=1024)
    print(f"  q1 (en): {dt1:.1f}s -> {a1[:150]!r}")
    print(f"  q2 (ru): {dt2:.1f}s -> {a2[:150]!r}")
    verdict = ("PASS" if (a1 and a2 and a1[:40] != a2[:40]) else
               "YELLOW (identical/empty)")
    print(f"  verdict: {verdict}")
    return {"a1": a1, "a2": a2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="32000,100000")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-reset", action="store_true",
                    help="skip phase 2 (no DEV_MODE)")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    out = {
        "model": args.model,
        "needle": phase_needle(args.base_url, args.model, lengths, args.runs),
    }
    if not args.skip_reset:
        out["restore"] = phase_restore(args.base_url, args.model)
    out["shared"] = phase_shared(args.base_url, args.model)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")
    print("=== canary done ===")


if __name__ == "__main__":
    sys.exit(main())
