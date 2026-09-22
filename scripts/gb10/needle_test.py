#!/usr/bin/env python3
"""
Needle-in-a-Haystack тест для DeepSeek-V4-Pro.

Вставляет кодовую фразу в случайный текст на 90% длины контекста,
затем спрашивает модель назвать эту фразу. Проверяет корректность ответа
на разных длинах контекста.

Два режима:
  cold — каждый промпт уникальный (нет prefix cache hit)
  warm — общий префикс + уникальный суффикс (prefix cache / CPU restore path)

Использование:
  python needle_test.py --base-url http://localhost:8011/v1 --mode cold
  python needle_test.py --base-url http://localhost:8011/v1 --mode warm
  python needle_test.py --base-url http://localhost:8011/v1 --mode both

Запуск с ноды, где крутится vLLM-контейнер, через docker exec:
  docker exec vllm-deepseek-v4-pro python /scripts/needle_test.py ...

Или удалённо через SSH + curl (скрипт сам делает API-вызовы).
"""

import argparse
import json
import random
import string
import sys
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

NEEDLE_CODE = "YELLOW-UNICORN-7391"
NEEDLE_SENTENCE = f"IMPORTANT SECRET: The special code mentioned in this document is {NEEDLE_CODE}. Remember this code."

QUESTION = "What is the special code mentioned in the document? Reply with ONLY the code, nothing else."

# Approximate chars per token for random lowercase text with DeepSeek tokenizer
# Measured: 16K target → 17,455 actual tokens at multiplier 2
CHARS_PER_TOKEN = 2

CONTEXT_LENGTHS = [16_000, 32_000, 64_000, 128_000, 256_000]


def generate_filler(num_tokens: int, seed: int) -> str:
    """Generate random filler text of approximately num_tokens tokens."""
    rng = random.Random(seed)
    # Generate random words of varying length
    words = []
    chars_needed = num_tokens * CHARS_PER_TOKEN
    chars_generated = 0
    while chars_generated < chars_needed:
        word_len = rng.randint(2, 12)
        word = ''.join(rng.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        chars_generated += word_len + 1  # +1 for space
        # Add punctuation and newlines occasionally for realism
        if rng.random() < 0.05:
            words[-1] += '.'
        if rng.random() < 0.02:
            words.append('\n')
    return ' '.join(words)


def build_prompt(context_length: int, mode: str, run_index: int,
                 shared_prefix: Optional[str] = None) -> str:
    """
    Build a prompt with needle inserted at 90% of context.

    For cold mode: entirely unique filler (seed = context_length * 1000 + run_index)
    For warm mode: shared prefix + unique suffix after needle
    """
    needle_position = int(context_length * 0.9)
    filler_before_tokens = needle_position
    filler_after_tokens = context_length - needle_position - 50  # ~50 tok for needle

    if mode == "cold":
        seed = context_length * 1000 + run_index
        filler_before = generate_filler(filler_before_tokens, seed)
        filler_after = generate_filler(max(filler_after_tokens, 100), seed + 999)
    elif mode == "warm":
        if shared_prefix is None:
            # First call — generate shared prefix
            shared_prefix = generate_filler(filler_before_tokens,
                                            seed=context_length * 2000)
        filler_before = shared_prefix
        # Unique suffix per run
        filler_after = generate_filler(max(filler_after_tokens, 100),
                                       seed=context_length * 2000 + run_index + 500)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    prompt = f"{filler_before}\n\n{NEEDLE_SENTENCE}\n\n{filler_after}"
    return prompt, shared_prefix if mode == "warm" else None


def run_needle_test(client: OpenAI, model: str, context_length: int,
                    mode: str, num_runs: int = 2) -> dict:
    """Run needle test at a specific context length."""
    results = []
    shared_prefix = None

    # Warmup for warm mode
    if mode == "warm":
        _, shared_prefix = build_prompt(context_length, "warm", run_index=0)
        # Send warmup request to populate prefix cache
        warmup_prompt, shared_prefix = build_prompt(context_length, "warm",
                                                     run_index=-1,
                                                     shared_prefix=shared_prefix)
        try:
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": warmup_prompt + "\n\n" + QUESTION}
                ],
                max_tokens=500,
                temperature=0,
            )
        except Exception as e:
            print(f"  Warmup failed: {e}")

    for i in range(num_runs):
        prompt, shared_prefix = build_prompt(context_length, mode, run_index=i,
                                              shared_prefix=shared_prefix)
        full_message = prompt + "\n\n" + QUESTION

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_message}],
                max_tokens=500,
                temperature=0,
            )
            elapsed = time.time() - t0
            answer = response.choices[0].message.content.strip()
            found = NEEDLE_CODE in answer
            results.append({
                "run": i,
                "answer": answer[:200],
                "found_needle": found,
                "latency_s": round(elapsed, 2),
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            })
            status = "✅ PASS" if found else "❌ FAIL (GARBAGE?)"
            print(f"  Run {i}: {status} | answer={answer[:80]!r} | {elapsed:.1f}s"
                  f" | prompt={results[-1]['prompt_tokens']}")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "run": i,
                "answer": f"ERROR: {e}",
                "found_needle": False,
                "latency_s": round(elapsed, 2),
                "prompt_tokens": None,
                "completion_tokens": None,
            })
            print(f"  Run {i}: ❌ ERROR | {e} | {elapsed:.1f}s")

    passed = sum(1 for r in results if r["found_needle"])
    return {
        "context_length": context_length,
        "mode": mode,
        "passed": passed,
        "total": num_runs,
        "all_pass": passed == num_runs,
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack test for DSV4Pro")
    parser.add_argument("--base-url", default="http://localhost:8011/v1",
                        help="vLLM API base URL")
    parser.add_argument("--model", default="DeepSeek-V4-Pro-0813",
                        help="Model name")
    parser.add_argument("--mode", choices=["cold", "warm", "both"], default="both",
                        help="Test mode")
    parser.add_argument("--lengths", type=str, default=None,
                        help="Comma-separated context lengths (default: 16000,32000,64000,128000,256000)")
    parser.add_argument("--runs", type=int, default=2,
                        help="Runs per length per mode (default: 2)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else CONTEXT_LENGTHS
    modes = ["cold", "warm"] if args.mode == "both" else [args.mode]

    all_results = []
    max_clean_length = {}  # mode -> max length with all passes

    print(f"=== Needle-in-a-Haystack Test ===")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Needle: {NEEDLE_CODE}")
    print(f"Lengths: {lengths}")
    print(f"Modes: {modes}")
    print(f"Runs per test: {args.runs}")
    print()

    for mode in modes:
        max_clean_length[mode] = 0
        print(f"--- Mode: {mode} ---")
        for length in lengths:
            print(f"\nContext: {length:,} tokens ({mode})")
            result = run_needle_test(client, args.model, length, mode, args.runs)
            all_results.append(result)

            if result["all_pass"]:
                max_clean_length[mode] = length
            else:
                print(f"  ⚠️  FAILURES at {length:,} — stopping this mode")
                # Continue to next lengths to see if it's intermittent
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode in modes:
        print(f"  {mode}: max clean length = {max_clean_length[mode]:,} tokens")
        for r in all_results:
            if r["mode"] == mode:
                status = "✅" if r["all_pass"] else "❌"
                print(f"    {status} {r['context_length']:>7,} tok: "
                      f"{r['passed']}/{r['total']} passed")
    print()

    # Output JSON
    output = {
        "model": args.model,
        "needle": NEEDLE_CODE,
        "modes": modes,
        "max_clean_length": max_clean_length,
        "results": all_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        # Print compact summary for pasting into RESULTS_dsv4pro.md
        for mode in modes:
            mcl = max_clean_length[mode]
            print(f"needle ({mode}): max clean = {mcl:,} tokens")

    return 0 if all(r["all_pass"] for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
