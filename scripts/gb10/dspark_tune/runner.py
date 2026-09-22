#!/usr/bin/env python3
"""Драйвер фаз T1/T2/T3 — оркестрация genload.py по зонам.

Пример:
  python3 runner.py --outdir /tmp/dspark-tune/20260912-A --label A \
      --n-list "1,2,3,4,5,7,8,12,16,20" --seconds 45 --with-dmon
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "genload.py")

# ~время жизни одного тёплого запроса по зоне (solo decode быстрый)
MAXTOK = {1: 3500, 2: 2500, 3: 2000, 4: 1800, 5: 1600, 7: 1400,
          8: 1200, 12: 1000, 16: 900, 20: 800, 24: 700}


def run_phase(outdir, run_id, args):
    cmd = [sys.executable, GEN,
           "--outdir", os.path.join(outdir, run_id),
           "--run-id", run_id,
           "--phase", args.phase,
           "--seconds", str(args.seconds),
           "--abort-waiting", str(args.abort_waiting),
           "--abort-secs", str(args.abort_secs)]
    if args.n is not None:
        cmd += ["--n", str(args.n),
                "--max-tokens", str(MAXTOK.get(args.n, 1500))]
    if args.prefix_words:
        cmd += ["--prefix-words", str(args.prefix_words)]
    if args.cold_words:
        cmd += ["--cold-words", str(args.cold_words)]
    if args.cold_rate:
        cmd += ["--cold-rate", str(args.cold_rate)]
    if args.cold_max_tokens:
        cmd += ["--cold-max-tokens", str(args.cold_max_tokens)]
    if args.ramp_rate:
        cmd += ["--ramp-rate", str(args.ramp_rate)]
    if args.ramp_factor:
        cmd += ["--ramp-factor", str(args.ramp_factor)]
    if args.ramp_step:
        cmd += ["--ramp-step", str(args.ramp_step)]
    if args.with_dmon:
        cmd.append("--with-dmon")
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--phase", choices=["t1", "t2", "t3", "micro"], required=True)
    ap.add_argument("--n-list", default="1,2,3,4,5,7,8,12,16,20")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--gap", type=float, default=8)
    ap.add_argument("--prefix-words", type=int, default=5200)
    ap.add_argument("--cold-words", type=int, default=11000)
    ap.add_argument("--cold-rate", type=float, default=None)
    ap.add_argument("--cold-max-tokens", type=int, default=128)
    ap.add_argument("--ramp-rate", type=float, default=0.25)
    ap.add_argument("--ramp-factor", type=float, default=1.4)
    ap.add_argument("--ramp-step", type=float, default=60)
    ap.add_argument("--abort-waiting", type=int, default=8)
    ap.add_argument("--abort-secs", type=int, default=10)
    ap.add_argument("--with-dmon", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "manifest.json"), "w") as fh:
        json.dump(vars(a), fh, indent=2, default=str)

    nlist = [int(x) for x in a.n_list.split(",") if x.strip()]
    start = time.time()
    codes = []
    for n in (nlist if a.phase == "t1" else ([a.n] if a.n else nlist)):
        a.n = n
        rid = "{}-{:02d}".format(a.label, n)
        codes.append(run_phase(a.outdir, rid, a))
        time.sleep(a.gap)

    print("runner {} done in {:.0f}s codes={} outdir={}".format(
        a.label, time.time() - start, codes, a.outdir), flush=True)


if __name__ == "__main__":
    main()
