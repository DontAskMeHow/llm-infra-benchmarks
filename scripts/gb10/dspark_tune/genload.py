#!/usr/bin/env python3
"""Генератор синтетической нагрузки для тюнинга DSpark DSV4-Pro.

Запускается на любой машине с доступом к API модели:
  python3 genload.py --outdir /tmp/dspark-tune/run1 \\
      --phase t1 --n 4 --seconds 45 --max-tokens 2500

Только stdlib. Пишет в outdir:
  timeline.jsonl — по строке на запрос (ts_sent/ts_first/ts_last/kind/usage)
  events.jsonl   — phase_start / phase_steady / phase_stop
  metrics.raw    — снэпшоты /metrics раз в 1 c (строки "@<unix_ts> <сэмпл>")
  dmon.raw       — nvidia-smi dmon -s u -d 1 -o DT (если задано --with-dmon)
  monitor.jsonl  — чтения guard-метрик каждые 2 c

Виды нагрузки:
  warm — общий префикс (prefix-cache) + уникальный хвост; слотов n, каждый
         слот держит ровно один запрос (закончился -> новый), K-зона задаётся N.
  cold — уникальный случайный промпт длины --cold-words (холодный prefill),
         впрыскивается с темпом --cold-rate запросов/с.
  ramp — то же, но темп растёт по экспоненте каждые --ramp-step с.

Guard: если num_requests_waiting > --abort-waiting непрерывно --abort-secs с
или num_preemptions растёт — выставляется abort, фаза останавливается.
run_id передаётся в заголовке x-dspark-test каждого запроса.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = ("lambda tensor kernel neutron orbit quartz velvet amber cobalt "
         "drifter meadow harbor forest beacon vector molar fusion whetstone "
         "granite recall bounce candle galaxy timber scaffold patron helix "
         "comet glacier summit meadow tarantula octave pivot quark zenith").split()
SENT = ("The {1} {2} settles behind the {3} ridge while the {4}-ton vessel "
        "keeps a steady {5} through the {6} channel.")


def words_str(n, rng):
    out, k = [], len(WORDS)
    while len(out) < n:
        out.append(rng.choice(WORDS))
    return " ".join(out)


def warm_messages(run_id, slot, seq, prefix_words):
    pfx = words_str(prefix_words, random.Random(run_id))  # фиксирован per run
    tail = ("\n\nStandalone tail #{}-{}-{}: the quick brown fox issues a "
            "nonce marker.".format(run_id, slot, seq))
    content = pfx + tail
    return [{"role": "user", "content": content}]


def cold_messages(run_id, seq, cold_words):
    rng = random.Random("{}-{}".format(run_id, seq))
    content = words_str(cold_words, rng)
    return [{"role": "user", "content": content}]


class Timeline(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.q = path, __import__("queue").Queue()
        self.start()

    def run(self):
        with open(self.path, "a", encoding="utf-8") as fh:
            while True:
                rec = self.q.get()
                if rec is None:
                    return
                fh.write(json.dumps(rec) + "\n")

    def put(self, rec):
        self.q.put(rec)

    def close(self):
        self.q.put(None)
        self.join(timeout=5)


class Guard(threading.Thread):
    def __init__(self, ep_metrics, outdir, abort_waiting, abort_secs, ev):
        super().__init__(daemon=True)
        self.murl = ep_metrics
        self.mf = open(os.path.join(outdir, "metrics.raw"), "a", encoding="utf-8")
        self.ef = open(os.path.join(outdir, "events.jsonl"), "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.abort_waiting, self.abort_secs = abort_waiting, abort_secs
        self.aborted, self.stop = False, threading.Event()
        self.ev = ev
        self.above_since, self.prev_preemptions = None, None

    def run(self):
        sel = re.compile(
            r"^(vllm:(iteration_tokens_total_(sum|count)|num_requests_running|"
            r"num_requests_waiting|num_requests_waiting_by_reason|"
            r"kv_cache_usage_perc|num_preemptions_total|"
            r"spec_decode_num_accepted_tokens_per_pos_total|"
            r"spec_decode_num_draft_tokens_total|"
            r"spec_decode_num_accepted_tokens_total|"
            r"prefix_cache_(queries|hits)_total|prompt_tokens_total|"
            r"generation_tokens_total|estimated_read_bytes_per_gpu_total))")
        tick = 0
        while not self.stop.is_set():
            t0 = time.time()
            try:
                with urllib.request.urlopen(self.murl, timeout=5) as r:
                    body = r.read().decode("utf-8", "replace")
                now = time.time()
                with self.lock:
                    for line in body.splitlines():
                        if sel.match(line):
                            self.mf.write("@%.3f %s\n" % (now, line))
                    self.mf.flush()
                waiting = preempt = 0.0
                for line in body.splitlines():
                    if line.startswith("vllm:num_requests_waiting{"):
                        waiting = float(line.rsplit(" ", 1)[1])
                    elif line.startswith("vllm:num_preemptions_total{"):
                        preempt = float(line.rsplit(" ", 1)[1])
                if self.prev_preemptions is None:
                    self.prev_preemptions = preempt
                preempt_grew = preempt > self.prev_preemptions
                self.prev_preemptions = preempt
                if self.abort_waiting > 0 and waiting > self.abort_waiting:
                    if self.above_since is None:
                        self.above_since = now
                else:
                    self.above_since = None
                if (self.above_since is not None
                        and now - self.above_since > self.abort_secs) or preempt_grew:
                    self.aborted = True
                    rec = {"ts": now, "event": "abort",
                           "waiting": waiting, "preempt": preempt}
                    self.ev.append(rec)
                    self.ef.write(json.dumps(rec) + "\n")
                    self.ef.flush()
                self.ev.append({"ts": now, "event": "monitor",
                                "waiting": waiting, "preempt": preempt})
            except Exception as e:  # сеть/метрики недоступны — не фатально
                self.ev.append({"ts": time.time(), "event": "monitor_err",
                                "err": str(e)})
            # защита от спирали записи events: пишем только раз в 2 с
            tick += 1
            time.sleep(max(0.0, t0 + 1.0 - time.time()))

    def finish(self):
        self.stop.set()
        self.mf.close()
        self.ef.close()


def stream_chat(ep, model, messages, max_tokens, run_id, kind, slot,
                timeline, temperature):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(ep + "/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "x-dspark-test": run_id,
        "x-test-kind": kind,
    })
    rec = {"run_id": run_id, "kind": kind, "slot": slot,
           "ts_sent": time.time(), "status": "ok"}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            first_ts = last_ts = None
            usage = {}
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ts = time.time()
                if first_ts is None and chunk.get("choices") \
                        and isinstance(chunk["choices"][0].get("delta"), dict) \
                        and chunk["choices"][0]["delta"]:
                    first_ts = ts
                if chunk.get("usage"):
                    usage = chunk["usage"]
                last_ts = ts
        rec.update(ts_first=first_ts, ts_last=last_ts, usage=usage)
        if first_ts is None:
            rec["status"] = "no_tokens"
    except urllib.error.HTTPError as e:
        rec.update(status="http%d" % e.code, err=e.read(400).decode("utf-8", "replace"))
    except Exception as e:
        rec.update(status="err", err=str(e))
    timeline.put(rec)


def warm_slot(ep, model, run_id, slot, prefix_words, max_tokens,
              timeline, stop, gap=0.0):
    seq = 0
    while not stop.is_set():
        stream_chat(ep, model,
                    warm_messages(run_id, slot, seq, prefix_words),
                    max_tokens, run_id, "warm", slot, timeline,
                    temperature=0.7)
        seq += 1
        if gap:
            time.sleep(gap)


def cold_injector(ep, model, run_id, cold_words, rate, max_tokens, timeline, stop):
    delay = 1.0 / rate
    seq = 0
    next_at = time.time()
    while not stop.is_set():
        now = time.time()
        if now < next_at:
            time.sleep(min(next_at - now, 0.5))
            continue
        next_at += delay
        threading.Thread(
            target=stream_chat,
            args=(ep, model, cold_messages(run_id, seq, cold_words),
                  max_tokens, run_id, "cold", None, timeline, 1.0),
            daemon=True).start()
        seq += 1


def ramp_injector(ep, model, run_id, cold_words, rate0, factor, step,
                  max_tokens, timeline, stop):
    delay, seq, t0 = 1.0 / rate0, 0, time.time()
    next_at = t0
    while not stop.is_set():
        now = time.time()
        if now < next_at:
            time.sleep(min(next_at - now, 0.5))
            continue
        next_at += delay
        threading.Thread(
            target=stream_chat,
            args=(ep, model, cold_messages(run_id, seq, cold_words),
                  max_tokens, run_id, "cold", None, timeline, 1.0),
            daemon=True).start()
        seq += 1
        if (now - t0) // step > (next_at - delay - t0) // step:
            delay = max(delay / factor, 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="DeepSeek-V4-Pro-0813")
    ap.add_argument("--phase", choices=["t1", "t2", "t3", "micro"], required=True)
    ap.add_argument("--n", type=int, default=4, help="warm-когорта (t1/t2/t3)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--prefix-words", type=int, default=5200, help="~4K токенов")
    ap.add_argument("--cold-words", type=int, default=11000, help="~8K токенов")
    ap.add_argument("--cold-rate", type=float, default=0.5, help="cold/с (t2)")
    ap.add_argument("--ramp-rate", type=float, default=0.25)
    ap.add_argument("--ramp-factor", type=float, default=1.4)
    ap.add_argument("--ramp-step", type=float, default=60)
    ap.add_argument("--cold-max-tokens", type=int, default=128)
    ap.add_argument("--abort-waiting", type=int, default=8)
    ap.add_argument("--abort-secs", type=int, default=10)
    ap.add_argument("--with-dmon", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    run_id = a.run_id or time.strftime("run%m%d%H%M%S")
    os.makedirs(a.outdir, exist_ok=True)
    tl_path = os.path.join(a.outdir, "timeline.jsonl")
    timeline = Timeline(tl_path)
    ev = []
    events_path = os.path.join(a.outdir, "events.jsonl")

    def evt(name, **kw):
        rec = {"ts": time.time(), "event": name, "run_id": run_id}
        rec.update(kw)
        ev.append(rec)
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    guard = Guard(a.endpoint.rsplit("/v1", 1)[0] + "/metrics", a.outdir,
                  a.abort_waiting, a.abort_secs, ev)
    guard.start()
    dmon = None
    if a.with_dmon:
        dmon = __import__("subprocess").Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-o", "DT"],
            stdout=open(os.path.join(a.outdir, "dmon.raw"), "w"), stderr=__import__("subprocess").DEVNULL)

    stop = threading.Event()
    print("[{}] phase={} n={} seconds={} outdir={}".format(
        run_id, a.phase, a.n, a.seconds, a.outdir), flush=True)
    evt("phase_start", phase=a.phase, n=a.n, seconds=a.seconds)

    if a.phase == "micro":
        stream_chat(a.endpoint, a.model, warm_messages(run_id, 0, 0, a.prefix_words),
                    a.max_tokens, run_id, "micro", 0, timeline, 0.0)
    elif a.phase == "t1":
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.n) as ex:
            futs = [ex.submit(warm_slot, a.endpoint, a.model, run_id, s,
                              a.prefix_words, a.max_tokens, timeline, stop)
                    for s in range(a.n)]
            deadline = time.time() + a.seconds
            while time.time() < deadline and not guard.aborted:
                time.sleep(1)
            stop.set()
            for f in futs:
                f.result(timeout=120)
    elif a.phase in ("t2", "t3"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.n + 2) as ex:
            futs = [ex.submit(warm_slot, a.endpoint, a.model, run_id, s,
                              a.prefix_words, a.max_tokens, timeline, stop)
                    for s in range(a.n)]
            if a.phase == "t2":
                inj = threading.Thread(target=cold_injector, args=(
                    a.endpoint, a.model, run_id, a.cold_words, a.cold_rate,
                    a.cold_max_tokens, timeline, stop), daemon=True)
            else:
                inj = threading.Thread(target=ramp_injector, args=(
                    a.endpoint, a.model, run_id, a.cold_words, a.ramp_rate,
                    a.ramp_factor, a.ramp_step, a.cold_max_tokens,
                    timeline, stop), daemon=True)
            inj.start()
            deadline = time.time() + a.seconds
            while time.time() < deadline and not guard.aborted:
                time.sleep(1)
            stop.set()
            inj.join(timeout=60)
            for f in futs:
                f.result(timeout=120)

    evt("phase_stop", aborted=guard.aborted)
    guard.finish()
    if dmon:
        dmon.terminate()
    timeline.close()
    time.sleep(1)
    print("[{}] done aborted={}".format(run_id, guard.aborted), flush=True)


if __name__ == "__main__":
    main()
