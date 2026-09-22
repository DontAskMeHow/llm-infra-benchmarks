import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genload import stream_chat, cold_messages

class TL:
    def __init__(self):
        self.rows = []
    def put(self, r):
        self.rows.append(r)

ep = "http://127.0.0.1:8011/v1"
model = "DeepSeek-V4-Pro-0813"
print("words  prompt_tok  ttft_s  prefill_tps  total_s  ct")
for words in [2500, 5000, 10000, 20000, 40000]:
    for rep in range(2):
        tl = TL()
        msgs = cold_messages("pftest-%d-%d" % (words, rep), rep, words)
        t0 = time.time()
        stream_chat(ep, model, msgs, 32, "pftest", "cold", None, tl, 1.0)
        r = tl.rows[0]
        if r.get("status") != "ok":
            print(words, "ERR", r.get("status"), str(r.get("err", ""))[:80])
            continue
        u = r.get("usage", {})
        pt = u.get("prompt_tokens", 0)
        ct = u.get("completion_tokens", 0)
        ttft = r["ts_first"] - r["ts_sent"]
        total = r["ts_last"] - r["ts_sent"]
        print("%d  %d  %.3f  %.0f  %.3f  %d" % (
            words, pt, ttft, pt / ttft if ttft else 0, total, ct))
        time.sleep(1)
