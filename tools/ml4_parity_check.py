#!/usr/bin/env python3
"""Compare strata-poly's next-token distribution with llama.cpp's for the same token ids, step by step.

    ml4_parity_check.py --ids 1,1255,2362 --steps 4 --llama http://127.0.0.1:8199 \
        --strata-log run.log            # the stderr of strata-poly --tokens ... --max-new N --top-probs 20
    ml4_parity_check.py --ids ... --llama ... --strata-cmd "CUDA_VISIBLE_DEVICES=1 ./strata-poly --model M.gguf"

llama.cpp's side is a llama-server /completion call with the ids as the prompt, greedy (temperature 0) and n_probs
top probabilities a step.  strata-poly's side is the "top id:prob ..." lines its --top-probs prints to stderr.
Both are greedy, so the two generate the same continuation while their argmax agree; the first disagreement is
reported with both distributions, and the comparison goes on over the probabilities of the ids both ranked.

Prints a table: step, strata argmax, llama argmax, p(argmax) on both sides, the top-5 overlap, and the largest
absolute difference over the shared ids.  Exit 0 when every step's argmax agrees, else 1.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import urllib.request


def llama_steps(url, ids, steps, n_probs):
    body = {"prompt": ids, "n_predict": steps, "n_probs": n_probs, "temperature": 0, "cache_prompt": False,
            "samplers": ["top_k"], "top_k": 1}
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        res = json.loads(r.read())
    return llama_steps_of(res)


def llama_steps_of(res):
    out = []
    for step in res.get("completion_probabilities", []):
        tops = step.get("top_logprobs") or step.get("probs") or []
        dist = {}
        for t in tops:
            tid = t.get("id")
            if tid is None:
                continue
            p = t.get("prob")
            if p is None and "logprob" in t:
                import math
                p = math.exp(t["logprob"])
            dist[int(tid)] = float(p)
        chosen = step.get("id")
        out.append((chosen, dist))
    return out, res.get("tokens", []), res.get("content", "")


def strata_steps(text):
    out = []
    for line in text.splitlines():
        if not line.startswith("top "):
            continue
        dist = {}
        for m in re.finditer(r"(\d+):([0-9.eE+-]+)", line):
            dist[int(m.group(1))] = float(m.group(2))
        argmax = max(dist, key=dist.get) if dist else None
        out.append((argmax, dist))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated token ids of the prompt")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--llama", help="llama-server base URL")
    ap.add_argument("--llama-json", help="a saved /completion answer (with n_probs) instead of calling the server")
    ap.add_argument("--strata-log", help="stderr of a strata-poly --top-probs run")
    ap.add_argument("--strata-cmd", help="strata-poly command (model included); --tokens/--max-new/--top-probs are added")
    ap.add_argument("--tol", type=float, default=0.05, help="largest |dp| over shared ids called a pass")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",") if x.strip()]

    if a.strata_log:
        text = open(a.strata_log).read()
    elif a.strata_cmd:
        cmd = a.strata_cmd + f" --tokens {a.ids} --max-new {a.steps} --top-probs {a.n_probs}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        text = r.stderr
        sys.stderr.write(text[-2000:])
    else:
        ap.error("--strata-log or --strata-cmd")
    s_steps = strata_steps(text)
    if not s_steps:
        print("no 'top' lines from strata-poly", file=sys.stderr)
        return 2

    if a.llama_json:
        l_steps, l_tokens, l_text = llama_steps_of(json.load(open(a.llama_json)))
    elif a.llama:
        l_steps, l_tokens, l_text = llama_steps(a.llama, ids, a.steps, a.n_probs)
    else:
        ap.error("--llama or --llama-json")
    print(f"llama.cpp continuation: {l_tokens} {l_text!r}")
    print(f"{'step':>4} {'strata':>8} {'llama':>8} {'p_strata':>9} {'p_llama':>9} {'top5':>5} {'max|dp|':>8} verdict")
    ok = True
    for i in range(min(len(s_steps), len(l_steps), a.steps)):
        s_arg, s_dist = s_steps[i]
        l_arg, l_dist = l_steps[i]
        if l_arg is None and l_dist:
            l_arg = max(l_dist, key=l_dist.get)
        shared = set(s_dist) & set(l_dist)
        maxd = max((abs(s_dist[t] - l_dist[t]) for t in shared), default=float("nan"))
        s5 = set(sorted(s_dist, key=s_dist.get, reverse=True)[:5])
        l5 = set(sorted(l_dist, key=l_dist.get, reverse=True)[:5])
        same = s_arg == l_arg
        verdict = "ok" if same and maxd <= a.tol else ("argmax differs" if not same else "probs differ")
        ok = ok and verdict == "ok"
        print(f"{i:>4} {s_arg!s:>8} {l_arg!s:>8} {s_dist.get(s_arg, float('nan')):>9.4f} "
              f"{l_dist.get(l_arg, float('nan')):>9.4f} {len(s5 & l5):>5} {maxd:>8.4f} {verdict}")
        if not same:
            print(f"     strata top: {sorted(s_dist.items(), key=lambda kv: -kv[1])[:8]}")
            print(f"     llama  top: {sorted(l_dist.items(), key=lambda kv: -kv[1])[:8]}")
            break   # the continuations part here; later steps are not comparable
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
