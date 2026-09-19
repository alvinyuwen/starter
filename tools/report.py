"""Fetch a Dryft run and decompose it into prefill vs decode cost.

The report gives tokens/sec, TTFT and TPOT per workload. Those three pin down
where the time actually goes:

    total    = TTFT + (output_tokens - 1) * TPOT
    prefill ~= TTFT            (one prompt pass plus the first token)
    decode  ~= total - TTFT

which is the number that decides whether the next change should attack the
prompt pass or the per-step path.

Usage:
    python tools/report.py            # latest run
    python tools/report.py <run-id>
"""

import json
import os
import sys
import urllib.request

BASE = "https://htn.dryft.ai"


def api(path):
    token = os.environ.get("DRYFT_TOKEN", "")
    if not token:
        raise SystemExit("set DRYFT_TOKEN")
    request = urllib.request.Request(
        f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def num(value):
    return value if isinstance(value, (int, float)) else None


def find_workloads(blob):
    """The report shape is not documented; locate the per-workload list."""
    if isinstance(blob, list) and blob and isinstance(blob[0], dict):
        keys = set(blob[0])
        if keys & {"workload", "name", "batch", "tokensPerSecond", "tokens_per_second"}:
            return blob
    if isinstance(blob, dict):
        for value in blob.values():
            found = find_workloads(value)
            if found:
                return found
    return None


def get(row, *names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not run_id:
        runs = api("/api/v1/runs").get("items") or []
        if not runs:
            raise SystemExit("no runs yet")
        run_id = runs[0]["id"]

    run = api(f"/api/v1/runs/{run_id}")["run"]
    print(f"run {run_id}")
    print(f"  state   {run.get('state')}  mode {run.get('mode')}  attempt {run.get('attempt')}")
    print(f"  commit  {(run.get('commitSha') or '')[:12]}")
    if run.get("errorCode"):
        print(f"  ERROR   {run['errorCode']}: {run.get('errorMessage')}")
    for key in ("score", "scoreValue", "metricValue"):
        if run.get(key) is not None:
            print(f"  score   {run[key]}")

    rows = find_workloads(run)
    if not rows:
        print("\n-- raw report --")
        blob = {k: v for k, v in run.items() if k not in ("logs",)}
        print(json.dumps(blob, indent=1)[:4000])
        return

    header = f"{'workload':<12}{'b':>4}{'prompt':>8}{'out':>6}{'tok/s':>10}{'TTFT ms':>10}{'TPOT ms':>10}{'prefill%':>10}{'ratios':>18}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        name = get(row, "workload", "name", "id") or "?"
        batch = get(row, "batch", "batchSize")
        prompt = get(row, "prompt", "promptTokens", "prompt_tokens")
        out = get(row, "output", "outputTokens", "output_tokens")
        tps = num(get(row, "tokensPerSecond", "tokens_per_second", "throughput"))
        ttft = num(get(row, "ttftMs", "timeToFirstTokenMs", "ttft_ms"))
        tpot = num(get(row, "tpotMs", "timePerOutputTokenMs", "tpot_ms"))
        ttft_r = num(get(row, "ttftRatio", "ttft_ratio", "nativeTtftRatio"))
        tpot_r = num(get(row, "tpotRatio", "tpot_ratio", "nativeTpotRatio"))

        share = ""
        if ttft and tpot and isinstance(out, int) and out > 1:
            total = ttft + (out - 1) * tpot
            share = f"{100 * ttft / total:8.1f}%"
        ratios = ""
        if ttft_r or tpot_r:
            ratios = f"ttft {ttft_r or 0:.2f} tpot {tpot_r or 0:.2f}"

        print(
            f"{str(name):<12}{str(batch or '?'):>4}{str(prompt or '?'):>8}{str(out or '?'):>6}"
            f"{tps if tps is None else round(tps, 1)!s:>10}"
            f"{ttft if ttft is None else round(ttft, 1)!s:>10}"
            f"{tpot if tpot is None else round(tpot, 2)!s:>10}"
            f"{share:>10}{ratios:>18}"
        )

    print("\nGates: ttft and tpot ratios must stay under 1.10; spread under 25%.")


if __name__ == "__main__":
    main()
