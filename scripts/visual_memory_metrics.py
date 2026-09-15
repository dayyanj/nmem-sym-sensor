#!/usr/bin/env python3
"""Phase-3 behavioral-advantage metric for visual memory.

Reads ``screen_pursuit_links`` (dhash-keyed screen ↔ pursuit-outcome links) and reports whether the
SEE→REMEMBER→ACT loop is actually helping — i.e. whether the agent stops re-failing on a screen it
has already failed on (the read-back should surface "you failed here before; try differently").

Distinct SCREENS are formed by clustering the perceptual dhashes with Hamming ≤ --radius (default 8;
different UI screens measure ≥18 apart, same-screen ≤4). For each screen we look at its pursuit
verdicts in time order and ask: after the FIRST failure on a screen, does the NEXT visit still fail
(bad — warning ignored / unheeded) or succeed / not-recur (good)?

Key metric — REPEATED-FAILURE RATE on revisited screens = (screens whose next visit after a failure
also failed) / (screens that failed at least once and were revisited). Lower is better; it should
fall once the read-back is doing its job. Also reports steps-to-completion if a goals table is
available (best-effort). No writes — read-only.

Usage:
    NMEM_SENSOR_DB_DSN=postgresql://…/agent_db python scripts/visual_memory_metrics.py [--agent michelle] [--radius 8]
"""
import argparse
import asyncio
import os
import sys


def _ham(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:  # noqa: BLE001
        return 64


def _cluster(phashes, radius):
    """Greedy single-link clustering of dhashes by Hamming radius → list of representative indices
    per row. Returns a list assigning each row a screen-cluster id."""
    reps: list[str] = []
    assign = []
    for ph in phashes:
        hit = None
        for i, rep in enumerate(reps):
            if _ham(ph, rep) <= radius:
                hit = i
                break
        if hit is None:
            reps.append(ph)
            hit = len(reps) - 1
        assign.append(hit)
    return assign, len(reps)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=None, help="restrict to one agent_id")
    ap.add_argument("--radius", type=int, default=8, help="same-screen dhash Hamming radius")
    ap.add_argument("--dsn", default=os.environ.get("NMEM_SENSOR_DB_DSN"))
    args = ap.parse_args()
    if not args.dsn:
        print("set NMEM_SENSOR_DB_DSN or pass --dsn", file=sys.stderr)
        return 2

    import asyncpg
    conn = await asyncpg.connect(args.dsn)
    try:
        where = "WHERE agent_id = $1" if args.agent else ""
        rows = await conn.fetch(
            f"""SELECT phash, agent_id, goal_id, verdict, readback, created_at
                FROM screen_pursuit_links {where} ORDER BY created_at ASC""",
            *([args.agent] if args.agent else []))
    finally:
        await conn.close()

    if not rows:
        print("no screen_pursuit_links yet — the loop has recorded nothing to measure.")
        return 0

    phashes = [r["phash"] for r in rows]
    assign, n_screens = _cluster(phashes, args.radius)

    # Per-screen timeline (chronological) of (verdict, readback-arm).
    timeline: dict[int, list[tuple[str, bool]]] = {}
    for idx, r in zip(assign, rows):
        timeline.setdefault(idx, []).append((r["verdict"], bool(r["readback"])))

    total_links = len(rows)
    failed_screens = [t for t in timeline.values() if any(v == "f" for v, _ in t)]
    revisited = [t for t in timeline.values() if len(t) >= 2]

    # The causal question: after a screen's FIRST failure, does the NEXT visit re-fail? Attribute
    # that next-visit to ITS OWN read-back arm (was the agent actually warned at that moment?).
    # arm → [n_next_visits, n_re_failed].
    arms = {True: [0, 0], False: [0, 0]}
    overall = [0, 0]
    for t in timeline.values():
        verdicts = [v for v, _ in t]
        if "f" not in verdicts:
            continue
        first_f = verdicts.index("f")
        after = t[first_f + 1:]
        if not after:
            continue
        nv_verdict, nv_arm = after[0]           # the immediate next visit after the first failure
        arms[nv_arm][0] += 1
        overall[0] += 1
        if nv_verdict == "f":
            arms[nv_arm][1] += 1
            overall[1] += 1

    def _rate(pair):
        n, f = pair
        return f"{f}/{n} = {f/n:.0%}" if n else "n/a (0)"

    print("── Visual-memory behavioral-advantage metric ──")
    print(f"agent               : {args.agent or 'ALL'}")
    print(f"screen links (rows) : {total_links}")
    print(f"distinct screens    : {n_screens}  (dhash Hamming ≤ {args.radius})")
    print(f"screens ever failed : {len(failed_screens)}")
    print(f"screens revisited   : {len(revisited)}")
    print(f"failed & revisited  : {overall[0]}")
    if overall[0]:
        print("REPEATED-FAILURE RATE (next visit after a first failure also failed) ← lower is better:")
        print(f"  overall          : {_rate(overall)}")
        print(f"  read-back ON  arm: {_rate(arms[True])}   (agent was warned)")
        print(f"  read-back OFF arm: {_rate(arms[False])}   (A/B baseline, no warning)")
        print("A genuine advantage = ON-arm rate materially BELOW the OFF-arm rate on comparable n.")
    else:
        print("REPEATED-FAILURE RATE: n/a — no failed screen has been revisited yet.")
        print("(Need more pursuits over the same screens; the metric fills in as michelle runs.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
