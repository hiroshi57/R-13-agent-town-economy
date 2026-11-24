"""Blind wage raises vs. raising "only at the right time" (option C).

The earlier sweep (RESULTS: wage-raise-prob) showed that simply raising wages
more often does NOT restore pass-through -- it slightly worsens it, because
over-raising drains business cash and the wage payments start to FAIL.

Here we compare the same connected-memory policy (WikiPolicy) with the wage
lever ungated vs. gated by affordability: raise only when the business can
cover it and the wage is not already above what each sale brings in. If the
gated arm pays more wages successfully (fewer failures) and holds a healthier
wage share, "when you raise" matters more than "how often".

    python experiments/wage_timing.py
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.metrics import compute_metrics, marginal_propensity_to_consume  # noqa: E402
from analysis.validate import validate  # noqa: E402
from sim.conditions import get_condition  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from sim.policy import WikiPolicy  # noqa: E402
from sim.wiki_memory import WikiMemory  # noqa: E402


def run(gate: bool, condition: str, seed: int, pulses: int, n: int) -> dict:
    policy = WikiPolicy(random.Random(seed + 7), afford_gate=gate)
    sim = Simulation(
        get_condition(condition),
        policy,
        seed=seed,
        n_agents=n,
        memory_factory=lambda: WikiMemory(mode="wiki"),
    )
    res = sim.run(pulses)
    m = compute_metrics(res)
    set_wage = res.tool_stats.get("set_wage", {"calls": 0})["calls"]
    row = {
        "mode": "適切な時だけ(gate)" if gate else "闇雲(no-gate)",
        "condition": condition,
        "validated": validate(res)["all_passed"],
        "set_wage_calls": set_wage,
        "wages_paid_ok": res.wages_paid_ok,
        "wages_failed": res.wages_failed,
        "wage_fail_rate": round(
            res.wages_failed / (res.wages_paid_ok + res.wages_failed), 3
        )
        if (res.wages_paid_ok + res.wages_failed)
        else 0.0,
        "wage_share": m["wage_share_of_revenue"],
        "gini": m["gini_terminal"],
    }
    mpc = marginal_propensity_to_consume(res)
    if mpc is not None:
        row["mpc"] = mpc["mpc"]
    return row


def main() -> int:
    rows = []
    for cond in ["tourism-high", "wealth-grant"]:
        for gate in (False, True):
            row = run(gate, cond, seed=1, pulses=200, n=60)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "results_wage_timing.json",
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
