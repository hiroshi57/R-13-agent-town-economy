"""Tests for the harness-bridge features: wiki recall (B) + wage timing (C)."""

import os
import random

from tools.wiki_recall import load_graph, recall, seeds_from_query
from sim.conditions import tourism_high
from sim.engine import Simulation
from sim.policy import WikiPolicy
from sim.wiki_memory import WikiMemory

DEMO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "demo_memory"
)


# --- B: multi-hop markdown recall -------------------------------------
def test_recall_reaches_connected_excludes_unrelated():
    path_of, edges, text_of = load_graph(DEMO)
    reached = {nid for nid, _, _ in recall(edges, ["preservation-rule"], hops=2)}
    # connected notes are pulled in, even without sharing a keyword
    assert {
        "memory-curation",
        "harness-retro",
        "store-health",
        "toolset-lever",
    } <= reached
    # an unlinked note is NOT pulled in
    assert "unrelated-ccmux" not in reached


def test_recall_hop_distance_monotonic():
    path_of, edges, text_of = load_graph(DEMO)
    dist = {nid: d for nid, d, _ in recall(edges, ["preservation-rule"], hops=2)}
    assert dist["preservation-rule"] == 0
    assert dist["harness-retro"] == 1
    assert dist["store-health"] == 2  # only reachable via harness-retro


def test_query_seeds_match():
    _, _, text_of = load_graph(DEMO)
    assert "toolset-lever" in seeds_from_query(text_of, "ツールセット")


# --- C: raising wages "at the right time" -----------------------------
def _run(gate):
    policy = WikiPolicy(random.Random(8), afford_gate=gate)
    sim = Simulation(
        tourism_high(),
        policy,
        seed=1,
        n_agents=40,
        memory_factory=lambda: WikiMemory(mode="wiki"),
    )
    return sim.run(120)


def test_afford_gate_raises_less_but_stays_conserved():
    ungated = _run(False)
    gated = _run(True)

    def calls(res):
        return res.tool_stats.get("set_wage", {"calls": 0})["calls"]

    # the gate fires the lever far less often (only when affordable)
    assert calls(gated) < calls(ungated)
    # and money is still conserved in both
    from analysis.validate import validate

    assert validate(ungated)["all_passed"] and validate(gated)["all_passed"]
    # the engine now tracks wage delivery
    assert gated.wages_paid_ok > 0
