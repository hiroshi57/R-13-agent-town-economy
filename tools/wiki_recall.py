"""Multi-hop memory recall over a folder of markdown notes linked with [[name]].

Ports the BFS idea of sim/wiki_memory.py to an Obsidian / .claude/agent-memory
style store: given a seed note (or keyword), return every note reachable within
K hops along [[wiki-links]] (and Markdown [text](file.md) links), so an agent
retrieves *connected* context instead of keyword-matched fragments.

This is the harness-migration prototype (option B): point --dir at your real
`.claude/agent-memory` to pull the memories connected to a task, not just the
ones that share a word with it.

Usage (PowerShell):
  python tools/wiki_recall.py --dir tools/demo_memory --start preservation-rule
  python tools/wiki_recall.py --dir tools/demo_memory --query "curation" --hops 2
  python tools/wiki_recall.py --dir .claude/agent-memory --query "memory" --hops 2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

try:  # keep Japanese + symbols printable on Windows (cp932) consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WIKILINK = re.compile(r"\[\[([^\]|#]+)")  # [[name]] / [[name|alias]] / [[name#h]]
MDLINK = re.compile(r"\]\(([^)]+?\.md)\)")  # [text](file.md)
NAME_FM = re.compile(r"^name:\s*(.+?)\s*$", re.M)  # frontmatter: name: slug


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_graph(root: str):
    """Return (nodes, edges, text). nodes: id->path, edges: id->[neighbor ids]."""
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".md"):
                files.append(os.path.join(dp, fn))

    id_of: Dict[str, str] = {}  # stem (lower) -> canonical id
    path_of: Dict[str, str] = {}
    text_of: Dict[str, str] = {}
    raw: Dict[str, str] = {}
    for p in files:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            body = fh.read()
        fm = NAME_FM.search(body)
        nid = fm.group(1).strip() if fm else _stem(p)
        id_of[_stem(p).lower()] = nid
        id_of[nid.lower()] = nid
        path_of[nid] = os.path.relpath(p, root)
        raw[nid] = body
        text_of[nid] = body

    edges: Dict[str, List[str]] = defaultdict(list)
    for nid, body in raw.items():
        targets = set(WIKILINK.findall(body))
        targets |= {_stem(m) for m in MDLINK.findall(body)}
        for t in targets:
            canon = id_of.get(t.strip().lower())
            if canon and canon != nid and canon not in edges[nid]:
                edges[nid].append(canon)
    # make links traversable both ways (a mention connects both notes)
    undirected: Dict[str, List[str]] = defaultdict(list)
    for a, bs in edges.items():
        for b in bs:
            if b not in undirected[a]:
                undirected[a].append(b)
            if a not in undirected[b]:
                undirected[b].append(a)
    return path_of, undirected, text_of


def seeds_from_query(text_of: Dict[str, str], query: str) -> List[str]:
    q = query.lower()
    return [
        nid for nid, body in text_of.items() if q in nid.lower() or q in body.lower()
    ]


def recall(
    edges, start_ids: List[str], hops: int
) -> List[Tuple[str, int, Optional[str]]]:
    """BFS: return [(id, hop_distance, via_parent)] reachable within `hops`."""
    seen = {s: (0, None) for s in start_ids}
    q = deque((s, 0) for s in start_ids)
    while q:
        cur, d = q.popleft()
        if d >= hops:
            continue
        for nb in edges.get(cur, []):
            if nb not in seen:
                seen[nb] = (d + 1, cur)
                q.append((nb, d + 1))
    out = [(nid, dist, via) for nid, (dist, via) in seen.items()]
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def _excerpt(body: str, n: int = 70) -> str:
    for line in body.splitlines():
        s = line.strip().lstrip("#").strip()
        if (
            s
            and not s.startswith("---")
            and not s.startswith("name:")
            and not s.startswith("description:")
        ):
            return (s[:n] + "…") if len(s) > n else s
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-hop markdown memory recall")
    ap.add_argument("--dir", required=True)
    ap.add_argument(
        "--start", default=None, help="seed note id (filename stem or frontmatter name)"
    )
    ap.add_argument("--query", default=None, help="seed by keyword match instead")
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()

    path_of, edges, text_of = load_graph(args.dir)
    if not path_of:
        print(f"no .md files under {args.dir!r}")
        return 1

    if args.start:
        lower = {k.lower(): k for k in path_of}
        start = lower.get(args.start.lower())
        starts = [start] if start else []
    elif args.query:
        starts = seeds_from_query(text_of, args.query)
    else:
        print("give --start or --query")
        return 1
    if not starts:
        print("no seed note matched")
        return 1

    print(f"seeds ({len(starts)}): {', '.join(starts)}")
    print(f"--- connected within {args.hops} hop(s) ---")
    for nid, dist, via in recall(edges, starts, args.hops):
        tag = "seed" if dist == 0 else f"{dist}-hop via {via}"
        print(f"[{tag:>16}] {nid}  —  {_excerpt(text_of[nid])}  ({path_of[nid]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
