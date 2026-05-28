# Network Graph Analysis — Ascendancy Take-Home

Analysis of a personal LinkedIn connections export (248 enriched profiles) to answer two questions:
**How well-connected is this network?** and **What communities does it consist of?**

## Quick start

```bash
# Run the analysis (needs Python ≥ 3.10)
python3 -m venv .venv && source .venv/bin/activate
pip install networkx python-louvain matplotlib pandas pyvis
python analyze.py
# → writes metrics.json, communities.csv, community_summary.json,
#   graph.graphml, network.png, network.html

# Serve the Astro report site
cd site
npm install
npm run dev        # localhost:4321
npm run build      # static output in site/dist/
```

## Repository layout

```
Connections.json          raw input (248 enriched profiles)
analyze.py                full analysis pipeline
metrics.json              headline graph metrics
communities.csv           per-person community + centrality
community_summary.json    per-community top people / companies / schools
graph.graphml             exportable graph (Gephi, Cytoscape, Neo4j)
network.png               static visualisation
network.html              interactive pyvis visualisation
REPORT.md                 full write-up
site/                     Astro report site (tabs: Overview · Network · Communities · Conclusions)
```

## Key findings

| Metric | Value |
|---|---|
| Nodes | 248 |
| Edges (inferred) | 532 |
| Density | 1.74% |
| Giant component | 139 nodes (56%) |
| Avg shortest path | 3.97 hops |
| Louvain communities | 9 |
| Modularity | 0.72 |

The network splits into a **Clemson alumni stratum** and a **startup-operator stratum** (Ascendancy / Invisible Technologies / Social Slooth), joined by ~5 broker nodes. 80 nodes are isolated — direct contacts with no inferable peer, the highest-value targets for a relationship-intelligence product.

See [`REPORT.md`](REPORT.md) for the full analysis including methodology, assumptions, limitations, and next steps.
