# Network Graph Analysis - Ascendancy Take-Home

> **The core problem:** the data arrives as 248 enriched profiles - *not* a
> graph. Building useful edges from career histories is the analytical
> challenge, and the quality of that construction determines everything that
> follows.

Relationship-intelligence products like Ascendancy derive their value from
knowing *how* people are connected, not just *that* they are. This analysis
turns a raw enrichment export into a weighted graph, then applies community
detection and centrality metrics to answer two questions the product must
answer at scale:

1. **How well-connected is this network?**
2. **What groups or clusters does it consist of?**

The findings inform product questions like: *Which contacts are bridges between
otherwise-disconnected communities? Which are likely to go dormant because they
have no shared context with anyone else in the network? Who is the right
introduction path between any two people?*

## Live demo

**→ https://gaspigz.github.io/th-gaspar/**

An interactive report with four tabs - Overview (metrics + top hubs/brokers),
Network (the full 248-node graph, click any node), Communities, and the full
written Conclusions. Deployed automatically from `main` via GitHub Actions
(`.github/workflows/deploy.yml`).

> First-time setup: in the repo's **Settings → Pages**, set **Source** to
> **GitHub Actions**. The workflow then builds and publishes on every push to
> `main` that touches `site/`.

---

## Quick start

```bash
# Run the analysis  (requires Python ≥ 3.10)
python3 -m venv .venv && source .venv/bin/activate
pip install networkx python-louvain matplotlib pandas pyvis
python analyze.py
# Writes: metrics.json · communities.csv · community_summary.json
#         graph.graphml · network.png · network.html

# Serve the interactive report site
cd site
npm install
npm run dev        # http://localhost:4321  (hot-reload)
npm run build      # static output → site/dist/
npm run preview    # preview the production build locally
```

---

## Repository layout

```
Connections.json          raw input - 248 enriched profiles (4.6 MB)
analyze.py                graph construction + metrics + community detection
metrics.json              headline graph metrics (output)
communities.csv           per-person community label + centrality scores (output)
community_summary.json    per-community top people / companies / schools (output)
graph.graphml             portable graph file (Gephi, Cytoscape, Neo4j-ready)
network.png               static spring-layout visualisation
network.html              interactive pyvis visualisation (standalone)
REPORT.md                 full write-up: methodology, findings, limitations, next steps
site/                     Astro static site - four tabs:
                            Overview · Network (interactive) · Communities · Conclusions
```

---

## Key findings

| Metric | Value | Method | Interpretation |
|---|---|---|---|
| Nodes | 248 | - | people in the export |
| Edges (inferred) | 532 | bipartite affiliation projection | weighted shared-employer/school ties |
| Density | 1.74% | `nx.density` | sparse - typical for a personal network |
| Giant component | 139 (56%) | `nx.connected_components` | largest connected subgraph |
| Avg shortest path | 3.97 hops | `nx.average_shortest_path_length` (giant) | small-world structure |
| Louvain communities | 9 | `python-louvain best_partition` | distinct social clusters |
| Modularity | **0.72** | `community.modularity` | very strong clustering; communities are real, not statistical noise |
| Isolated nodes | **80 (32%)** | degree = 0 after projection | highest-risk contacts for relationship decay |

### What the numbers mean for a relationship-intelligence product

- **Modularity 0.72** is well above the 0.4–0.6 range typical for personal
  networks. The ego's connections live in tightly-separated silos - information
  and introductions do *not* flow freely across the network without deliberate
  brokerage. This is exactly the gap a product like Ascendancy fills.

- **The 80 isolates** are direct connections with no inferable peer inside the
  export. They are the contacts most likely to go cold: no mutual context, no
  ambient social reinforcement. Surfacing these proactively ("You haven't
  connected with X in 6 months, and they know no one else you know") is a
  high-signal feature.

- **Hubs ≠ brokers.** The top eigenvector-centrality node (Francis Pedraza,
  eig = 0.58) and the top betweenness node (Matt Franchi, btw = 0.29) are
  different people. A CRM that ranks contacts by degree alone misses the
  brokers - the people whose removal would fragment the network. Both lists
  matter for different use cases (influence vs. bridge-building).

- **The two-stratum structure** (Clemson alumni network <-> startup-operator
  core) is joined by only ~5 people. Strengthening those bridges - or
  identifying new ones - is the highest-leverage action for network growth.

---

## How the graph is built

Because the source data contains no explicit edges, connections are inferred
via a **bipartite affiliation projection**: two people share an edge when they
overlap at the same employer or school. Edge weight reflects the strength of
that signal - tenure overlap in months, damped by log(affiliation size) to
prevent large employers from artificially linking the whole graph. Full
methodology in [`REPORT.md`](REPORT.md).

---

See [`REPORT.md`](REPORT.md) for the complete write-up including assumptions,
limitations, and suggested next steps.
