"""
Network analysis of a personal LinkedIn-style connections export.

Graph model
-----------
Nodes  : 248 people (one per record).
Edges  : Undirected, weighted. Two people get an edge when they share a
         real-world affiliation. Three edge "channels" are merged into a
         single weighted graph:

           1. coworker_overlap  - same company_id, tenure windows overlap
                                  (weight = months of overlap, capped at 60)
           2. coworker_shared   - same company_id, no/unknown overlap
                                  (weight = 2; weaker signal of "worked together")
           3. classmate_overlap - same school, study windows overlap
                                  (weight = months of overlap / 6, capped at 12)
           4. classmate_shared  - same school, no/unknown overlap (weight = 1)

         Edges from very large employers (>1000 people in this dataset's
         affiliations, or famous mega-employers we know are noisy) are
         downweighted: alumni of "Amazon" or "Deloitte" shouldn't link the
         whole graph. We compute a per-company size and divide each shared-
         company edge by log(1 + size).

         This is a *projection* of an underlying bipartite graph
         (People <-> Affiliations). The projection is what we analyse.

Output
------
  - metrics.json   : numeric metrics
  - communities.csv: per-person community + centrality
  - graph.graphml  : the graph (for Gephi/Neo4j/Cytoscape)
  - network.html   : interactive pyvis visualization
  - network.png    : static layout image
"""

from __future__ import annotations
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import networkx as nx
import community as community_louvain  # python-louvain
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = json.loads((HERE / "Connections.json").read_text())["data"]


# ---------- helpers ----------------------------------------------------------

def parse_date(s):
    """Parse 'YYYY-MM-DD' or 'YYYY-MM' or 'YYYY'. Return date or None."""
    if not s:
        return None
    if isinstance(s, date):
        return s
    s = str(s)[:10]
    parts = s.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, d)
    except (ValueError, IndexError):
        return None


def overlap_months(a_start, a_end, b_start, b_end):
    """Months of overlap of two date intervals. Open-ended end -> today.
    Returns 0 if either start is unknown."""
    if not a_start or not b_start:
        return 0
    today = date.today()
    a_end = a_end or today
    b_end = b_end or today
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    if hi < lo:
        return 0
    return (hi.year - lo.year) * 12 + (hi.month - lo.month) + 1


def norm_school(name: str) -> str:
    """Normalise a school name for matching."""
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # collapse common variants
    s = s.replace("the university of ", "university of ")
    return s


# ---------- extract per-person affiliations ----------------------------------

people = []          # list of dicts (id, name, headline, current_company, ...)
person_companies = []  # list per person of [(company_id, start, end, title)]
person_schools = []    # list per person of [(school_key, start, end)]

for rec in DATA:
    pid = rec.get("id") or rec.get("linkedin_url")
    people.append({
        "id": pid,
        "name": rec.get("full_name") or "",
        "headline": rec.get("headline") or "",
        "current_title": rec.get("current_title") or "",
        "current_company": rec.get("current_company_name") or "",
        "location": (rec.get("current_location") or {}).get("name") if isinstance(rec.get("current_location"), dict) else (rec.get("current_location") or ""),
        "linkedin": rec.get("linkedin_url") or "",
    })
    cos = []
    for e in (rec.get("experience") or []):
        comp = e.get("company") or {}
        cid = e.get("company_id") or comp.get("id")
        cname = comp.get("name")
        if not cid and not cname:
            continue
        key = cid or ("name::" + cname.strip().lower())
        cos.append((key, cname, parse_date(e.get("start_date")), parse_date(e.get("end_date"))))
    person_companies.append(cos)

    schs = []
    for e in (rec.get("education") or []):
        sch = e.get("school") or {}
        sname = sch.get("name") or ""
        if not sname:
            continue
        key = norm_school(sname)
        if not key:
            continue
        schs.append((key, sname, parse_date(e.get("start_date")), parse_date(e.get("end_date"))))
    person_schools.append(schs)


# ---------- count affiliation sizes -----------------------------------------

company_size = Counter()
company_name = {}
for cos in person_companies:
    seen = set()
    for cid, cname, _s, _e in cos:
        if cid in seen:
            continue
        seen.add(cid)
        company_size[cid] += 1
        if cname:
            company_name[cid] = cname

school_size = Counter()
school_name = {}
for schs in person_schools:
    seen = set()
    for skey, sname, _s, _e in schs:
        if skey in seen:
            continue
        seen.add(skey)
        school_size[skey] += 1
        school_name[skey] = sname


# ---------- build graph ------------------------------------------------------

G = nx.Graph()
for p in people:
    G.add_node(p["id"], **{k: v for k, v in p.items() if k != "id"})

# A helper to add / accumulate weight with provenance
def add_edge(u, v, w, channel, label):
    if u == v:
        return
    if G.has_edge(u, v):
        d = G[u][v]
        d["weight"] += w
        d["channels"].add(channel)
        d["evidence"].append(label)
    else:
        G.add_edge(u, v, weight=w, channels={channel}, evidence=[label])


N = len(people)
COMPANY_NOISE_CAP = 25   # if more than this many people share an employer, treat as background
SCHOOL_NOISE_CAP = 30

# Group people by company key
co_to_people = defaultdict(list)
for i, cos in enumerate(person_companies):
    seen_keys = set()
    for cid, cname, s, e in cos:
        seen_keys.add(cid)
        co_to_people[cid].append((i, s, e, cname))
    # de-dup person within company (keep all stints — overlap math handles them)

for cid, members in co_to_people.items():
    size = company_size[cid]
    cname = company_name.get(cid, "?")
    if size < 2:
        continue
    # Background damping: log scale, and skip very-noisy companies entirely
    if size > COMPANY_NOISE_CAP:
        continue
    damping = 1.0 / math.log(1 + size + 1)  # ~0.91 for size 2, 0.30 for size 25
    # All pairs
    M = len(members)
    for a in range(M):
        i, sa, ea, _ = members[a]
        for b in range(a + 1, M):
            j, sb, eb, _ = members[b]
            if i == j:
                continue
            ov = overlap_months(sa, ea, sb, eb)
            if ov > 0:
                w = min(ov, 60) / 12.0 * damping  # years of overlap, capped
                add_edge(people[i]["id"], people[j]["id"], w, "coworker_overlap",
                         f"coworker@{cname} ~{ov}mo")
            else:
                # shared employer but no overlap (or dates missing) - weaker
                add_edge(people[i]["id"], people[j]["id"], 0.5 * damping,
                         "coworker_shared", f"shared@{cname}")

sch_to_people = defaultdict(list)
for i, schs in enumerate(person_schools):
    for skey, sname, s, e in schs:
        sch_to_people[skey].append((i, s, e, sname))

for skey, members in sch_to_people.items():
    size = school_size[skey]
    sname = school_name.get(skey, "?")
    if size < 2:
        continue
    if size > SCHOOL_NOISE_CAP:
        continue
    damping = 1.0 / math.log(1 + size + 1)
    M = len(members)
    for a in range(M):
        i, sa, ea, _ = members[a]
        for b in range(a + 1, M):
            j, sb, eb, _ = members[b]
            if i == j:
                continue
            ov = overlap_months(sa, ea, sb, eb)
            if ov > 0:
                w = min(ov, 48) / 12.0 * damping
                add_edge(people[i]["id"], people[j]["id"], w, "classmate_overlap",
                         f"classmate@{sname} ~{ov}mo")
            else:
                add_edge(people[i]["id"], people[j]["id"], 0.25 * damping,
                         "classmate_shared", f"alum@{sname}")


# convert channels (set) to comma string for graphml export
for u, v, d in G.edges(data=True):
    d["channels"] = ",".join(sorted(d["channels"]))
    d["evidence"] = " | ".join(d["evidence"][:5])


# ---------- metrics ----------------------------------------------------------

n = G.number_of_nodes()
m = G.number_of_edges()
density = nx.density(G)
components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)
giant = G.subgraph(components[0]).copy() if components else G

avg_deg = (2 * m) / n if n else 0
deg = dict(G.degree())
deg_w = dict(G.degree(weight="weight"))
isolates = [nid for nid, d in deg.items() if d == 0]

clustering = nx.average_clustering(G, weight=None) if m else 0
transitivity = nx.transitivity(G) if m else 0
giant_diameter = nx.diameter(giant) if giant.number_of_nodes() > 1 else 0
giant_avg_path = nx.average_shortest_path_length(giant) if giant.number_of_nodes() > 1 else 0

# Centralities on the giant component (more meaningful)
deg_cent = nx.degree_centrality(giant)
btw_cent = nx.betweenness_centrality(giant, weight=None, normalized=True)
try:
    eig_cent = nx.eigenvector_centrality(giant, max_iter=1000, weight="weight")
except nx.PowerIterationFailedConvergence:
    eig_cent = {n: 0.0 for n in giant.nodes()}

# Louvain community detection on the giant
partition = community_louvain.best_partition(giant, weight="weight", random_state=42)
n_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, giant, weight="weight")

# ---------- write outputs ----------------------------------------------------

metrics = {
    "nodes": n,
    "edges": m,
    "density": round(density, 5),
    "avg_degree": round(avg_deg, 2),
    "isolated_nodes": len(isolates),
    "connected_components": len(components),
    "component_sizes_top5": [len(c) for c in components[:5]],
    "giant_component_size": giant.number_of_nodes(),
    "giant_component_edges": giant.number_of_edges(),
    "giant_diameter": giant_diameter,
    "giant_avg_shortest_path": round(giant_avg_path, 3),
    "avg_clustering_coef": round(clustering, 4),
    "global_transitivity": round(transitivity, 4),
    "louvain_communities": n_communities,
    "louvain_modularity": round(modularity, 4),
    "top_companies_by_size_in_network": [
        {"company": company_name.get(k, str(k)), "people": v}
        for k, v in company_size.most_common(15)
    ],
    "top_schools_by_size_in_network": [
        {"school": school_name.get(k, k), "people": v}
        for k, v in school_size.most_common(10)
    ],
    "excluded_noisy_companies": [
        {"company": company_name.get(k, str(k)), "people": v}
        for k, v in company_size.most_common() if v > COMPANY_NOISE_CAP
    ],
    "excluded_noisy_schools": [
        {"school": school_name.get(k, k), "people": v}
        for k, v in school_size.most_common() if v > SCHOOL_NOISE_CAP
    ],
}

(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))

# Per-person table
rows = []
for p in people:
    pid = p["id"]
    rows.append({
        "id": pid,
        "name": p["name"],
        "current_company": p["current_company"],
        "current_title": p["current_title"],
        "location": p["location"],
        "degree": deg.get(pid, 0),
        "weighted_degree": round(deg_w.get(pid, 0), 3),
        "degree_centrality": round(deg_cent.get(pid, 0), 4),
        "betweenness": round(btw_cent.get(pid, 0), 4),
        "eigenvector": round(eig_cent.get(pid, 0), 4),
        "community": partition.get(pid, -1),
        "in_giant_component": pid in giant.nodes(),
    })
df = pd.DataFrame(rows).sort_values(["community", "weighted_degree"], ascending=[True, False])
df.to_csv(HERE / "communities.csv", index=False)

# Community summary: top employers/schools per community
community_members = defaultdict(list)
for nid, c in partition.items():
    community_members[c].append(nid)

community_summary = []
for c, members in sorted(community_members.items(), key=lambda kv: -len(kv[1])):
    # gather companies & schools for members
    co_counter = Counter()
    sch_counter = Counter()
    member_set = set(members)
    for i, p in enumerate(people):
        if p["id"] not in member_set:
            continue
        for cid, cname, _s, _e in person_companies[i]:
            if cname:
                co_counter[cname] += 1
        for skey, sname, _s, _e in person_schools[i]:
            sch_counter[sname] += 1
    top_names = [people[i]["name"] for i in range(len(people))
                 if people[i]["id"] in member_set]
    # rank within-community by weighted degree
    top_names_sorted = sorted(
        top_names,
        key=lambda nm: -max((r["weighted_degree"] for r in rows if r["name"] == nm), default=0)
    )[:5]
    community_summary.append({
        "community": c,
        "size": len(members),
        "top_people": top_names_sorted,
        "top_companies": co_counter.most_common(5),
        "top_schools": sch_counter.most_common(3),
    })

(HERE / "community_summary.json").write_text(json.dumps(community_summary, indent=2))

# graphml export - attach community + centrality so Gephi/Cytoscape can colour
for nid in G.nodes():
    G.nodes[nid]["community"] = partition.get(nid, -1)
    G.nodes[nid]["degree"] = deg.get(nid, 0)
    G.nodes[nid]["weighted_degree"] = deg_w.get(nid, 0)
    G.nodes[nid]["betweenness"] = btw_cent.get(nid, 0)
    G.nodes[nid]["eigenvector"] = eig_cent.get(nid, 0)

nx.write_graphml(G, HERE / "graph.graphml")

# ---------- visualisations ---------------------------------------------------

# static png — spring layout, colour by community, size by weighted degree
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(giant, seed=42, k=0.45, iterations=120, weight="weight")
node_colors = [partition[n] for n in giant.nodes()]
node_sizes = [40 + 25 * deg_w[n] for n in giant.nodes()]
nx.draw_networkx_edges(giant, pos, alpha=0.18, width=0.4)
nx.draw_networkx_nodes(giant, pos, node_color=node_colors, cmap="tab20",
                       node_size=node_sizes, linewidths=0.4, edgecolors="#222")
# label only the top hubs
top_by_eig = sorted(giant.nodes(), key=lambda n: -eig_cent.get(n, 0))[:15]
labels = {n: G.nodes[n].get("name", "") for n in top_by_eig}
nx.draw_networkx_labels(giant, pos, labels=labels, font_size=8)
plt.title(f"Personal network ({n} nodes, {m} edges, {n_communities} Louvain communities)")
plt.axis("off")
plt.tight_layout()
plt.savefig(HERE / "network.png", dpi=180)
plt.close()

# interactive html
from pyvis.network import Network
net = Network(height="800px", width="100%", bgcolor="#0e0e0e", font_color="#eee",
              notebook=False, cdn_resources="in_line")
net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=120,
                      spring_strength=0.04, damping=0.6)
palette = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4",
           "#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#fffac8",
           "#800000","#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9"]
for nid in G.nodes():
    c = partition.get(nid, 0)
    p = G.nodes[nid]
    title = f"<b>{p.get('name','')}</b><br>{p.get('current_title','')} @ {p.get('current_company','')}<br>{p.get('location','') or ''}<br>deg={deg[nid]}, w-deg={deg_w[nid]:.1f}<br>community={c}"
    net.add_node(nid, label=p.get("name", "")[:24],
                 title=title,
                 color=palette[c % len(palette)],
                 value=max(1, deg_w[nid]))
for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d["weight"], title=d.get("evidence", ""))
net.write_html(str(HERE / "network.html"), notebook=False, open_browser=False)

# ---------- console summary --------------------------------------------------

print("=" * 70)
print("NETWORK ANALYSIS")
print("=" * 70)
for k, v in metrics.items():
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print(f"{k}:")
        for row in v[:10]:
            print("   ", row)
    else:
        print(f"{k}: {v}")

print("\nTop 10 hubs by eigenvector centrality:")
top_eig = sorted(giant.nodes(), key=lambda n: -eig_cent.get(n, 0))[:10]
for n in top_eig:
    p = G.nodes[n]
    print(f"  eig={eig_cent[n]:.3f}  deg={deg[n]:3d}  {p.get('name','')[:35]:35s}  {p.get('current_title','')[:25]:25s} @ {p.get('current_company','')[:25]}")

print("\nTop 10 brokers by betweenness centrality:")
top_btw = sorted(giant.nodes(), key=lambda n: -btw_cent.get(n, 0))[:10]
for n in top_btw:
    p = G.nodes[n]
    print(f"  btw={btw_cent[n]:.3f}  deg={deg[n]:3d}  {p.get('name','')[:35]:35s}  {p.get('current_title','')[:25]:25s} @ {p.get('current_company','')[:25]}")

print("\nCommunities (Louvain):")
for cs in community_summary:
    cos = ", ".join(f"{c}({n})" for c, n in cs["top_companies"][:3])
    print(f"  C{cs['community']:2d}  size={cs['size']:3d}  top: {', '.join(cs['top_people'][:3])}  | cos: {cos}")
