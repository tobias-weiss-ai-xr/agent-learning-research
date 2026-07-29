#!/usr/bin/env python3
"""Generate enhanced graph_analysis.html with D3 force-directed graph + comprehensive analysis."""

import json
import re
import os
import yaml
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATEGORY_ORDER = [
    "supervised",
    "unsupervised",
    "reinforcement",
    "self-supervised",
    "meta-learning",
    "continual",
    "transfer",
    "multi-agent",
    "active",
    "online",
    "federated",
    "curriculum",
    "neurosymbolic",
    "causal",
]
SUBCATEGORY_ORDER = [
    "theory",
    "algorithm",
    "architecture",
    "optimization",
    "scaling",
    "efficient",
    "robust",
    "application",
]
CATEGORY_COLORS = {
    "supervised": "#58a6ff",
    "unsupervised": "#3fb950",
    "reinforcement": "#d29922",
    "self-supervised": "#f0883e",
    "meta-learning": "#db6d8a",
    "continual": "#7ee787",
    "transfer": "#a5d6ff",
    "multi-agent": "#79c0ff",
    "active": "#d2a8ff",
    "online": "#ff7b72",
    "federated": "#ffa657",
    "curriculum": "#ffd700",
    "neurosymbolic": "#56d4dd",
    "causal": "#b392f0",
}

with open(os.path.join(BASE, "papers.yaml"), encoding="utf-8") as f:
    _data = yaml.safe_load(f)
entries = _data.get("papers", [])
print(f"Parsed {len(entries)} papers")

cat_counter = Counter()
subcat_counter = Counter()
cat_subcat = defaultdict(lambda: defaultdict(int))
pub_dates = []
venue_counter = Counter()

for e in entries:
    cat = e.get("category", "unknown")
    sub = e.get("subcategory", "unknown")
    cat_counter[cat] += 1
    subcat_counter[sub] += 1
    cat_subcat[cat][sub] += 1
    d = e.get("date", "")
    if d and len(d) >= 7:
        pub_dates.append((d[:7], cat, sub))
    v = e.get("venue", "")
    if v:
        venue_counter[v] += 1
    else:
        venue_counter["Unknown/None"] += 1

total = len(entries)
cat_vals = [cat_counter.get(c, 0) for c in CATEGORY_ORDER]
subcat_vals = [subcat_counter.get(s, 0) for s in SUBCATEGORY_ORDER]
cat_color_vals = [CATEGORY_COLORS.get(c, "#8b949e") for c in CATEGORY_ORDER]

ym_set = sorted(set(ym for ym, _, _ in pub_dates))
if not ym_set:
    ym_set = ["2024-01"]
ym_total_counter = Counter()
ym_cat_counter = defaultdict(lambda: Counter())
for ym, cat, _ in pub_dates:
    ym_total_counter[ym] += 1
    ym_cat_counter[ym][cat] += 1

tl_total = [ym_total_counter[ym] for ym in ym_set]

cum_total = []
t = 0
for i in range(len(ym_set)):
    t += tl_total[i]
    cum_total.append(t)

year_total = defaultdict(int)
year_cat = defaultdict(lambda: defaultdict(int))
for ym, cat, _ in pub_dates:
    y = ym[:4]
    year_total[y] += 1
    year_cat[y][cat] += 1
years_sorted = sorted(year_total.keys())

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "to",
    "and",
    "on",
    "with",
    "via",
    "by",
    "from",
    "as",
    "at",
    "is",
    "that",
    "this",
    "its",
    "their",
    "our",
    "are",
    "based",
    "using",
    "toward",
    "towards",
    "across",
    "over",
    "through",
    "into",
    "between",
    "after",
    "under",
    "during",
    "without",
    "before",
    "all",
    "each",
    "both",
    "more",
    "than",
    "most",
    "some",
    "any",
    "new",
    "large",
    "long",
    "short",
    "high",
    "low",
    "multi",
    "self",
    "co",
}
word_counter = Counter()
for e in entries:
    title = e.get("title", "")
    words = re.findall(r"[A-Za-z][A-Za-z-]+", title)
    for w in words:
        wl = w.lower()
        if (
            len(wl) > 2
            and wl not in STOPWORDS
            and not wl.startswith("260")
            and not wl.startswith("250")
        ):
            word_counter[wl] += 1
top_words = word_counter.most_common(30)

nodes = []
edges = []
edge_set = set()
venue_groups = defaultdict(list)
catyear_groups = defaultdict(list)

for i, e in enumerate(entries):
    d = e.get("date", "")
    year = d[:4] if d and len(d) >= 4 else "2026"
    nodes.append(
        {
            "id": i,
            "title": e.get("title", ""),
            "cat": e.get("category", "unknown"),
            "sub": e.get("subcategory", "unknown"),
            "year": year,
            "url": e.get("url", ""),
            "venue": e.get("venue", ""),
        }
    )
    v = e.get("venue", "")
    if v:
        venue_groups[v].append(i)
    catyear_groups[(e.get("category", "unknown"), year)].append(i)

for v, ids in venue_groups.items():
    if len(ids) < 2:
        continue
    ids_sorted = sorted(ids)
    for idx in range(len(ids_sorted)):
        for jdx in range(idx + 1, min(idx + 6, len(ids_sorted))):
            key = (ids_sorted[idx], ids_sorted[jdx])
            if key not in edge_set:
                edge_set.add(key)
                edges.append(
                    {
                        "source": ids_sorted[idx],
                        "target": ids_sorted[jdx],
                        "strength": 0.5,
                    }
                )

for (cat, yr), ids in catyear_groups.items():
    if len(ids) < 2 or len(ids) > 20:
        continue
    ids_sorted = sorted(ids)
    for idx in range(len(ids_sorted)):
        for jdx in range(idx + 1, min(idx + 3, len(ids_sorted))):
            key = (ids_sorted[idx], ids_sorted[jdx])
            if key not in edge_set:
                edge_set.add(key)
                edges.append(
                    {
                        "source": ids_sorted[idx],
                        "target": ids_sorted[jdx],
                        "strength": 0.2,
                    }
                )

node_degree = Counter()
for edge in edges:
    s = edge["source"]
    t = edge["target"]
    if isinstance(s, dict):
        s = s["id"]
    if isinstance(t, dict):
        t = t["id"]
    node_degree[s] += 1
    node_degree[t] += 1

max_degree = max(node_degree.values()) if node_degree else 1
for i, n in enumerate(nodes):
    n["degree"] = node_degree.get(i, 0)
    n["normDegree"] = round(node_degree.get(i, 0) / max_degree, 3)

num_nodes = len(nodes)
num_edges = len(edges)
max_possible = num_nodes * (num_nodes - 1) / 2
graph_density = round(num_edges / max_possible, 6) if max_possible > 0 else 0
avg_degree = round(sum(node_degree.values()) / num_nodes, 2) if num_nodes > 0 else 0

print(
    f"Force graph: {num_nodes} nodes, {num_edges} edges (density={graph_density}, avg_deg={avg_degree})"
)

sorted_entries = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)
recent = sorted_entries[:20]
recent_rows = ""
for e in recent:
    date = e.get("date", "")
    title = e.get("title", "")
    cat = e.get("category", "")
    sub = e.get("subcategory", "")
    url = e.get("url", "")
    row = f'<tr><td>{date}</td><td><a href="{url}" target="_blank">{title}</a></td><td>{cat}</td><td>{sub}</td></tr>\n'
    recent_rows += row


def js_str(s):
    return json.dumps(s)


HTML = (
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Learning Research - Graph Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0f1117; color: #e1e4e8; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; }
h1 { font-size: 1.8rem; margin-bottom: 8px; color: #f0f6fc; }
h2 { font-size: 1.2rem; margin: 24px 0 12px; color: #79c0ff; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
.stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.stat-card .num { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.stat-card .label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; overflow: hidden; }
.chart-box.full { grid-column: 1 / -1; }
canvas { max-height: 400px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; padding: 8px 10px; border-bottom: 2px solid #30363d; color: #79c0ff; font-weight: 600; }
td { padding: 7px 10px; border-bottom: 1px solid #21262d; }
tr:hover td { background: #1c2128; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
#forceGraph { width: 100%; height: 550px; background: #0d1117; border-radius: 6px; cursor: grab; }
#forceGraph:active { cursor: grabbing; }
.tooltip { position: absolute; background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; font-size: 0.8rem; pointer-events: none; color: #e1e4e8; max-width: 300px; z-index: 100; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
@media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>Agent Learning Research Papers</h1>
<p>Analysis of """
    + str(total)
    + """ papers on learning in AI</p>

<div class="stats-grid">
  <div class="stat-card"><div class="num">"""
    + str(total)
    + """</div><div class="label">Total Papers</div></div>
  <div class="stat-card"><div class="num">"""
    + str(num_nodes)
    + """</div><div class="label">Graph Nodes</div></div>
  <div class="stat-card"><div class="num">"""
    + str(num_edges)
    + """</div><div class="label">Graph Edges</div></div>
  <div class="stat-card"><div class="num">"""
    + str(graph_density)
    + """</div><div class="label">Graph Density</div></div>
  <div class="stat-card"><div class="num">"""
    + str(avg_degree)
    + """</div><div class="label">Avg Degree</div></div>
</div>

<h2>Paper Relationship Graph</h2>
<div class="chart-box full">
<div class="legend">
  """
    + "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{CATEGORY_COLORS[c]}"></span> {c}</span>'
        for c in CATEGORY_ORDER
    )
    + """
</div>
<svg id="forceGraph"></svg>
<div id="tooltip" class="tooltip" style="display:none"></div>
</div>

<div class="chart-row">
  <div class="chart-box"><canvas id="catChart"></canvas></div>
  <div class="chart-box"><canvas id="subcatChart"></canvas></div>
</div>

<h2>Publication Timeline</h2>
<div class="chart-box full">
<canvas id="timelineChart"></canvas>
</div>

<h2>Cumulative Growth</h2>
<div class="chart-box full">
<canvas id="cumulativeChart"></canvas>
</div>

<h2>Most Common Title Keywords</h2>
<div class="chart-box full">
<canvas id="wordChart"></canvas>
</div>

<h2>20 Most Recent Papers</h2>
<table>
<thead><tr><th>Date</th><th>Title</th><th>Category</th><th>Subcategory</th></tr></thead>
<tbody>"""
    + recent_rows
    + """</tbody>
</table>

<script>
const catLabels = """
    + js_str(CATEGORY_ORDER)
    + """;
const catValues = """
    + js_str(cat_vals)
    + """;
const catColors = """
    + js_str(cat_color_vals)
    + """;

const subcatLabels = """
    + js_str(SUBCATEGORY_ORDER)
    + """;
const subcatValues = """
    + js_str(subcat_vals)
    + """;
const subcatColors = """
    + js_str(
        [
            "#58a6ff",
            "#3fb950",
            "#d29922",
            "#f0883e",
            "#db6d8a",
            "#7ee787",
            "#a5d6ff",
            "#ff7b72",
        ]
    )
    + """;

const tlLabels = """
    + js_str(ym_set)
    + """;
const tlTotal = """
    + js_str(tl_total)
    + """;
const cumTotal = """
    + js_str(cum_total)
    + """;

const graphNodes = """
    + js_str(nodes)
    + """;
const graphEdges = """
    + js_str(edges)
    + """;

const catColorMap = """
    + js_str(CATEGORY_COLORS)
    + """;

const wordLabels = """
    + js_str([w for w, _ in top_words])
    + """;
const wordValues = """
    + js_str([n for _, n in top_words])
    + """;

const width = document.getElementById('forceGraph').clientWidth;
const height = 550;

const svg = d3.select('#forceGraph').attr('viewBox', [0, 0, width, height]);
const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', (event) => { g.attr('transform', event.transform); }));
const tooltip = d3.select('#tooltip');

const sim = d3.forceSimulation(graphNodes)
  .force('link', d3.forceLink(graphEdges).id(d => d.id).distance(60).strength(d => d.strength || 0.3))
  .force('charge', d3.forceManyBody().strength(-25))
  .force('center', d3.forceCenter(width/2, height/2))
  .force('collision', d3.forceCollide().radius(d => 3 + d.normDegree * 10));

g.append('g').selectAll('line').data(graphEdges).join('line')
  .attr('stroke', '#30363d').attr('stroke-width', d => 0.3 + (d.strength || 0.3) * 0.8).attr('stroke-opacity', d => 0.15 + (d.strength || 0.3) * 0.5);

g.append('g').selectAll('circle').data(graphNodes).join('circle')
  .attr('r', d => 3 + d.normDegree * 10)
  .attr('fill', d => catColorMap[d.cat] || '#8b949e')
  .attr('stroke', '#0d1117').attr('stroke-width', 1).attr('opacity', 0.8)
  .call(d3.drag()
    .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
  .on('mouseover', (event, d) => { tooltip.style('display','block').html('<strong>'+d.title+'</strong><br>'+d.cat+' / '+d.sub+' | '+d.year).style('left',(event.pageX+12)+'px').style('top',(event.pageY-10)+'px'); })
  .on('mouseout', () => tooltip.style('display','none'))
  .on('click', (event, d) => { if (d.url) window.open(d.url,'_blank'); });

sim.on('tick', () => {
  g.selectAll('line').attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  g.selectAll('circle').attr('cx',d=>d.x).attr('cy',d=>d.y);
});

new Chart(document.getElementById('catChart'), {
  type:'bar', data:{labels:catLabels, datasets:[{label:'Papers',data:catValues,backgroundColor:catColors,borderRadius:4}]},
  options:{responsive:true,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,ticks:{color:'#8b949e'},grid:{color:'#21262d'}},y:{ticks:{color:'#8b949e',font:{size:11}},grid:{color:'#21262d'}}}}
});

new Chart(document.getElementById('subcatChart'), {
  type:'bar', data:{labels:subcatLabels, datasets:[{label:'Papers',data:subcatValues,backgroundColor:subcatColors,borderRadius:4}]},
  options:{responsive:true,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,ticks:{color:'#8b949e'},grid:{color:'#21262d'}},y:{ticks:{color:'#8b949e',font:{size:11}},grid:{color:'#21262d'}}}}
});

new Chart(document.getElementById('timelineChart'), {
  type:'line', data:{labels:tlLabels, datasets:[{label:'Papers per month',data:tlTotal,borderColor:'#7ee787',backgroundColor:'rgba(126,231,135,0.12)',fill:true,tension:0.3,pointRadius:2}]},
  options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:25},grid:{color:'#21262d'}},y:{beginAtZero:true,ticks:{stepSize:1,color:'#8b949e'},grid:{color:'#21262d'}}}}
});

new Chart(document.getElementById('cumulativeChart'), {
  type:'line', data:{labels:tlLabels, datasets:[{label:'Cumulative papers',data:cumTotal,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.05)',fill:true,tension:0.3,pointRadius:1,borderWidth:2}]},
  options:{responsive:true,plugins:{legend:{labels:{color:'#8b949e'}}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:25},grid:{color:'#21262d'}},y:{beginAtZero:true,ticks:{color:'#8b949e'},grid:{color:'#21262d'}}}}
});

new Chart(document.getElementById('wordChart'), {
  type:'bar', data:{labels:wordLabels, datasets:[{label:'Occurrences',data:wordValues,backgroundColor:'#58a6ff',borderRadius:3}]},
  options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,ticks:{color:'#8b949e'},grid:{color:'#21262d'}},y:{ticks:{color:'#8b949e',font:{size:11}},grid:{color:'#21262d'}}}}
});
</script>
</body>
</html>"""
)

with open(
    os.path.join(BASE, "docs", "graph_analysis.html"), "w", encoding="utf-8"
) as f:
    f.write(HTML)
print(f"Wrote docs/graph_analysis.html ({len(HTML)} bytes)")
