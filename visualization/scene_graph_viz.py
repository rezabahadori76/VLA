from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def export_scene_graph_visuals(scene_graph_path: Path, output_png: Path, enabled: bool = True) -> None:
    if not enabled or not scene_graph_path.exists():
        return

    payload = json.loads(scene_graph_path.read_text(encoding='utf-8'))
    g = nx.DiGraph()
    for n in payload.get('nodes', []):
        g.add_node(n['id'], label=n.get('label', n['id']), ntype=n.get('type', 'unknown'))
    for e in payload.get('edges', []):
        g.add_edge(e['source'], e['target'], relation=e.get('relation', 'rel'))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=7)
    colors = ['#66c2a5' if g.nodes[n].get('ntype') == 'room' else '#8da0cb' for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=550)
    nx.draw_networkx_edges(g, pos, arrows=True, alpha=0.4)
    nx.draw_networkx_labels(g, pos, labels={n: g.nodes[n].get('label', n) for n in g.nodes}, font_size=8)
    plt.title('Scene Graph (Phase 1)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
