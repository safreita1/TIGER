"""Generate the data and figures used by the influence-simulation guide."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graph_tiger.influence import Influence


HERE = Path(__file__).resolve().parent
OUT = HERE if HERE.name == 'guide-results' else HERE / 'site' / 'guide-results'
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    None: '#dce4e6',
    0: '#dce4e6',
    1: '#e56b3f',
    'claim': '#d85864',
    'correction': '#3979aa'
}


def save_figure(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def state_at(simulation, step):
    return dict(zip(simulation.node_order, simulation.sim_info[step]['status']))


def draw_state(ax, graph, positions, state, title, labels=True,
               edge_labels=None, directed=False):
    edge_options = {'arrows': True, 'arrowsize': 16, 'arrowstyle': '-|>'} \
        if directed else {}
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edge_color='#9aa8ae', width=1.8,
        **edge_options
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_size=660,
        node_color=[COLORS[state[node]] for node in graph.nodes],
        edgecolors='#24484f', linewidths=1.2
    )
    if labels:
        nx.draw_networkx_labels(graph, positions, ax=ax, font_size=10)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph, positions, edge_labels=edge_labels, ax=ax,
            font_size=8, rotate=False, label_pos=0.45
        )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_axis_off()


rows = []
parameters = {}

# Independent cascade: newly active nodes receive one propagation opportunity
# in the following round.
ic_graph = nx.DiGraph([('A', 'B'), ('B', 'C'), ('B', 'D')])
ic = Influence(ic_graph, model='independent_cascade', seeds={'A'},
               probability=1.0, runs=1, steps=2, seed=17)
ic.run_single_sim()
ic_pos = {'A': (0, 0), 'B': (1, 0), 'C': (2, .55), 'D': (2, -.55)}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
for ax, step in zip(axes, [0, 1, 2]):
    draw_state(ax, ic_graph, ic_pos, state_at(ic, step),
               f'Round {step}', directed=True)
    for node, state in state_at(ic, step).items():
        rows.append(['independent_cascade', 0, step, node, state])
fig.suptitle('Independent cascade: activation advances one frontier per round',
             fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, 'influence-independent-cascade')
parameters['independent_cascade'] = {
    'graph': 'A→B, B→C, B→D', 'seeds': ['A'], 'probability': 1.0,
    'steps': 2, 'seed': 17
}

# Linear threshold: C requires reinforcement from both active predecessors.
lt_graph = nx.DiGraph()
lt_graph.add_weighted_edges_from([
    ('A', 'C', .35), ('B', 'C', .35), ('C', 'D', 1.0)
])
thresholds = {'A': 1.0, 'B': 1.0, 'C': .60, 'D': .80}
lt = Influence(lt_graph, model='linear_threshold', seeds={'A', 'B'},
               threshold=thresholds, runs=1, steps=2, seed=17)
lt.run_single_sim()
lt_pos = {'A': (0, .55), 'B': (0, -.55), 'C': (1.15, 0), 'D': (2.3, 0)}
edge_labels = {(u, v): f'w={data["weight"]:.2g}'
               for u, v, data in lt_graph.edges(data=True)}
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
for ax, step in zip(axes, [0, 1, 2]):
    draw_state(ax, lt_graph, lt_pos, state_at(lt, step),
               f'Round {step}', edge_labels=edge_labels, directed=True)
    ax.text(lt_pos['C'][0], lt_pos['C'][1] - .35, 'threshold = 0.60',
            ha='center', va='top', fontsize=8, color='#36545a')
    for node, state in state_at(lt, step).items():
        rows.append(['linear_threshold', 0, step, node, state])
fig.suptitle('Linear threshold: combined influence activates C, then D',
             fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, 'influence-linear-threshold')
parameters['linear_threshold'] = {
    'edges': [['A', 'C', .35], ['B', 'C', .35], ['C', 'D', 1.0]],
    'seeds': ['A', 'B'], 'thresholds': thresholds, 'steps': 2, 'seed': 17
}

# Voter model: one node copies one neighbor per asynchronous update.
voter_graph = nx.cycle_graph(20)
initial_state = {node: int(node >= 10) for node in voter_graph}
voter_curves = []
sample = None
for run in range(40):
    simulation = Influence(voter_graph, model='voter',
                           initial_state=initial_state, tracked_state=1,
                           runs=1, steps=100, seed=run)
    curve = simulation.run_single_sim()
    voter_curves.append(curve)
    if run == 17:
        sample = simulation
    for step, count in enumerate(curve):
        rows.append(['voter', run, step, 'tracked_state_1', count])
voter_curves = np.asarray(voter_curves, dtype=float) / len(voter_graph)
voter_pos = nx.circular_layout(voter_graph)
fig = plt.figure(figsize=(14, 4.2))
grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.3])
for column, step in enumerate([0, 20, 100]):
    draw_state(fig.add_subplot(grid[0, column]), voter_graph, voter_pos,
               state_at(sample, step), f'Update {step}', labels=False)
ax = fig.add_subplot(grid[0, 3])
x = np.arange(voter_curves.shape[1])
ax.plot(x, voter_curves.mean(axis=0), color='#087f73', linewidth=2,
        label='Mean')
lower, upper = np.quantile(voter_curves, [.1, .9], axis=0)
ax.fill_between(x, lower, upper, color='#087f73', alpha=.18,
                label='10th–90th percentile')
ax.axhline(.5, color='#9aa8ae', linestyle=':', linewidth=1)
ax.set(xlabel='Asynchronous node updates', ylabel='Fraction in state 1',
       xlim=(0, 100), ylim=(0, 1))
ax.legend(frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Voter model: local copying can move the population in either direction',
             fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, 'influence-voter')
parameters['voter'] = {
    'graph': '20-node cycle', 'initial_state': 'nodes 0–9: 0; 10–19: 1',
    'tracked_state': 1, 'runs': 40, 'steps': 100, 'seeds': list(range(40)),
    'displayed_run': 17
}

# Competitive cascade: both messages reach the middle node in the same round;
# the declared priority resolves the collision.
competitive_graph = nx.path_graph(9)
competitive = Influence(
    competitive_graph, model='competitive_cascade',
    message_seeds={'claim': {0}, 'correction': {8}}, probability=1.0,
    tracked_state='correction', tie_break='priority',
    priority=['correction', 'claim'], runs=1, steps=4, seed=17
)
competitive.run_single_sim()
competitive_pos = {node: (node, 0) for node in competitive_graph}
fig, axes = plt.subplots(3, 1, figsize=(12.5, 5.5))
for ax, step in zip(axes, [0, 2, 4]):
    draw_state(ax, competitive_graph, competitive_pos,
               state_at(competitive, step), f'Round {step}')
    if step == 4:
        ax.text(4, -.28, 'simultaneous arrival → correction wins',
                ha='center', va='top', fontsize=9, color='#36545a')
    for node, state in state_at(competitive, step).items():
        rows.append(['competitive_cascade', 0, step, node, state])
fig.suptitle('Competitive cascade: a stated rule resolves simultaneous arrivals',
             fontsize=14, y=1.01)
fig.tight_layout()
save_figure(fig, 'influence-competitive-cascade')
parameters['competitive_cascade'] = {
    'graph': '9-node path',
    'message_seeds': {'claim': [0], 'correction': [8]},
    'probability': 1.0, 'tie_break': 'priority',
    'priority': ['correction', 'claim'], 'steps': 4, 'seed': 17
}

with (OUT / 'influence.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['model', 'run', 'step', 'node_or_outcome', 'value'])
    writer.writerows(rows)

(OUT / 'influence.json').write_text(
    json.dumps(parameters, indent=2, sort_keys=True), encoding='utf-8'
)

print(f'Wrote influence guide figures and {len(rows)} data rows to {OUT}')
