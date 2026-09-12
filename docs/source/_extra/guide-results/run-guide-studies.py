"""Generate the main documentation results with graph-tiger 0.6.0.

Run: python run-guide-studies.py [attacks|defenses|epidemics|cascades|visualization]
Without an argument, run all studies. Install graph-tiger, numpy and matplotlib.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "public-0.6.0"), str(ROOT / "experiment-deps")]
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"
import csv
import json
import platform
from importlib.metadata import version
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graph_tiger import graphs
from graph_tiger.attacks import Attack, run_attack_method
from graph_tiger.defenses import Defense
from graph_tiger.diffusion import Diffusion
from graph_tiger.cascading import Cascading

OUT = ROOT / "site" / "guide-results"
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "datasets"
DATA.mkdir(exist_ok=True)
graphs.graph_dir = str(DATA) + os.sep
os.chdir(OUT)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "svg.fonttype": "path", "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2.1})
COLORS = ["#157a79", "#cf6636", "#5765b0", "#bb4670", "#789333", "#343e47"]


def panel(title, ylabel, xlabel, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.subplots_adjust(left=.12, right=.96, bottom=.24, top=.78)
    ax.text(0, 1.28, title, transform=ax.transAxes, fontsize=16, weight="bold")
    ax.set_ylabel(ylabel, rotation=0, ha="left", va="bottom", fontsize=12)
    ax.yaxis.set_label_coords(0, 1.08)
    ax.set_xlabel(xlabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=.2)
    return fig, ax


def save(fig, ax, slug, legend=True):
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18),
                  ncol=2, frameon=False, fontsize=10)
    fig.canvas.draw()
    for label in [ax.xaxis.label, ax.yaxis.label, *ax.texts]:
        box = label.get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    for extension in ["svg", "png", "pdf"]:
        fig.savefig(OUT / f"{slug}.{extension}", dpi=160)
    plt.close(fig)


def band(ax, values, label, color):
    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.plot(x, values.mean(axis=0), label=label, color=color)
    low, high = np.quantile(values, [.1, .9], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=.12)


def record(study, rows, metadata):
    with (OUT / f"{study}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata["environment"] = {p: version(p) for p in ["graph-tiger", "networkx", "numpy", "matplotlib"]}
    metadata["python"] = platform.python_version()
    (OUT / f"{study}.json").write_text(json.dumps(metadata, indent=2))
    print(f"Completed {study}: {len(rows)} observations", flush=True)


def rows_for(rows, policy, seed, values, **extra):
    for step, value in enumerate(values):
        assert np.isfinite(value)
        rows.append(dict(policy=policy, seed=seed, step=step, value=float(value), **extra))


def attacks():
    G = graphs.graph_loader("ky2")
    rows = []
    for kind in ["node", "edge"]:
        fig, ax = panel(f"KY-2: {kind} attacks", "Largest component / original nodes", f"{kind.title()}s removed", (0, 1.03))
        for prefix, color in zip(["rnd", "id", "rd", "ib", "rb"], COLORS):
            method = f"{prefix}_{kind}"
            trials = []
            for seed in range(10):
                sim = Attack(G, attack=method, steps=30, runs=1, seed=seed,
                             attack_approx=min(80, len(G)), robust_measure="largest_connected_component")
                curve = np.asarray(sim.run_single_sim()) / len(G)
                trials.append(curve)
                rows_for(rows, method, seed, curve)
            band(ax, trials, method, color)
            print(f"Attack {method} finished", flush=True)
        save(fig, ax, f"attacks-{kind}")
    record("attacks", rows, dict(graph="ky2", nodes=len(G), edges=G.number_of_edges(), seeds=list(range(10)), steps=30, attack_approx=80, interval="10th–90th percentile across runs"))


def defenses():
    G = graphs.graph_loader("ky2")
    removed = run_attack_method(G, "rb_node", k=30, approx=80, seed=17)
    damaged = G.copy()
    damaged.remove_nodes_from(removed)
    baseline = max(map(len, nx.connected_components(damaged))) / len(G)
    rows = []
    fig, ax = panel("KY-2: reconnect after the same attack", "Largest component / original nodes", "Edge interventions", (0, 1.03))
    ax.axhline(baseline, color="#343e47", ls="--", label="No repair")
    rows_for(rows, "none", 17, [baseline] * 31)
    policies = ["rewire_edge_random", "rewire_edge_random_neighbor", "rewire_edge_preferential_random", "add_edge_random", "add_edge_preferential"]
    labels = ["Random rewiring", "Random-neighbor rewiring", "Preferential-random rewiring", "Random addition", "Low-degree addition"]
    for policy, label, color in zip(policies, labels, COLORS):
        trials = []
        for seed in range(10):
            sim = Defense(damaged, attack=None, defense=policy, k_d=30, runs=1,
                          steps=30, seed=seed, robust_measure="largest_connected_component")
            values = np.asarray(sim.run_single_sim()) / len(G)
            assert abs(values[0] - baseline) < 1e-10
            trials.append(values)
            rows_for(rows, policy, seed, values)
        band(ax, trials, label, color)
        print(f"Defense {policy} finished", flush=True)
    save(fig, ax, "defenses-edge")
    record("defenses-edge", rows, dict(graph="ky2", nodes=len(G), edges=G.number_of_edges(), removed=list(map(int, removed)), attack="rb_node", attack_seed=17, attack_approx=80, defense_seeds=list(range(10)), k_d=30, interval="10th–90th percentile across runs"))
    G = graphs.karate()
    rows = []
    fig, ax = panel("Karate club: protect three nodes", "Largest component / original nodes", "Attack attempts", (0, 1.03))
    for method, label, color in zip([None, "rnd_node", "ns_node"], ["No protection", "Random protection", "NetShield protection"], COLORS):
        trials = []
        for seed in range(20):
            sim = Attack(G, attack="rd_node", steps=8, defense=method,
                         k_d=0 if method is None else 3, runs=1, seed=seed)
            values = np.asarray(sim.run_single_sim()) / len(G)
            trials.append(values)
            rows_for(rows, method or "none", seed, values)
        band(ax, trials, label, color)
    save(fig, ax, "defenses-node")
    record("defenses-node", rows, dict(graph="karate", nodes=len(G), attack="rd_node", k_d=3, seeds=list(range(20)), steps=8))


def epidemics():
    G = graphs.graph_loader("BA", n=100, m=3, seed=17)
    rows = []
    for method, title in [(None, "No vaccination"), ("rnd_node", "Random vaccination"), ("ns_node", "NetShield vaccination")]:
        fig, ax = panel(f"SIS: {title.lower()}", "Infected fraction", "Simulation steps", (0, 1.03))
        for b, color in zip([.02, .04, .08], COLORS):
            trials = []
            for seed in range(20):
                sim = Diffusion(G, model="SIS", b=b, d=.2, c=.1, runs=1, steps=150, seed=seed,
                                diffusion=None if method is None else "min", method=method, k=0 if method is None else 5)
                values = np.asarray(sim.run_single_sim()) / len(G)
                trials.append(values)
                rows_for(rows, method or "none", seed, values, b=b)
            band(ax, trials, f"b = {b:.2f}", color)
        save(fig, ax, f"epidemics-{method or 'none'}")
    record("epidemics", rows, dict(graph="BA", n=100, m=3, graph_seed=17, b=[.02,.04,.08], d=.2, c=.1, k=5, steps=150, seeds=list(range(20)), interval="10th–90th percentile across runs", initial_infection="Vaccinated initial infections are removed, not replaced."))
    G = graphs.karate()
    sim = Diffusion(G, model="SIR", b=.08, d=.2, c=.1, runs=1, steps=150, seed=17)
    sim.run_single_sim()
    infected = np.array([sim.sim_info[t]["failed"] for t in range(151)]) / len(G)
    recovered = np.array([sim.sim_info[t]["recovered"] for t in range(151)]) / len(G)
    susceptible = 1 - infected - recovered
    fig, ax = panel("Karate club: one SIR outbreak", "Fraction of original nodes", "Simulation steps", (0, 1.03))
    rows = []
    for label, values, color in zip(["Susceptible", "Infected", "Recovered"], [susceptible, infected, recovered], COLORS):
        ax.plot(values, label=label, color=color)
        rows_for(rows, label, 17, values)
    assert np.allclose(susceptible + infected + recovered, 1)
    save(fig, ax, "epidemics-sir")
    record("epidemics-sir", rows, dict(graph="karate", b=.08, d=.2, c=.1, seed=17, steps=150, final_recovered=float(recovered[-1]), final_infected=float(infected[-1]), peak_infected=float(max(infected))))
    rows = []
    fig, ax = panel("Karate club: adding five contact edges", "Infected fraction", "Simulation steps", (0, 1.03))
    for method, label, color in zip([None, "add_edge_random"], ["Unchanged network", "Five random edges added"], COLORS):
        trials = []
        for seed in range(20):
            sim = Diffusion(G, model="SIS", b=.08, d=.2, c=.1, runs=1, steps=150, seed=seed,
                            diffusion="max" if method else None, method=method, k=5 if method else 0)
            values = np.asarray(sim.run_single_sim()) / len(G)
            trials.append(values)
            rows_for(rows, method or "none", seed, values)
        band(ax, trials, label, color)
    save(fig, ax, "epidemics-spread")
    record("epidemics-spread", rows, dict(graph="karate", b=.08, d=.2, c=.1, seeds=list(range(20)), steps=150, k=5))


def cascades():
    G = graphs.graph_loader("electrical")
    rows = []
    fig, ax = panel("Power network: local load sharing", "Cumulative failed nodes", "Redistribution rounds")
    fig2, ax2 = panel("Power network: untransferred load", "Cumulative lost load (degree-based units)", "Redistribution rounds")
    for beta, color in zip([0, 1, 2], COLORS):
        sim = Cascading(G, model="local_load_sharing", beta=beta, r=.2, attack="id_node", k_a=1, runs=1, steps=20, seed=7)
        sim.run_single_sim()
        for field, axis in [("failed", ax), ("lost_load", ax2)]:
            values = [sim.sim_info[t][field] for t in range(21)]
            axis.plot(values, label=f"β = {beta}", color=color, marker="o", markersize=3)
            rows_for(rows, f"beta={beta}", 7, values, outcome=field)
        print(f"Local load sharing beta={beta} finished", flush=True)
    save(fig, ax, "cascades-local-failed")
    save(fig2, ax2, "cascades-local-shed")
    record("cascades-local", rows, dict(graph="electrical", nodes=len(G), edges=G.number_of_edges(), r=.2, beta=[0,1,2], attack="id_node", k_a=1, seed=7, steps=20, capacity="(1+r) * intact degree"))
    G = graphs.karate()
    rows = []
    for model, title, ylabel in [("motter_lai", "Motter–Lai: structural connectivity", "Largest component / original nodes"), ("crucitti", "Crucitti: weighted service", "Weighted efficiency / intact efficiency")]:
        fig, ax = panel(title, ylabel, "Cascade rounds", (0, 1.1))
        for r, color in zip([.2, .5, 1.], COLORS):
            sim = Cascading(G, model=model, r=r, attack="id_node", k_a=1, runs=1, steps=20, seed=7)
            values = np.asarray(sim.run_single_sim())
            values = values / (sim.get_efficiency(G) if model == "crucitti" else len(G))
            ax.plot(values, label=f"r = {r:g}", color=color, marker="o", markersize=3)
            rows_for(rows, model, 7, values, r=r)
        save(fig, ax, f"cascades-{model}")
    record("cascades-global", rows, dict(graph="karate", nodes=len(G), r=[.2,.5,1.], attack="id_node", k_a=1, seed=7, steps=20, note="Different model-specific outcomes; step 0 is after initial attack. Crucitti denominator uses surviving nodes before division by intact efficiency."))
def visualization():
    G = graphs.graph_loader("ky2")
    pos = nx.get_node_attributes(G, "pos")
    if len(pos) != len(G):
        pos = nx.spring_layout(G, seed=17)
    removed = run_attack_method(G, "rd_node", k=30, seed=17)
    rows = []
    for k in [0, 10, 30]:
        H = G.copy()
        H.remove_nodes_from(removed[:k])
        largest = max(nx.connected_components(H), key=len)
        fig, ax = plt.subplots(figsize=(8, 8.5))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#e4e8eb", width=.6)
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#8b9aa2", width=.9)
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=list(H), node_color=["#157a79" if n in largest else "#e4a24b" for n in H], node_size=15, linewidths=0)
        if k:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=removed[:k], node_color="#b83e58", node_shape="x", node_size=38, linewidths=1.3)
        ax.set_title(f"{k} nodes removed  ·  largest component {len(largest)}/{len(G)}", fontsize=16, pad=16)
        ax.set_aspect("equal")
        ax.axis("off")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], ls="", marker="o", color=c, label=l) for c,l in [("#157a79","Largest component"),("#e4a24b","Other components")]]
        handles.append(Line2D([], [], ls="", marker="x", color="#b83e58", label="Removed"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.015), ncol=3, frameon=False, fontsize=10)
        fig.tight_layout(pad=2)
        for extension in ["svg", "png", "pdf"]:
            fig.savefig(OUT / f"network-{k}.{extension}", dpi=160)
        plt.close(fig)
        rows.append(dict(removed=k, largest_component=len(largest), original_nodes=len(G)))
    record("visualization", rows, dict(graph="ky2", attack="rd_node", seed=17, removed=list(map(int, removed)), layout="Dataset positions, unchanged between states; faint edges show original network."))


if __name__ == "__main__":
    selected = sys.argv[1:] or ["attacks", "defenses", "epidemics", "cascades", "visualization"]
    for name in selected:
        globals()[name]()
