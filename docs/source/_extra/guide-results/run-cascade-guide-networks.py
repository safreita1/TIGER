"""Generate the cascade guide's illustrative chain and empirical routing study.

Run beside run-guide-studies.py with graph-tiger 0.6.0 installed.
"""
from run_guide_bootstrap import *
import shutil


def chain():
    # DOC-BEGIN chain
    G = nx.path_graph(["F", "A", "B", "C"])
    loads = dict(F=6, A=2, B=2, C=1)
    capacities = dict(F=6, A=5, B=8, C=12)
    sim = Cascading(G, model="local_load_sharing", allocation="degree", beta=0,
                    initial_load=loads, capacities=capacities,
                    initial_failures=["F"], runs=1, steps=3)
    states = []
    for t in range(4):
        states.append(dict(round=t, failed=sorted(sim.failed),
                           load=sim.load.copy(), lost_load=sim.lost_load))
        if t < 3:
            sim.run_local_load_sharing_step()
    # DOC-END
    assert [s['failed'] for s in states] == [['F'], ['A', 'F'], ['A', 'B', 'F'], ['A', 'B', 'F']]
    assert all(np.isclose(sum(s['load'].values()) + s['lost_load'], 11) for s in states)
    assert states[-1]['load']['C'] == 11 and states[-1]['lost_load'] == 0
    titles = ['F fails: 6 units await transfer', 'F sends 6 to A: 8 exceeds capacity 5',
              'A sends 8 to B: 10 exceeds capacity 8', 'B sends 10 to C: 11 fits capacity 12']
    for state, title in zip(states, titles):
        fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
        pos = {n: (i, 0) for i, n in enumerate(G)}
        nx.draw_networkx_edges(G, pos, ax=ax, width=2.2, edge_color='#82969b')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500,
            node_color=['#bb4670' if n in state['failed'] else '#157a79' for n in G])
        nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=17)
        for n, (x, _) in pos.items():
            ax.text(x, -.30, f"Load {state['load'][n]:g}\nCapacity {capacities[n]}", ha='center', fontsize=12)
        ax.set_title(f"Round {state['round']}\n{title}", pad=20, fontsize=15)
        ax.text(1.5, -.64, 'Pink: failed   •   Teal: functioning   •   Lines: original connections', ha='center', fontsize=10)
        ax.set(xlim=(-.5, 3.5), ylim=(-.8, .5)); ax.axis('off')
        for ext in ['svg', 'png', 'pdf']:
            fig.savefig(OUT / f"chain-cascade-{state['round']}.{ext}", dpi=160)
        plt.close(fig)
    (OUT / 'chain-cascade.json').write_text(json.dumps(states, indent=2), encoding='utf8')


def routing():
    # DOC-BEGIN routing
    power = graphs.graph_loader("electrical")
    center = max(power, key=power.degree)
    G = nx.ego_graph(power, center, radius=2)
    nx.set_edge_attributes(G, 1.0, "weight")
    results = {}
    for model in ["motter_lai", "crucitti"]:
        for r in [0.2, 0.5, 1.0]:
            sim = Cascading(G, model=model, r=r, attack="id_node", k_a=1,
                            runs=1, steps=200, seed=7)
            raw = np.asarray(sim.run_single_sim())
            baseline = sim.get_efficiency(G) if model == "crucitti" else len(G)
            results[model, r] = raw / baseline
    # DOC-END
    assert len(G) == 45 and G.number_of_edges() == 71
    assert nx.is_connected(G)
    rows, summary = [], {}
    for model, ylabel in [('motter_lai', 'Largest component / original nodes'),
                          ('crucitti', 'Weighted efficiency / intact efficiency')]:
        fig, ax = panel(model.replace('_', ' ').title() + ': power-grid subnetwork', ylabel, 'Cascade rounds')
        summary[model] = {}
        for r, color in zip([.2, .5, 1.], COLORS):
            values = results[model, r]
            assert np.isfinite(values).all() and (values >= 0).all()
            if model == 'motter_lai':
                assert np.all(np.diff(values) <= 1e-12)
            ax.plot(values, label=f'r = {r:g}', color=color)
            rows_for(rows, model, 7, values, r=r)
            summary[model][str(r)] = dict(final=float(values[-1]), tail_range=float(np.ptp(values[-20:])))
        save(fig, ax, 'power-routing-' + model)
    pos = nx.spring_layout(G, seed=17, weight=None)
    attacked = run_attack_method(G, 'id_node', k=1, seed=7)[0]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.6, edge_color='#a2b4ba')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color=['#bb4670' if n == attacked else '#157a79' for n in G])
    ax.set_title('Power-grid subnetwork: 45 nodes, 71 connections', fontsize=15, pad=18)
    ax.text(.5, -.02, 'Pink: initially attacked node  •  Layout shows topology, not geography', transform=ax.transAxes, ha='center', fontsize=10)
    ax.axis('off')
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT / f'power-routing-network.{ext}', dpi=160)
    plt.close(fig)
    record('power-routing', rows, dict(graph='electrical', original_nodes=len(power),
        center=center, radius=2, nodes=list(G), edges=list(G.edges), attacked=attacked,
        steps=200, seed=7, networkx=nx.__version__, results=summary,
        boundary='Induced two-hop subnetwork: all connections to outside nodes are excluded.'))
    print(json.dumps(summary))


if __name__ == '__main__':
    chain()
    routing()
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    print('Validated chain propagation, workload conservation, and six power-subnetwork trajectories.')
