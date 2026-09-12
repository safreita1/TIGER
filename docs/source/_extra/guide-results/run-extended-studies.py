"""Generate the extended documentation experiments with graph-tiger 0.6.0.

The published download is named run-extended-studies.py. Keep it beside
run-guide-studies.py and run_guide_bootstrap.py.
Each function's experiment block is also displayed in the guide.
"""
from run_guide_bootstrap import *


def preventive_defense():
    # DOC-BEGIN preventive
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")
    order = run_attack_method(G, "rb_node", k=8, seed=17)
    policies = [None, "add_edge_random", "add_edge_preferential", "rewire_edge_random"]
    results = {}
    for policy in policies:
        trials = []
        for seed in range(20):
            H = G.copy()
            if policy:
                edits = run_defense_method(H, policy, k=5, seed=seed)
                if policy.startswith("rewire_"):
                    for removed, added in zip(edits["removed"], edits["added"]):
                        H.remove_edge(*removed)
                        H.add_edge(*added)
                else:
                    H.add_edges_from(edits["added"])
            curve = [run_measure(H, "largest_connected_component") / len(G)]
            for node in order:
                H.remove_node(node)
                curve.append(run_measure(H, "largest_connected_component") / len(G))
            trials.append(curve)
        results[policy or "none"] = trials
    # DOC-END
    fig, ax = panel("Protect topology before the attack", "Largest component / original nodes", "Nodes removed", (0, 1.03))
    rows = []
    for (policy, trials), color in zip(results.items(), COLORS):
        band(ax, trials, policy.replace('_', ' ') if policy != 'none' else 'No defense', color)
        for seed, curve in enumerate(trials):
            rows_for(rows, policy, seed, curve)
    save(fig, ax, "preventive-defense")
    record("preventive-defense", rows, dict(graph="unweighted karate", attack="rb_node", attack_seed=17, attack_order=order, k_d=5, seeds=list(range(20)), threat="Fixed order computed on intact graph; no retargeting after defense."))
    pos = nx.spring_layout(G, seed=17, weight=None)
    H = G.copy()
    edits = run_defense_method(H, "add_edge_preferential", k=5, seed=17)
    H.add_edges_from(edits['added'])
    network_panel(G, pos, {n: COLORS[0] for n in G}, "Before defense", "defense-before", [])
    network_panel(H, pos, {n: COLORS[0] for n in H}, "Five low-degree-pair edges added", "defense-after", edits['added'])


def sir_ensemble():
    # DOC-BEGIN sir-ensemble
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")
    outbreaks, infected_runs = [], []
    for seed in range(100):
        sim = Diffusion(G, model="SIR", b=0.08, d=0.2, c=0.1,
                        runs=1, steps=300, seed=seed)
        sim.run_single_sim()
        outbreaks.append(sim.sim_info[300]["recovered"] / len(G))
        infected_runs.append([sim.sim_info[t]["failed"] / len(G) for t in range(301)])
    # DOC-END
    fig, ax = panel("SIR: outbreak sizes over 100 runs", "Number of runs", "Final recovered fraction", None)
    ax.hist(outbreaks, bins=np.linspace(0, 1, 18), color=COLORS[0], edgecolor='white')
    save(fig, ax, "sir-outbreak-distribution", legend=False)
    fig, ax = panel("SIR: prevalence across 100 runs", "Infected fraction", "Simulation steps", (0, .7))
    band(ax, infected_runs, "Mean and 10th–90th percentile", COLORS[1])
    ax.set_xlim(0, 80)
    save(fig, ax, "sir-ensemble-prevalence")
    rows = []
    for seed, value in enumerate(outbreaks):
        rows.append(dict(seed=seed, final_recovered_fraction=float(value), final_infected_fraction=float(infected_runs[seed][-1])))
    assert max(r['final_infected_fraction'] for r in rows) == 0
    record('sir-ensemble', rows, dict(graph="unweighted karate", seeds=list(range(100)), b=.08, d=.2, c=.1, steps=300, initial_infected=3, fadeouts_without_secondary=int(sum(np.isclose(outbreaks, 3/34))), mean_outbreak=float(np.mean(outbreaks))))
    trajectories = []
    for seed, values in enumerate(infected_runs):
        rows_for(trajectories, "SIR", seed, values)
    record("sir-prevalence", trajectories, dict(graph="unweighted karate", seeds=list(range(100)),
           b=.08, d=.2, c=.1, steps=300, value="infected fraction",
           summary="Mean and 10th–90th percentiles across seeds at each step."))


def complete_local():
    # DOC-BEGIN complete-local
    G = graphs.graph_loader("electrical")
    results = {}
    for beta in [0, 1, 2]:
        sim = Cascading(G, model="local_load_sharing", beta=beta, r=0.2,
                        attack="id_node", k_a=1, runs=1, steps=200, seed=7)
        sim.run_single_sim()
        results[beta] = {
            "failed": [sim.sim_info[t]["failed"] for t in range(201)],
            "lost_load": [sim.sim_info[t]["lost_load"] for t in range(201)],
            "pending": len(sim.failed - sim.processed),
        }
    # DOC-END
    rows = []
    completion={}
    for beta, result in results.items():
        assert result['pending']==0
        assert result['failed'][-1]==result['failed'][-2]
        assert np.isclose(result['lost_load'][-1],result['lost_load'][-2])
        changes=[t for t in range(1,201) if result['failed'][t]!=result['failed'][t-1] or not np.isclose(result['lost_load'][t],result['lost_load'][t-1])]
        completion[beta]=max(changes,default=0)
        for field in ['failed','lost_load']:
            rows_for(rows,f'beta={beta}',7,result[field],outcome=field)
    for field,ylabel,slug in [('failed','Cumulative failed nodes','failed'),('lost_load','Cumulative lost load (degree-based units)','shed_load')]:
        fig,ax=panel('Local sharing: completed cascades',ylabel,'Redistribution rounds')
        for (beta,result),color in zip(results.items(),COLORS):
            ax.plot(result[field],label=f'β = {beta}',color=color)
        ax.set_xlim(0,max(completion.values())+8)
        save(fig,ax,'local-complete-'+slug)
    record('local-complete',rows,dict(graph='electrical',nodes=len(G),edges=G.number_of_edges(),r=.2,beta=[0,1,2],seed=7,steps=200,last_change_round=completion,completion='No unprocessed failed nodes and unchanged failure/lost-load totals.',final_failures={b:v['failed'][-1] for b,v in results.items()}))


def network_panel(G,pos,colors,title,slug,added=None):
    from matplotlib.lines import Line2D
    fig,ax=plt.subplots(figsize=(8,6.5))
    nx.draw_networkx_edges(G,pos,ax=ax,width=1.2,edge_color='#b6c3c8')
    if added:
        nx.draw_networkx_edges(G,pos,ax=ax,edgelist=added,width=3,edge_color='#cf6636',style='dashed')
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=[colors[n] for n in G],node_size=140,edgecolors='white',linewidths=.8)
    ax.set_title(title,fontsize=17,pad=18)
    ax.axis('off')
    ax.set_aspect('equal')
    if slug.startswith('sir-state'):
        legend=[('#157a79','Susceptible'),('#cf6636','Infected'),('#5765b0','Recovered')]
    elif slug.startswith('cascade-state'):
        legend=[('#157a79','Functioning'),('#bb4670','Failed')]
    else:
        legend=[('#157a79','Node')]
    handles=[Line2D([],[],color=color,marker='o',linestyle='',label=label) for color,label in legend]
    if added:
        handles.append(Line2D([],[],color='#cf6636',linestyle='--',linewidth=3,label='Added edge'))
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.08),ncol=len(handles),frameon=False,fontsize=10)
    fig.tight_layout(pad=2)
    for ext in ['svg','png','pdf']:
        fig.savefig(OUT/f'{slug}.{ext}',dpi=160)
    plt.close(fig)


def process_snapshots():
    # DOC-BEGIN epidemic-snapshots
    G = nx.barabasi_albert_graph(60, 2, seed=17)
    pos = nx.spring_layout(G, seed=17, weight=None)
    states = {}
    counts = []
    for t in [0, 3, 10, 30]:
        sim = Diffusion(G, model="SIR", b=0.15, d=0.08, c=0.1,
                        runs=1, steps=t, seed=17)
        sim.run_single_sim()
        recovered = set(sim.vaccinated)
        infected = {n: n in sim.infected for n in G}
        counts.append(dict(step=t, infected=len(sim.infected), recovered=len(recovered)))
        states[t] = {n: "#5765b0" if n in recovered else
                     "#cf6636" if infected[n] else "#157a79" for n in G}
    # DOC-END
    for t, colors in states.items():
        network_panel(G,pos,colors,f'SIR · step {t}',f'sir-state-{t}')
    record('sir-snapshots',counts,dict(graph='BA',n=60,m=2,graph_seed=17,seed=17,b=.15,d=.08,c=.1,steps=30,selection='Fixed seed and snapshot times; same realization rerun to each horizon.'))
    # DOC-BEGIN cascade-snapshots
    G = nx.karate_club_graph()
    nx.set_edge_attributes(G, 1.0, "weight")
    pos = nx.spring_layout(G, seed=17, weight=None)
    cascade = Cascading(G, model="local_load_sharing", beta=0, r=0.2,
                        attack="id_node", k_a=1, runs=1, steps=3, seed=7)
    states = {}
    for t in range(4):
        states[t] = {n: "#bb4670" if n in cascade.failed else "#157a79" for n in G}
        if t < 3:
            cascade.run_local_load_sharing_step()
    # DOC-END
    for t, colors in states.items():
        network_panel(G,pos,colors,f'Local cascade · round {t}',f'cascade-state-{t}')
    record('cascade-snapshots',[dict(step=t,failed=sum(c=='#bb4670' for c in colors.values())) for t,colors in states.items()],dict(graph='unweighted karate',beta=0,r=.2,seed=7,attack='id_node',k_a=1,manual_steps='Public local-step method; captures failed-node identities before the next transition.'))


if __name__=='__main__':
    from graph_tiger.defenses import run_defense_method
    from graph_tiger.measures import run_measure
    tasks = {f.__name__: f for f in [preventive_defense,sir_ensemble,complete_local,process_snapshots]}
    for task in ([tasks[name] for name in sys.argv[1:]] if len(sys.argv)>1 else tasks.values()):
        task()
