"""Generate the local load-allocation examples for graph-tiger 0.6.0."""
import os, sys, csv, json, hashlib, tempfile
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parent
for folder in ['experiment-deps','public-0.6.0']:
    if (ROOT/folder).exists():sys.path.insert(0,str(ROOT/folder))
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from graph_tiger.cascading import Cascading

OUT=ROOT if ROOT.name=='guide-results' else ROOT/'site'/'guide-results'
OUT.mkdir(parents=True,exist_ok=True)

def examples():
    # DOC-BEGIN single
    G = nx.Graph([("F", "a"), ("F", "b"), ("F", "c")])
    policies = ["degree", "greedy", "proportional", "max_flow"]
    allocations = {}
    for demand in [12, 18]:
        for policy in policies:
            sim = Cascading(G.copy(), model="local_load_sharing", allocation=policy,
                            initial_load={"F": demand, "a": 0, "b": 0, "c": 0},
                            capacities={"F": demand, "a": 3, "b": 5, "c": 7},
                            initial_failures=["F"], beta=0, runs=1, steps=1)
            sim.run_single_sim()
            allocations[demand, policy] = [sim.last_transfers["F", n] for n in "abc"]
    # DOC-END

    # DOC-BEGIN shared
    G = nx.Graph([("A", "U"), ("A", "V"), ("B", "U")])
    policies = ["degree", "greedy", "proportional", "max_flow"]
    trials = {}
    for policy in policies:
        sim = Cascading(G.copy(), model="local_load_sharing", allocation=policy,
                        initial_load={"A": 5, "B": 7, "U": 0, "V": 0},
                        capacities={"A": 5, "B": 7, "U": 7, "V": 5},
                        initial_failures=["A", "B"], beta=0, runs=1, steps=1)
        sim.run_single_sim()
        trials[policy] = {"transfers": sim.last_transfers.copy(),
                          "failed": sim.failed.copy(), "load": sim.load.copy(),
                          "lost_load": sim.sim_info[1]["lost_load"]}
    # DOC-END
    return allocations,trials

def main():
    from inspect import signature
    if 'allocation' not in signature(Cascading).parameters:
        raise RuntimeError('graph-tiger 0.6.0 or later is required for allocation policies.')
    previous=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='tiger-allocation-figures-') as tmp:
        os.chdir(tmp)
        try:allocations,trials=examples()
        finally:os.chdir(previous)
    rows=[]
    labels={'degree':'Equal sharing','greedy':'Greedy','proportional':'Proportional','max_flow':'Maximum flow'}
    colors=['#157a79','#cf6636','#5765b0','#657783']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'path'})
    def save(fig,slug):
        for ext in ['svg','pdf','png']:fig.savefig(OUT/(slug+'.'+ext),dpi=150,bbox_inches='tight')
        plt.close(fig)
    for demand in [12,18]:
        fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
        x=np.arange(3)
        for k,(policy,label) in enumerate(labels.items()):
            amounts=allocations[demand,policy]
            assert np.isclose(sum(amounts),demand)
            ax.bar(x+(k-1.5)*.19,amounts,width=.18,color=colors[k],label=label)
            for recipient,amount in zip('abc',amounts):
                rows.append(dict(example='single',demand=demand,policy=policy,source='F',recipient=recipient,transfer=amount))
        for i,headroom in enumerate([3,5,7]):
            ax.hlines(headroom,i-.43,i+.43,colors='#1c3038',linestyles='--',linewidth=2)
            ax.text(i,headroom+.2,f'Spare: {headroom}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=['Recipient a','Recipient b','Recipient c'],ylim=(0,10),ylabel='Work received in the first round',title=f'One failed source: {demand} units to transfer')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.11),ncol=2,frameon=False)
        save(fig,'local-allocation-single-'+str(demand))
    G=nx.DiGraph([('A','U'),('A','V'),('B','U')])
    pos={'A':(0,1),'B':(1,1),'U':(1,0),'V':(0,0)}
    for policy,trial in trials.items():
        fig,ax=plt.subplots(figsize=(7,5.5),layout='constrained')
        node_colors=['#657783' if n in ['A','B'] else '#cf6636' if n in trial['failed'] else '#157a79' for n in G]
        nx.draw_networkx_nodes(G,pos,ax=ax,node_color=node_colors,node_size=1500,edgecolors='white',linewidths=2)
        nx.draw_networkx_labels(G,pos,ax=ax,font_color='white',font_size=18)
        nx.draw_networkx_edges(G,pos,ax=ax,node_size=1500,arrows=True,arrowsize=22,width=2,edge_color='#82969b')
        nx.draw_networkx_edge_labels(G,pos,ax=ax,edge_labels={e:f'{v:.2f}'.rstrip('0').rstrip('.') for e,v in trial['transfers'].items()},font_size=14,rotate=False,bbox=dict(facecolor='white',edgecolor='none',pad=4))
        ax.text(0,1.21,'5 units displaced',ha='center');ax.text(1,1.21,'7 units displaced',ha='center')
        for n,spare in [('U',7),('V',5)]:
            ax.text(pos[n][0],-.23,f"Received {trial['load'][n]:.2f} / spare {spare}",ha='center',fontsize=12)
        outcome='No new failures' if trial['failed']=={'A','B'} else 'U overloads'
        ax.set_title(labels[policy]+': '+outcome,pad=22,fontsize=16)
        ax.set_xlim(-.38,1.38);ax.set_ylim(-.46,1.34);ax.axis('off')
        ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in [('#657783','Initially failed'),('#cf6636','New overload'),('#157a79','Functioning')]],loc='lower center',bbox_to_anchor=(.5,-.07),ncol=3,frameon=False,fontsize=10)
        for (source,recipient),amount in trial['transfers'].items():
            rows.append(dict(example='shared',demand=12,policy=policy,source=source,recipient=recipient,transfer=amount))
        assert trial['lost_load']==0 and np.isclose(sum(trial['load'].values()),12)
        save(fig,'local-allocation-shared-'+policy)
    with (OUT/'local-allocation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata={'package_version':version('graph-tiger'),'policies':list(labels),
        'rounds':1,'beta':0,'single':{'demands':[12,18],'recipient_spare_capacity':[3,5,7]},
        'shared':{'initial_failures':['A','B'],'loads':{'A':5,'B':7,'U':0,'V':0},'recipient_spare_capacity':{'U':7,'V':5},'edges':list(G.edges)},
        'conventions':'Pre-round headroom; synchronous overload checks; Edmonds-Karp max flow with graph node/edge insertion order; equal overflow; no deliberate shedding.',
        'shared_new_failures':{p:sorted(t['failed']-{'A','B'}) for p,t in trials.items()},
        'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('local-allocation-*') if f.suffix in ['svg','pdf','png']}}
    (OUT/'local-allocation.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print('Six allocation figures and 36 transfers generated; full workload conserved in every example.')

if __name__=='__main__':main()
