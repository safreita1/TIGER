"""Generate comparable drawings using TIGER's layout/bundling and SIR state rules.

Install graph-tiger[visualization], then run this file. No experiment results are overwritten.
ForceAtlas2 uses the same settings as TIGER with an explicit initialization seed.
"""
import os,sys,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for folder in ['experiment-deps','visualization-deps']:
    if (ROOT/folder).exists():sys.path.insert(0,str(ROOT/folder))
if (ROOT/'public-0.6.0').exists():sys.path.insert(0,str(ROOT/'public-0.6.0'))
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
if os.name=='nt' and os.environ.get('TIGER_VIS_DLLS'):
    dll_handle=os.add_dll_directory(os.environ['TIGER_VIS_DLLS'])
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from importlib.metadata import version
from fa2 import ForceAtlas2
from graph_tiger.simulations import Simulation
from graph_tiger.diffusion import Diffusion

OUT=ROOT if ROOT.name=='guide-results' else ROOT/'site'/'guide-results'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path','font.size':12})
G=nx.karate_club_graph()
nx.set_edge_attributes(G,1.,'weight')
original_edges=set(G.edges)
saved=[]

def draw(slug,title,pos,edge_pos=None,states=None):
    fig,ax=plt.subplots(figsize=(7,6.4))
    if edge_pos is None:nx.draw_networkx_edges(G,pos,ax=ax,edge_color='#a7b9bd',width=1.25,alpha=.85)
    else:ax.plot(edge_pos.x,edge_pos.y,color='#a7b9bd',linewidth=1.25,alpha=.85,zorder=1)
    colors=states if states is not None else {n:'#cf6636' if n==0 else '#5765b0' if n==33 else '#157a79' for n in G}
    nx.draw_networkx_nodes(G,pos,ax=ax,node_size=210,node_color=[colors[n] for n in G],edgecolors='white',linewidths=.7)
    nx.draw_networkx_labels(G,pos,ax=ax,font_size=7,font_color='white')
    ax.set_title(title,fontsize=16,pad=18)
    if states is not None:
        labels=[('#157a79','Susceptible'),('#cf6636','Infected'),('#5765b0','Recovered')]
        ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in labels],loc='lower center',bbox_to_anchor=(.5,-.09),ncol=3,frameon=False,fontsize=10)
    xy=np.array(list(pos.values()))
    lo,hi=xy.min(axis=0),xy.max(axis=0)
    margin=(hi-lo)*.12
    ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0])
    ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout(pad=1.8)
    for ext in ['svg','png','pdf']:fig.savefig(OUT/(slug+'.'+ext),dpi=130)
    plt.close(fig);saved.append(slug)

# DOC-BEGIN layouts
spectral,_=Simulation(G.copy(),runs=1,steps=0).get_graph_coordinates()
layouts={'spectral':spectral}
for iterations in [20,200]:
    force=ForceAtlas2(outboundAttractionDistribution=True,edgeWeightInfluence=0,
                      scalingRatio=6.0,verbose=False,seed=17)
    layouts['force-'+str(iterations)]=force.forceatlas2_networkx_layout(G,pos=None,iterations=iterations)
provided=nx.circular_layout(G)
H=G.copy()
nx.set_node_attributes(H,provided,'pos')
layouts['provided'],_=Simulation(H,runs=1,steps=0,node_style='force_atlas').get_graph_coordinates()
# DOC-END
assert all(np.allclose(layouts['provided'][n],provided[n]) for n in G)
for key,label in [('spectral','Default: spectral layout'),('provided','Supplied positions: circular'),('force-20','ForceAtlas2: 20 iterations'),('force-200','ForceAtlas2: 200 iterations')]:
    assert len(layouts[key])==34 and np.isfinite(np.array(list(layouts[key].values()))).all()
    draw('vis-layout-'+key,label,layouts[key])

# DOC-BEGIN bundling
H=G.copy()
nx.set_node_attributes(H,layouts['force-200'],'pos')
straight,_=Simulation(H,runs=1,steps=0,edge_style=None).get_graph_coordinates()
bundled_pos,bundled_edges=Simulation(H,runs=1,steps=0,edge_style='bundled').get_graph_coordinates()
# DOC-END
assert all(np.allclose(straight[n],bundled_pos[n]) for n in G)
assert not bundled_edges.empty
draw('vis-edges-straight','Straight edges',straight)
draw('vis-edges-bundled','Bundled edges',bundled_pos,bundled_edges)
print('Rendered four layouts and matched straight/bundled edges.',flush=True)

os.chdir(OUT)
sim=Diffusion(G,model='SIR',b=.2,d=.08,c=.15,runs=1,steps=30,seed=17)
sim.run_single_sim()
selected=[]
sim.plot_network=lambda step:selected.append(step)
sim.plot_graph_transition(sim.sim_info)
animation_steps=list(range(0,31,10))
counts=[]
for t in sorted(set(selected+animation_steps)):
    # Rerun to each horizon so every image reads the state at that step.
    snapshot=Diffusion(G.copy(),model='SIR',b=.2,d=.08,c=.15,runs=1,steps=t,seed=17)
    snapshot.run_single_sim()
    state=snapshot.sim_info[t]
    colors={n:'#5765b0' if n in state['protected'] else '#cf6636' if infected else '#157a79' for n,infected in zip(G,state['status'])}
    counts.append(dict(step=t,susceptible=34-state['failed']-state['recovered'],infected=state['failed'],recovered=state['recovered']))
    assert sum(counts[-1][k] for k in ['susceptible','infected','recovered'])==34
    for key,color in [('susceptible','#157a79'),('infected','#cf6636'),('recovered','#5765b0')]:
        assert list(colors.values()).count(color)==counts[-1][key]
    assert (state['failed'],state['recovered'])==(sim.sim_info[t]['failed'],sim.sim_info[t]['recovered'])
    draw('vis-state-'+str(t),'SIR state: step '+str(t),layouts['force-200'],states=colors)
with (OUT/'vis-states.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
frames=[Image.open(OUT/('vis-state-'+str(t)+'.png')).convert('RGB') for t in animation_steps]
frames[0].save(OUT/'vis-playback.gif',save_all=True,append_images=frames[1:],duration=800,loop=0)
for frame in frames:frame.close()
assert set(G.edges)==original_edges
manifest=dict(graph='unweighted karate',nodes=34,edges=78,fa_initialization_seed=17,
    layout_coordinates={k:{str(n):list(map(float,p)) for n,p in pos.items()} for k,pos in layouts.items()},
    simulation=dict(model='SIR',b=.2,d=.08,c=.15,steps=30,seed=17),
    snapshot_method='Each horizon is rerun from seed 17 so every image reads its own current state.',
    transition_steps=selected,animation_steps=animation_steps,
    rendering='TIGER spectral/provided positions and hammer_bundle; ForceAtlas2 matches TIGER parameters with seed=17 added explicitly. Uniform documentation colors and labels.',
    playback='Pillow GIF from the recorded states; not the Windows-disabled TIGER MP4 exporter. Slowed to 800 ms per frame for inspection.',
    dependencies={p:version(p) for p in ['graph-tiger','networkx','numpy','matplotlib','fa2','datashader','dask','numba','pillow']},
    files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('vis-*') if f.suffix in ['svg','png','pdf','gif','csv']})
(OUT/'vis-gallery.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Rendered snapshot states '+str(selected)+' and portable playback frames '+str(animation_steps)+'.',flush=True)
