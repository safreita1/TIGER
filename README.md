![Version](https://img.shields.io/pypi/v/graph-tiger?color=dark)
[![Documentation Status](https://readthedocs.org/projects/graph-tiger/badge/?version=latest)](https://graph-tiger.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2006.05648-<COLOR>.svg)](https://arxiv.org/pdf/2006.05648.pdf)


![TIGER Library](images/TIGER.jpg)

**TIGER** is a Python toolbox to conduct graph vulnerability and robustness research. For additional information, please take a look at the   **[Documentation](https://graph-tiger.readthedocs.io/)** and relevant **[Paper](https://arxiv.org/pdf/2006.05648.pdf)**.

 
**TIGER** contains state-of-the-art methods to help users conduct graph vulnerability and robustness analysis on graph structured data.
It contains 22 graph **robustness measures** with both original and fast approximate versions; 
17 **attack strategies**; 15 heuristic and optimization based **defense techniques**; and 4 **simulation tools**.
Specifically, TIGER is specifically designed to help users:

1. **Quantify** network *vulnerability* and *robustness*, 
2. **Simulate** a variety of network attacks, cascading failures and spread of dissemination of entities
3. **Augment** a network's structure to resist *attacks* and recover from *failure* 
4. **Regulate** the dissemination of entities on a network (e.g., viruses, propaganda). 

--------------------------------------------------------------------------------

### Setup
To quickly get started, install TIGER using pip

```sh
$ pip install graph-tiger
``` 

Alternatively, you can clone [TIGER](https://github.com/safreita1/TIGER.git) and create a new Anaconda environment 
using the [YAML](environment.yml) file.

--------------------------------------------------------------------------------

### Tutorials
To help you get started we provide 5 in-depth tutorials in the **[Documentation](https://graph-tiger.readthedocs.io/)**;
each tutorial covers an aspect of TIGER's core functionality: 

Tutorial 1. [Measuring Graph Vulnerability and Robustness](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-1.html)

Tutorial 2. [Attacking a Network](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-2.html)

Tutorial 3. [Defending A Network](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-3.html)

Tutorial 4. [Simulating Cascading Failures on Networks](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-4.html)

Tutorial 5. [Simulating Entity Dissemination on Networks](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-5.html)

--------------------------------------------------------------------------------

### Citing

If you find *TIGER* useful in your research, please consider citing the following paper:

```bibtex
@article{freitas2020evaluating,
    title={Evaluating Graph Vulnerability and Robustness using TIGER},
    author={Freitas, Scott and Chau, Duen Horng},
    journal={arXiv preprint arXiv:2006.05648},
    year={2020}
}
```

--------------------------------------------------------------------------------

### Techniques Included

**Vulnerability and Robustness Measures**

* **[Vertex Connectivity](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.node_connectivity)** from Ye *et al.*: [Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection](https://github.com/benedekrozemberczki/DANMF/blob/master/18DANMF.pdf) (CIKM 2018)
* **[Edge Connectivity](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.edge_connectivity)**
* **[Diameter](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.diameter)**
* **[Average Distance](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_distance)**
* **[Average Inverse Distance (Efficiency)](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_inverse_distance)**
* **[Average Vertex Betweenness](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_vertex_betweenness)**
* **[Approximate Average Vertex Betweenness](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_vertex_betweenness)**
* **[Average Edge Betweenness](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_edge_betweenness)**
* **[Approximate Average Edge Betweenness](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.avg_edge_betweenness)**
* **[Average Clustering Coefficient](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.average_clustering_coefficient)**
* **[Largest Connected Component](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.largest_connected_component)**
* **[Spectral Radius](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.spectral_radius)**
* **[Spectral Gap](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.spectral_gap)**
* **[Natural Connectivity](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.natural_connectivity)**
* **[Approximate Natural Connectivity](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.natural_connectivity)**
* **[Spectral Scaling](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.spectral_scaling)**
* **[Generalized Robustness Index](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.generalized_robustness_index)**
* **[Algebraic Connectivity](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.algebraic_connectivity)**
* **[Number of Spanning Trees](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.num_spanning_trees)**
* **[Approximate Number of Spanning Trees](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.num_spanning_trees)**
* **[Effective Resistance](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.effective_resistance)**
* **[Approximate Effective Resistance](https://graph-tiger.readthedocs.io/en/latest/measures.html#graph_tiger.measures.effective_resistance)**


**Attack Strategies**
* **[Remove Node: Netshield](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_ns)**
* **[Remove Node: PageRank](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_pr)**
* **[Remove Node: Eigenvector Centrality](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_eig)**
* **[Remove Node: Initial Degree (ID) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_id)**
* **[Remove Node: Recalculated Degree (RD) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_rd)**
* **[Remove Node: Initial Betweenness (IB) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_ib)**
* **[Remove Node: Recalculated Betweenness (RB) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_rb)**
* **[Remove Node: Random Selection](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_node_rnd)**
* **[Remove Edge: Netshield Line](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_line_ns)**
* **[Remove Edge: PageRank Line](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_line_pr)**
* **[Remove Edge: Eigenvector Centrality Line](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_line_eig)**
* **[Remove Edge: Degree Line](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_line_deg)**
* **[Remove Edge: Initial Betweenness (IB) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_ib)**
* **[Remove Edge: Recalculated Betweenness (RB) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_rb)**
* **[Remove Edge: Initial Degree (ID) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_id)**
* **[Remove Edge: Recalculated Degree (RD) Removal](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_rd)**
* **[Remove Edge: Random Selection](https://graph-tiger.readthedocs.io/en/latest/attacks.html#graph_tiger.attacks.get_edge_rnd)**


**Defense Strategies**
* **[Vaccinate Node: Netshield](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_ns)**
* **[Vaccinate Node: PageRank](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_pr)**
* **[Vaccinate Node: Eigenvector Centrality](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_eig)**
* **[Vaccinate Node: Initial Degree (ID) Removal](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_id)**
* **[Vaccinate Node: Recalculated Degree (RD) Removal](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_rd)**
* **[Vaccinate Node: Initial Betweenness (IB) Removal](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_ib)**
* **[Vaccinate Node: Recalculated Betweenness (RB) Removal](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_rb)**
* **[Vaccinate Node: Random Selection](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.get_node_rnd)**
* **[Add Edge: PageRank](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.add_edge_pr)**
* **[Add Edge: Eigenvector Centrality](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.add_edge_eig)**
* **[Add Edge: Degree Centrality](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.add_edge_degree)**
* **[Add Edge: Random Selection](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.add_edge_rnd)**
* **[Add Edge: Preferential](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.add_edge_pref)**
* **[Rewire Edge: Random](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.rewire_edge_rnd)**
* **[Rewire Edge: Random Neighbor](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.rewire_edge_rnd_neighbor)**
* **[Rewire Edge: Preferential](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.rewire_edge_pref)**
* **[Rewire Edge: Preferential Random](https://graph-tiger.readthedocs.io/en/latest/defenses.html#graph_tiger.defenses.rewire_edge_pref_rnd)**


**Simulation Frameworks**
* **[Susceptible-Infected-Susceptible (SIS) Model](https://graph-tiger.readthedocs.io/en/latest/diffusion.html#graph_tiger.diffusion.Diffusion)**
* **[Susceptible-Infected-Recovered (SIR) Model](https://graph-tiger.readthedocs.io/en/latest/diffusion.html#graph_tiger.diffusion.Diffusion)**
* **[Cascading Failure Model](https://graph-tiger.readthedocs.io/en/latest/cascading.html#graph_tiger.cascading.Cascading)**

--------------------------------------------------------------------------------

### Quick Examples

#### EX 1. Calculate graph robustness (e.g., spectral radius, effective resistance)
    from graph_tiger.measures import run_measure
    from graph_tiger.graphs import graph_loader
    
    graph = graph_loader(graph_type='BA', n=1000, seed=1)
    
    spectral_radius = run_measure(graph, measure='spectral_radius')
    print("Spectral radius:", spectral_radius)
    
    effective_resistance = run_measure(graph, measure='effective_resistance')
    print("Effective resistance:", effective_resistance)
        
    

### EX 2. Run a cascading failure simulation on a Barabasi Albert graph
    from graph_tiger.cascading import Cascading
    from graph_tiger.graphs import graph_loader
    
    graph = graph_loader('BA', n=400, seed=1)
    
    params = {
        'runs': 1,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.2,
        'c': int(0.1 * len(graph)),
    
        'k_a': 30,
        'attack': 'rb_node',
        'attack_approx': int(0.1 * len(graph)),
    
        'k_d': 0,
        'defense': None,
    
        'robust_measure': 'largest_connected_component',
    
        'plot_transition': True,  # False turns off key simulation image "snapshots"
        'gif_animation': False,  # True creaets a video of the simulation (MP4 file)
        'gif_snaps': False,  # True saves each frame of the simulation as an image
    
        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 2000,
    }
    
    cascading = Cascading(graph, **params)
    results = cascading.run_simulation()
    
    cascading.plot_results(results)
    
Step 0: Network pre-attack | Step 6: Beginning of cascading failure | Step 99: Collapse of network
:-------------------------:|:-------------------------:|:-------------------------:
![](images/Cascading:step=0,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg)  |  ![](images/Cascading:step=6,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg)  |  ![](images/Cascading:step=99,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg)
    
    
[comment]: ![](images/Cascading:step=100,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None_results.jpg)
    
### EX 3. Run an SIS virus simulation on a Barabasi Albert graph
    from graph_tiger.diffusion import Diffusion
    from graph_tiger.graphs import graph_loader
    
    graph = graph_loader('BA', n=400, seed=1)
    
    
    sis_params = {
        'model': 'SIS',
        'b': 0.001,
        'd': 0.01,
        'c': 1,
    
        'runs': 1,
        'steps': 5000,
        'seed': 1,
    
        'diffusion': 'min',
        'method': 'ns_node',
        'k': 5,
    
        'plot_transition': True,
        'gif_animation': False,
    
        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 2000
    }
    
    diffusion = Diffusion(graph, **sis_params)
    results = diffusion.run_simulation()
    
    diffusion.plot_results(results)
    
    
Step 0: Virus infected network |Step 80: Partially infected network | Step 4999: Virus contained
:-------------------------:|:-------------------------:|:-------------------------:
![](images/SIS_epidemic:step=0,diffusion=min,method=ns_node,k=5.jpg)  |![](images/SIS_epidemic:step=80,diffusion=min,method=ns_node,k=5.jpg)  |  ![](images/SIS_epidemic:step=4999,diffusion=min,method=ns_node,k=5.jpg)

[comment]: ![](images/SIS_epidemic:step=5000,diffusion=min,method=ns_node,k=5_results.jpg)

--------------------------------------------------------------------------------
### License

[MIT License](LICENSE)