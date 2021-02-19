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

* **[Vertex Connectivity](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.community_detection.overlapping.danmf.DANMF)** from Ye *et al.*: [Deep Autoencoder-like Nonnegative Matrix Factorization for Community Detection](https://github.com/benedekrozemberczki/DANMF/blob/master/18DANMF.pdf) (CIKM 2018)
* **[Edge Connectivity]()**
* **[Diameter]()**
* **[Average Distance]()**
* **[Efficiency]()**
* **[Average Vertex Betweenness]()**
* **[Approximate Average Vertex Betweenness]()**
* **[Approximate Average Edge Betweenness]()**
* **[Average Clustering Coefficient]()**
* **[Largest Connected Component]()**
* **[Spectral Radius]()**
* **[Spectral Gap]()**
* **[Natural Connectivity]()**
* **[Approximate Natural Connectivity]()**
* **[Spectral Scaling]()**
* **[Generalized Robustness Index]()**
* **[Algebraic Connectivity]()**
* **[Number of Spanning Trees]()**
* **[Effective Resistance]()**
* **[Approximate Effective Resistance]()**



**Defense Strategies**
* **[Vaccinate Node: Netshield]()**
* **[Vaccinate Node: PageRank]()**
* **[Vaccinate Node: Eigenvector Centrality]()**
* **[Vaccinate Node: Initial Degree (ID) Removal]()**
* **[Vaccinate Node: Recalculated Degree (RD) Removal]()**
* **[Vaccinate Node: Initial Betweenness (IB) Removal]()**
* **[Vaccinate Node: Recalculated Betweenness (RB) Removal]()**
* **[Vaccinate Node: Random Selection]()**

* **[Add Edge: PageRank]()**
* **[Add Edge: Eigenvector Centrality]()**
* **[Add Edge: Degree Centrality]()**
* **[Add Edge: Random Selection]()**
* **[Add Edge: Preferential]()**

* **[Rewire Edge: Random]()**
* **[Rewire Edge: Random Neighbor]()**
* **[Rewire Edge: Preferential]()**
* **[Rewire Edge: Preferential Random]()**

**Attack Strategies**
* **[Remove Node: Netshield]()**
* **[Remove Node: PageRank]()**
* **[Remove Node: Eigenvector Centrality]()**
* **[Remove Node: Initial Degree (ID) Removal]()**
* **[Remove Node: Recalculated Degree (RD) Removal]()**
* **[Remove Node: Initial Betweenness (IB) Removal]()**
* **[Remove Node: Recalculated Betweenness (RB) Removal]()**
* **[Remove Node: Random Selection]()**

* **[Remove Edge: Netshield Line]()**
* **[Remove Edge: PageRank Line]()**
* **[Remove Edge: Eigenvector Centrality Line]()**
* **[Remove Edge: Degree Line]()**
* **[Remove Edge: Initial Betweenness (IB) Removal]()**
* **[Remove Edge: Recalculated Betweenness (RB) Removal]()**
* **[Remove Edge: Initial Degree (ID) Removal]()**
* **[Remove Edge: Recalculated Degree (RD) Removal]()**


**Simulation Frameworks**
* **[Susceptible-Infected-Susceptible (SIS)]()**
* **[Susceptible-Infected-Recovered (SIR)]()**
* **[]()**
SIS
SIR
Cascading failure


--------------------------------------------------------------------------------

### Detailed Tutorials and Examples
To help you get started we provide 5 in-depth tutorials in the **[Documentation](https://graph-tiger.readthedocs.io/)**;
each tutorial covers an aspect of TIGER's core functionality: 

Tutorial 1. [Measuring Graph Vulnerability and Robustness](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-1.html)

Tutorial 2. [Attacking a Network](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-2.html)

Tutorial 3. [Defending A Network](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-3.html)

Tutorial 4. [Simulating Cascading Failures on Networks](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-4.html)

Tutorial 5. [Simulating Entity Dissemination on Networks](https://graph-tiger.readthedocs.io/en/latest/tutorials/tutorial-5.html)


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
**License**

[MIT License](LICENSE)