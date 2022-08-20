Introduction
============

``TIGER`` is a Python **T**\ oolbox for  evaluat\ **I**\ ing  **G**\ raph vuln\ **E**\ rability and **R**\ obustness that allows users to (1) measure graph vulnerability and robustness, (2) attack networks using a variety of offensive techniques, (3) defend a network using a variety of heuristic and optimization based defense techniques. ``TIGER`` is specifically designed to help users:

- Quantify network vulnerability and robustness.
- Simulate a variety of network attacks, cascading failures and spread of dissemination of entities.
- Augment a network's structure to resist attacks and recover from failure.
- Regulate the dissemination of entities on a network (e.g., viruses, propaganda).

Background
**********

First mentioned as early as the 1970's, network robustness has a rich and diverse history spanning numerous fields of engineering and science. This diversity of research has generated a variety of unique perspectives, providing fresh insight into challenging problems while equipping researchers with fundamental knowledge for their investigations. While the fields of study are diverse, they are linked by a common definition of **robustness**, which is defined as a measure of a network's ability to continue functioning when part of the network is naturally damaged or targeted for attack.

The study of network robustness is critical to the understanding of complex interconnected systems. For example, consider an example of a power grid network that is susceptible to both natural *failures* and targeted *attacks*. A natural failure occurs when a *single* power substation fails due to erosion of parts or natural disasters. However, when one substation fails, additional load is routed to alternative substations, potentially causing a series of *cascading failures*. Not all failures originate from natural causes, some come from *targeted* attacks, such as enemy states hacking into the grid to sabotage key equipment to maximally damage the operations of the electrical grid. A natural counterpart to network robustness is **vulnerability**, defined as *measure of a network's susceptibility to the dissemination of entities across the network*, such as how quickly a virus spreads across a computer network. 

Motivation
**********
Unfortunately, the nature of cross-disciplinary research also comes with significant challenges. Oftentimes important discoveries made in one field are not quickly disseminated, leading to missed innovation opportunities. We believe a unified and easy-to-use software framework is key to standardizing the study of network robustness, helping accelerate reproducible research and dissemination of ideas.


Installation
************
TIGER was designed for the Linux environment using Python 3, however, we don't foresee any issues running it on Mac or Windows. To quickly get started you can install TIGER using:


Or you can git clone TIGER from https://github.com/safreita1/TIGER and create an Anaconda environment using the environment.yml file.

To use the built-in graph dataset helper functions, do the following:

create a folder called "dataset" in the main directory.
call the function get_graph_urls() to get a list of urls containing the datasets used in all our experiments.
run wget "url goes here" and place the downloaded graph inside the dataset folder.


Examples
********

We provide detailed tutorials on how to user TIGER in the Tutorials section. Below we look at a few simple examples to get you started.


Example 1: Measuring graph robustness
-------------------------------------

How to measure graph robustness using the spectral methods: :func:`measures.spectral_radius` and :func:`measures.effective_resistance`.

.. code-block:: python
   :name: robustness-example-1 

   from measures import run_measure
   from graphs import graph_loader

   graph = graph_loader(graph_type='BA', n=1000, seed=1)

   spectral_radius = run_measure(graph, measure='spectral_radius')
   print("Spectral radius:", spectral_radius)

   effective_resistance = run_measure(graph, measure='effective_resistance')
   print("Effective resistance:", effective_resistance)


Example 2: Measuring approximate graph robustness
-------------------------------------------------

How to measure *approximate* graph robustness using spectral method: :func:`measures.effective_resistance`.

.. code-block:: python
   :name: robustness-example-2

   from measures import run_measure
   from graphs import graph_loader

   graph = graph_loader(graph_type='BA', n=1000, seed=1)

   effective_resistance = run_measure(graph, measure='effective_resistance', k=30)
   print("Effective resistance (k=30):", effective_resistance)



Example 3: Cascading Failure Simulation
---------------------------------------

In this example, we run a cascading failure simulation on a Barabasi Albert (BA) graph. In the network, node size represents load capacity (i.e., larger size -> higher capacity), and color indicates the load of each node on a gradient scale from blue (low load) to red (high load); dark red indicates node failure (overloaded). Below, we show a TIGER cascading failure simulation on a BA graph when 30 nodes in the network randomly fail (untargeted attack). 

.. code-block:: python
   :name: cascading-failure-example

   from cascading import Cascading
   from graphs import graph_loader

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


.. _fig-coordsys-rect:

.. figure:: ../../images/Cascading:step=0,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg
   :width: 100 %
   :align: center
   
   Time step 0: shows the network under normal operating conditions.


.. _fig-coordsys-rect:

.. figure:: ../../images/Cascading:step=6,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg
   :width: 100 %
   :align: center
   
   Step 5: we observe a series of failures across the network.


.. _fig-coordsys-rect:

.. figure:: ../../images/Cascading:step=99,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None.jpg
   :width: 100 %
   :align: center
   
   Step 99: most of the network has collapsed.


.. _fig-coordsys-rect:

.. figure:: ../../images/Cascading:step=100,l=0.8,r=0.2,k_a=30,attack=rb_node,k_d=0,defense=None_results.jpg
   :width: 100 %
   :align: center
   
   Graph connectivity over time (measured by graph's largest connected component) during attack.


Example 4: SIS Model Network Vaccination
----------------------------------------

In this example, we run a computer virus simulation (SIS infection model) on a BA graph. The network starts off highly infected, and the goal is to vaccinate critical nodes to reduce disease resurgence. Using the Netshield techniqe, we select 5 nodes to vaccinate to maximally reduce the infection.

.. code-block:: python
   :name: sis-example

   from diffusion import Diffusion
   from graphs import graph_loader

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



.. _fig-coordsys-rect:

.. figure:: ../../images/SIS_epidemic:step=0,diffusion=min,method=ns_node,k=5.jpg
   :width: 100 %
   :align: center
   
   Step 0: A highly infected network with 4 nodes "vaccinated" according to Netshield defense.


.. _fig-coordsys-rect:

.. figure:: ../../images/SIS_epidemic:step=80,diffusion=min,method=ns_node,k=5.jpg
   :width: 100 %
   :align: center
   
   Step 80: The computer virus begins to remit. 


.. _fig-coordsys-rect:

.. figure:: ../../images/SIS_epidemic:step=4999,diffusion=min,method=ns_node,k=5.jpg
   :width: 100 %
   :align: center
   
   Step 4999: The virus is nearly contained.


.. _fig-coordsys-rect:

.. figure:: ../../images/SIS_epidemic:step=5000,diffusion=min,method=ns_node,k=5_results.jpg
   :width: 100 %
   :align: center
   
   A plot of the number of infected nodes in the network at each time stamp.


.. note:: Figures will auto-populate in the "plots" folder.





