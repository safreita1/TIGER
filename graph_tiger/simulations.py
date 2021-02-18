import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d
from datashader.bundling import hammer_bundle
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from graph_tiger.utils import get_sparse_graph, curved_edges


class Simulation:
    """
    The parent class for all simulation classes i.e., attack, defense, cascading failure and diffusion models.
    Provides a shared set of functions, largely for network visualization and plotting of results
    """
    def __init__(self, graph, runs, steps, **kwargs):
        """

        :param graph: undirected NetworkX graph
        :param runs: number of times to run the simulation
        :param steps: number of time steps to run each simulation
        :param kwargs: optional parameters to change visualization settings
        """
        self.graph = graph

        self.prm = {
            'runs': runs,
            'steps': steps,

            'seed': 1,
            'max_val': 1,

            'gif_animation': False,
            'gif_snaps': False,
            'plot_transition': False,

            'edge_style': None,
            'node_style': None,
            'fa_iter': 200
        }

        self.sim_info = defaultdict()
        self.sparse_graph = get_sparse_graph(self.graph)

        if self.prm['seed'] is not None:
            random.seed(self.prm['seed'])
            np.random.seed(self.prm['seed'])

    def child_class(self):
        """
        Gets the child class name
        :return: string
        """
        return self.__class__.__name__

    def get_graph_coordinates(self):
        """
        Gets the graph coordinates, which can be:
        (1) set in the graph itself with the 'pos' tag on the vertices,
        (2) positioned according to the force atlas2 algorithm,
        (3) positioned using a spectral layout.

        Then lays out the edges, can be curved, bundled, or straight

        :return: Tuple containing node and edge positions
        """
        edge_pos = None

        node_pos = {idx: v['pos'] for idx, (k, v) in enumerate(dict(self.graph.nodes).items()) if 'pos' in v}   # check graph for coords
        node_pos = node_pos if len(node_pos) == len(self.graph) else None

        # node positions
        if self.prm['node_style'] == 'force_atlas' and node_pos is None:
            force = ForceAtlas2(outboundAttractionDistribution=True, edgeWeightInfluence=0, scalingRatio=6.0, verbose=False)
            node_pos = force.forceatlas2_networkx_layout(self.graph, pos=None, iterations=self.prm['fa_iter'])

        elif node_pos is None:
            node_pos = nx.spectral_layout(self.graph)

        # edge positions
        if self.prm['edge_style'] == 'bundled':
            pos = pd.DataFrame.from_dict(node_pos, orient='index', columns=['x', 'y']).rename_axis('name').reset_index()
            edge_pos = hammer_bundle(pos, nx.to_pandas_edgelist(self.graph))

        return node_pos, edge_pos

    def plot_results(self, results):
        """
        Plots the compiled simulation results

        :param results: a list of floats representing each simulation output
        """
        results_norm = [r / len(self.graph) for r in results]

        plt.figure(figsize=(6.4, 4.8))

        if self.child_class() == 'Diffusion':
            plt.plot(results_norm, label="Effective strength: {}".format(self.get_effective_strength()))

            if self.prm['model'] == 'SIS':
                plt.ylabel('Infected Nodes')
            else:
                plt.ylabel('Recovered Nodes')

            plt.legend()
            plt.yscale('log')
            plt.ylim(0.001, 1)

        elif self.child_class() == 'Cascading' or self.child_class() == 'Attack' or self.child_class() == 'Defense':
            plt.plot(results_norm)
            plt.ylabel(self.prm['robust_measure'])
            plt.ylim(0, 1)

        plt.xlabel('Steps')
        plt.title(self.child_class())
        plt.savefig(os.path.join(self.save_dir, self.get_plot_title(self.prm['steps']) + '_results.pdf'))
        plt.show()

        plt.clf()

    def get_plot_title(self, step):
        """
        Gets the title for each plot

        :param step: the current simulation iteration
        :return: title string
        """
        if self.child_class() == 'Diffusion':
            title = '{}_epidemic:step={},diffusion={},method={},k={}'.format(self.prm['model'], step, self.prm['diffusion'], self.prm['method'], self.prm['k'])

        elif self.child_class() == 'Cascading':
            title = 'Cascading:step={},l={},r={},k_a={},attack={},k_d={},defense={}'.format(step, self.prm['l'], self.prm['r'], self.prm['k_a'],
                                                                                            self.prm['attack'], self.prm['k_d'], self.prm['defense'])
        elif self.child_class() == 'Attack':
            title = 'Attack:step={},attack={},k_d={},defense={}'.format(step, self.prm['attack'], self.prm['k_d'], self.prm['defense'])

        elif self.child_class() == 'Defense':
            title = 'Defense:step={},attack={},k_a={},defense={}'.format(step, self.prm['attack'], self.prm['k_a'], self.prm['defense'])

        else:
            title = ''

        return title

    def plot_graph_transition(self, sim_info):
        """
        Helper function to decide which snapshots to take for network visualization

        :param sim_info: the information stored at each step in the simulation
        """
        history = [info['failed'] for step, info in sim_info.items()]

        start = history[0]
        end = history[-1]
        middle = start - int((start - end) / 2)
        mid_step, _ = min(enumerate(history), key=lambda x: abs(x[1] - middle))

        steps_to_plot = [0, 1, 2, mid_step, self.prm['steps'] - 1]

        for step in steps_to_plot:
            self.plot_network(step=step)

    def get_visual_settings(self, step):
        """
        Sets the visual settings for the network visualization

        :param step: current iteration of the simulation
        :return: four lists, each containing a number corresponding to the size or color of each node in the graph + cmap representing color scheme
        """
        if self.child_class() == 'Cascading':
            nc, ns = [], []
            ew = 1
            ec = 'gray'

            for idx, load in enumerate(self.sim_info[step]['status']):
                cval = interp1d([0, self.prm['max_val']], [20, 1500])
                ns.append(float(cval(self.capacity[idx])))

                if load <= self.capacity[idx]:
                    cval = interp1d([0, self.capacity[idx]], [0, 0.8])
                    nc.append(float(cval(load)))
                else:
                    nc.append(1)

            cmap = plt.get_cmap('jet', 5)

        elif self.child_class() == 'Diffusion':
            nc, ns = [], []
            ew = 0.1
            ec = '#1F76B4'

            for idx, s in enumerate(self.sim_info[step]['status']):
                if idx in self.sim_info[0]['protected']:
                    nc.append(0.5)
                    ns.append(200)
                elif s == 1:
                    nc.append(s)
                    ns.append(40)
                else:
                    nc.append(s)
                    ns.append(20)
            cmap = LinearSegmentedColormap.from_list('mycmap', ['#67CAFF', '#17255A', '#FF5964'])

        elif self.child_class() == 'Attack' or self.child_class() == 'Defense':
            ew = 5
            ec = 'gray'
            nc = self.sim_info[step]['status']
            ns = [120 if status == 1 else 40 for status in self.sim_info[step]['status']]
            cmap = plt.get_cmap('gist_rainbow_r')

        nc = np.array(nc)
        ns = np.array(ns)

        return nc, ns, ec, ew, cmap

    def draw_graph(self, step):
        """
        Draws the graph

        :param step: current iteration of the simulation
        :return: matplotlib.collections.PathCollection PathCollection` of the nodes.
        """
        nc, ns, ec, ew, cmap = self.get_visual_settings(step)

        if self.prm['edge_style'] == 'curved':
            plt.gca().add_collection(LineCollection(curved_edges(self.graph, self.node_pos), linewidth=ew, color=ec))

        elif self.prm['edge_style'] == 'bundled':
            plt.plot(self.edge_pos.x, self.edge_pos.y, zorder=1, linewidth=ew, color=ec)

        else:
            nx.draw_networkx_edges(self.graph, pos=self.node_pos, width=ew, edge_color=ec)

        nodes = nx.draw_networkx_nodes(self.graph, pos=self.node_pos, cmap=cmap, vmin=0, vmax=self.prm['max_val'], node_size=ns, node_color=nc)

        return nodes

    def plot_network(self, step):
        """
        Plots the compiled simulation results

        :param step: current iteration of the simulation
        """
        plt.figure(figsize=(20, 20))

        self.draw_graph(step)

        plt.axis('image')
        title = self.get_plot_title(step)
        plt.savefig(os.path.join(self.save_dir, title + '.pdf'))
        plt.show()

        plt.clf()

    def create_simulation_gif(self):
        """
        Draws and saves the network simulation to an MP4 file
        """
        fig = plt.figure(figsize=(20, 20))
        nodes = self.draw_graph(step=0)

        snap_dir = os.path.join(self.save_dir, 'gif_snaps/')
        os.makedirs(snap_dir, exist_ok=True)

        def update(step):
            nc, ns, _, _, _ = self.get_visual_settings(step)

            nodes.set_array(nc)
            nodes.set_sizes(ns)

            if self.prm['gif_snaps']:
                plt.savefig(snap_dir + 'step_{}.pdf'.format(step))

            return nodes,

        if self.child_class() == 'Diffusion':
            frames = iter(list(range(0, self.prm['steps'], 10)))
            interval = 20
            fps = 5
        elif self.child_class() == 'Cascading':
            frames = self.prm['steps']
            interval = 20
            fps = 3
        else:
            frames = self.prm['steps']
            interval = 20
            fps = 1

        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=not self.prm['gif_snaps'], repeat=False)

        title = self.get_plot_title(self.prm['steps'])
        gif_path = os.path.join(self.save_dir, title + '.mp4')
        anim.save(gif_path, fps=fps, extra_args=['-vcodec', 'libx264'])

        plt.clf()

    def run_simulation(self):
        """
        Averages the simulation over the number of 'runs'.

        :return: a list containing the average value at each 'step' of the simulation.
        """
        print('Running simulation {} times'.format(self.prm['runs']))

        sim_results = list(range(self.prm['runs']))
        for r in range(self.prm['runs']):
            sim_results[r] = self.run_single_sim()

            if self.prm['plot_transition'] and r == 0:
                self.plot_graph_transition(self.sim_info)

            if self.prm['gif_animation'] and r == 0:
                self.create_simulation_gif()

            self.reset_simulation()

        avg_results = []
        for t in range(self.prm['steps']):
            avg_results.append(np.mean([sim_results[r][t] for r in range(self.prm['runs'])]))

        return avg_results

    def reset_simulation(self):
        """
        Implemented by child class
        """
        pass

    def run_single_sim(self):
        """
        Implemented by child class
        """
        pass

    def get_effective_strength(self):
        """
        Implemented by child class
        """
        pass
