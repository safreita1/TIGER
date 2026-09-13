"""Benchmark non-spectral TIGER CPU and GPU application workloads."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time

import networkx as nx
import numpy as np

from graph_tiger.attacks import run_attack_method
from graph_tiger.cascading import Cascading
from graph_tiger.defenses import run_defense_method
from graph_tiger.diffusion import Diffusion
from graph_tiger.influence import Influence
from graph_tiger.utils import gpu_status, networkx_gpu_status


def build_graph(family, nodes, seed, mean_degree):
    """Build one sparse benchmark graph with a reproducible topology."""

    if family == 'BA':
        return nx.barabasi_albert_graph(nodes, max(1, mean_degree // 2), seed=seed)
    if family == 'WS':
        degree = min(mean_degree, nodes - 1)
        degree -= degree % 2
        return nx.watts_strogatz_graph(nodes, degree, 0.1, seed=seed)
    probability = min(1.0, mean_degree / max(1, nodes - 1))
    return nx.fast_gnp_random_graph(nodes, probability, seed=seed)


def digest(value):
    """Return a compact stable representation of one benchmark result."""

    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def exact_parity(cpu, gpu):
    return cpu == gpu


def cascade_parity(cpu, gpu):
    if cpu['trajectory'] != gpu['trajectory'] or cpu['failed'] != gpu['failed']:
        return False
    return np.allclose(cpu['load'], gpu['load'], rtol=1e-5, atol=1e-4)


def workloads(graph, seed, steps, samples):
    """Return representative calls from each accelerable TIGER area."""

    nodes = tuple(graph.nodes)
    node_order = {node: index for index, node in enumerate(nodes)}
    initial = set(nodes[:max(1, min(20, len(nodes) // 100))])
    first = nodes[0]
    second = nodes[1] if len(nodes) > 1 else nodes[0]

    def attack_pagerank(backend):
        return run_attack_method(graph, 'pr_node', 10, backend=backend)

    def attack_eigenvector(backend):
        return run_attack_method(graph, 'eig_node', 10, backend=backend)

    def attack_betweenness(backend):
        return run_attack_method(
            graph, 'ib_node', 10, approx=min(samples, len(graph)),
            seed=seed, backend=backend
        )

    def defense_pagerank(backend):
        return list(run_defense_method(
            graph, 'pr_node', 10, seed=seed, backend=backend
        ))

    def motter_lai(backend):
        simulation = Cascading(
            graph, model='motter_lai', runs=1, steps=min(3, steps),
            r=0.3, k_a=min(3, len(graph)), attack='id_node',
            c=min(samples, len(graph)), seed=seed, backend=backend
        )
        trajectory = simulation.run_single_sim()
        return {
            'trajectory': trajectory,
            'failed': sorted(simulation.failed, key=node_order.__getitem__),
            'load': [simulation.load[node] for node in nodes]
        }

    def diffusion(model, backend):
        simulation = Diffusion(
            graph, model=model, runs=1, steps=steps, b=0.08, d=0.03,
            c=0.02, seed=seed, backend=backend
        )
        trajectory = simulation.run_single_sim()
        return {
            'trajectory': trajectory,
            'infected': sorted(simulation.infected, key=node_order.__getitem__),
            'recovered': sorted(simulation.vaccinated, key=node_order.__getitem__)
        }

    def influence(model, backend):
        params = {}
        if model == 'independent_cascade':
            params.update({'seeds': {first}, 'probability': 0.2})
        elif model == 'linear_threshold':
            params.update({
                'seeds': initial, 'threshold': 2, 'weight': None
            })
        else:
            params.update({
                'message_seeds': {'a': {first}, 'b': {second}},
                'probability': 0.2, 'tracked_state': 'a',
                'tie_break': 'priority', 'priority': ['a', 'b']
            })
        simulation = Influence(
            graph, model=model, runs=1, steps=steps, seed=seed,
            backend=backend, **params
        )
        trajectory = simulation.run_single_sim()
        return {
            'trajectory': trajectory,
            'state': [simulation.state[node] for node in nodes]
        }

    return [
        ('attacks', 'pagerank_node', attack_pagerank, exact_parity),
        ('attacks', 'eigenvector_node', attack_eigenvector, exact_parity),
        ('attacks', 'betweenness_node', attack_betweenness, exact_parity),
        ('defenses', 'pagerank_node', defense_pagerank, exact_parity),
        ('cascading', 'motter_lai', motter_lai, cascade_parity),
        ('epidemics', 'sis', lambda backend: diffusion('SIS', backend), exact_parity),
        ('epidemics', 'sir', lambda backend: diffusion('SIR', backend), exact_parity),
        ('influence', 'independent_cascade',
         lambda backend: influence('independent_cascade', backend), exact_parity),
        ('influence', 'linear_threshold',
         lambda backend: influence('linear_threshold', backend), exact_parity),
        ('influence', 'competitive_cascade',
         lambda backend: influence('competitive_cascade', backend), exact_parity)
    ]


def time_workload(call, warmups, repeats):
    """Time alternating CPU/GPU calls after paired warm-ups."""

    for _ in range(warmups):
        call('cpu')
        call('gpu')

    observations = {'cpu': [], 'gpu': []}
    results = {}
    for repeat in range(repeats):
        order = ('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')
        for backend in order:
            start = time.perf_counter()
            results[backend] = call(backend)
            observations[backend].append(time.perf_counter() - start)
    return observations, results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', nargs='+', type=int, default=[1000, 5000, 20000])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 17, 41])
    parser.add_argument('--families', nargs='+', choices=['BA', 'WS', 'ER'],
                        default=['BA', 'WS', 'ER'])
    parser.add_argument('--mean-degree', type=int, default=8)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='gpu-application-benchmark-results.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    if not gpu_status()['available']:
        raise RuntimeError(gpu_status()['reason'])
    if not networkx_gpu_status()['available']:
        raise RuntimeError(networkx_gpu_status()['reason'])
    if hasattr(nx.config, 'warnings_to_ignore'):
        nx.config.warnings_to_ignore.add('cache')

    rows = []
    for family in args.families:
        for requested_nodes in args.nodes:
            for seed in args.seeds:
                graph = build_graph(family, requested_nodes, seed, args.mean_degree)
                for area, workload, call, compare in workloads(
                        graph, seed, args.steps, args.samples):
                    observations, results = time_workload(
                        call, args.warmups, args.repeats
                    )
                    cpu_median = statistics.median(observations['cpu'])
                    gpu_median = statistics.median(observations['gpu'])
                    parity = compare(results['cpu'], results['gpu'])
                    row = {
                        'area': area,
                        'workload': workload,
                        'family': family,
                        'requested_nodes': requested_nodes,
                        'nodes': len(graph),
                        'edges': graph.number_of_edges(),
                        'seed': seed,
                        'steps': args.steps,
                        'samples': min(args.samples, len(graph)),
                        'cpu_median_seconds': cpu_median,
                        'gpu_median_seconds': gpu_median,
                        'speedup': cpu_median / gpu_median,
                        'cpu_observations': json.dumps(observations['cpu']),
                        'gpu_observations': json.dumps(observations['gpu']),
                        'cpu_digest': digest(results['cpu']),
                        'gpu_digest': digest(results['gpu']),
                        'parity': parity
                    }
                    rows.append(row)
                    print(
                        area, workload, family, requested_nodes, seed,
                        '{:.2f}x'.format(row['speedup']), 'parity={}'.format(parity),
                        flush=True
                    )

    with open(args.output, 'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        'arguments': vars(args),
        'python': platform.python_version(),
        'networkx': nx.__version__,
        'gpu': gpu_status(),
        'networkx_gpu': networkx_gpu_status(),
        'rows': len(rows),
        'parity_passed': sum(row['parity'] for row in rows),
        'parity_failed': sum(not row['parity'] for row in rows)
    }
    metadata_path = os.path.splitext(args.output)[0] + '.metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as output:
        json.dump(metadata, output, indent=2)

    if metadata['parity_failed']:
        raise SystemExit('{} parity checks failed'.format(metadata['parity_failed']))


if __name__ == '__main__':
    main()
