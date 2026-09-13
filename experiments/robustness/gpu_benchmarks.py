"""Benchmark CPU and GPU implementations of TIGER spectral measures.

The benchmark records end-to-end public-API time, including conversion and
transfer costs. It also compares numerical results before reporting speedup.
"""

import argparse
import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_tiger.measures import run_measure
from graph_tiger.utils import (
    get_adjacency_spectrum,
    get_laplacian_spectrum,
    gpu_status,
    select_backend
)


MEASURES = [
    'spectral_radius',
    'spectral_gap',
    'natural_connectivity',
    'spectral_scaling',
    'generalized_robustness_index',
    'algebraic_connectivity',
    'number_spanning_trees',
    'effective_resistance'
]


def measure_k(measure, requested_k):
    """Return the eigenpair count actually used by a measure."""

    return {
        'spectral_radius': 1,
        'spectral_gap': 2,
        'algebraic_connectivity': 2
    }.get(measure, requested_k)


def build_graph(family, nodes, mean_degree, seed):
    """Create one connected benchmark graph."""

    if family == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(
            nodes, max(1, mean_degree // 2), seed=seed
        )
    elif family == 'watts_strogatz':
        degree = min(nodes - 1, mean_degree)
        degree -= degree % 2
        graph = nx.connected_watts_strogatz_graph(
            nodes, max(2, degree), 0.1, tries=200, seed=seed
        )
    elif family == 'erdos_renyi':
        probability = min(1.0, mean_degree / max(1, nodes - 1))
        graph = nx.fast_gnp_random_graph(nodes, probability, seed=seed)
        if len(graph) and not nx.is_connected(graph):
            largest = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest).copy()
    else:
        raise ValueError('unknown graph family {}'.format(family))

    return nx.convert_node_labels_to_integers(graph)


def synchronize_gpu():
    """Wait for queued CUDA work so timings include completed computation."""

    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def timed_measure(graph, measure, k, backend):
    """Measure one end-to-end call through TIGER's public measure API."""

    gc.collect()
    if backend == 'gpu':
        synchronize_gpu()
    start = time.perf_counter()
    value = run_measure(graph, measure, k=k, backend=backend)
    if backend == 'gpu':
        synchronize_gpu()
    return value, time.perf_counter() - start


def percentiles(values):
    values = np.asarray(values, dtype=float)
    return {
        'median': float(np.median(values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90))
    }


def result_error(cpu_value, gpu_value, atol, rtol):
    """Return absolute error, relative error, and a parity decision."""

    if cpu_value is None or gpu_value is None:
        passed = cpu_value is None and gpu_value is None
        return np.nan, np.nan, passed

    cpu_value = float(cpu_value)
    gpu_value = float(gpu_value)
    if np.isinf(cpu_value) or np.isinf(gpu_value):
        passed = np.isinf(cpu_value) and np.isinf(gpu_value)
        return 0.0 if passed else np.inf, 0.0 if passed else np.inf, passed

    absolute = abs(cpu_value - gpu_value)
    relative = absolute / max(abs(cpu_value), np.finfo(float).eps)
    return absolute, relative, bool(np.isclose(
        cpu_value, gpu_value, atol=atol, rtol=rtol
    ))


def raw_spectrum_error(graph, measure, k):
    """Compare unrounded eigenvalues used by the CPU and GPU paths."""

    if measure in {
            'spectral_radius', 'spectral_gap', 'natural_connectivity',
            'spectral_scaling', 'generalized_robustness_index'}:
        which = 'LM' if measure in {
            'spectral_scaling', 'generalized_robustness_index'
        } else 'LA'
        count = {
            'spectral_radius': 1,
            'spectral_gap': 2
        }.get(measure, k)
        cpu = get_adjacency_spectrum(
            graph, k=count, which=which, eigvals_only=True, backend='cpu'
        )
        gpu = get_adjacency_spectrum(
            graph, k=count, which=which, eigvals_only=True, backend='gpu'
        )
    else:
        count = 2 if measure == 'algebraic_connectivity' else k
        cpu = get_laplacian_spectrum(
            graph, k=count, eigvals_only=True, backend='cpu'
        )
        gpu = get_laplacian_spectrum(
            graph, k=count, eigvals_only=True, backend='gpu'
        )

    cpu = np.sort(np.asarray(cpu, dtype=float))
    gpu = np.sort(np.asarray(gpu, dtype=float))
    absolute = float(np.max(np.abs(cpu - gpu))) if len(cpu) else 0.0
    scale = max(float(np.max(np.abs(cpu))) if len(cpu) else 0.0,
                np.finfo(float).eps)
    return absolute, absolute / scale


def benchmark_case(graph, measure, k, repeats, warmups, atol, rtol):
    """Benchmark one graph and measure after independent backend warm-ups."""

    cpu_cold_value, cpu_cold = timed_measure(graph, measure, k, 'cpu')
    gpu_cold_value, gpu_cold = timed_measure(graph, measure, k, 'gpu')

    for _ in range(warmups):
        timed_measure(graph, measure, k, 'cpu')
        timed_measure(graph, measure, k, 'gpu')

    cpu_times = []
    gpu_times = []
    cpu_value = cpu_cold_value
    gpu_value = gpu_cold_value
    for repeat in range(repeats):
        order = ('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')
        for backend in order:
            value, duration = timed_measure(graph, measure, k, backend)
            if backend == 'cpu':
                cpu_value = value
                cpu_times.append(duration)
            else:
                gpu_value = value
                gpu_times.append(duration)

    cpu_stats = percentiles(cpu_times)
    gpu_stats = percentiles(gpu_times)
    absolute, relative, passed = result_error(
        cpu_value, gpu_value, atol, rtol
    )
    eigen_absolute, eigen_relative = raw_spectrum_error(graph, measure, k)
    passed = passed and (
        eigen_absolute <= atol or eigen_relative <= rtol
    )
    return {
        'cpu_value': cpu_value,
        'gpu_value': gpu_value,
        'absolute_error': absolute,
        'relative_error': relative,
        'parity_passed': passed,
        'eigenvalue_absolute_error': eigen_absolute,
        'eigenvalue_relative_error': eigen_relative,
        'cpu_cold_seconds': cpu_cold,
        'gpu_cold_seconds': gpu_cold,
        'cpu_median_seconds': cpu_stats['median'],
        'cpu_p10_seconds': cpu_stats['p10'],
        'cpu_p90_seconds': cpu_stats['p90'],
        'gpu_median_seconds': gpu_stats['median'],
        'gpu_p10_seconds': gpu_stats['p10'],
        'gpu_p90_seconds': gpu_stats['p90'],
        'speedup': cpu_stats['median'] / gpu_stats['median'],
        'cpu_times': json.dumps(cpu_times),
        'gpu_times': json.dumps(gpu_times)
    }


def system_metadata():
    """Record enough environment information to interpret the results."""

    status = gpu_status()
    metadata = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'gpu': status
    }
    try:
        import scipy
        metadata['scipy'] = scipy.__version__
    except ImportError:
        pass
    try:
        import cupy
        metadata['cupy'] = cupy.__version__
    except ImportError:
        pass
    metadata['networkx'] = nx.__version__
    metadata['numpy'] = np.__version__
    return metadata


def run_benchmarks(args):
    status = gpu_status()
    if not status['available']:
        raise RuntimeError('GPU benchmark unavailable: {}'.format(status['reason']))

    rows = []
    for family in args.families:
        for nodes in args.nodes:
            for seed in range(args.graph_seeds):
                graph = build_graph(family, nodes, args.mean_degree, seed)
                for measure in args.measures:
                    k = measure_k(measure, args.k)
                    if args.exact and measure not in {
                            'spectral_radius', 'spectral_gap',
                            'algebraic_connectivity'}:
                        k = np.inf
                    selection = select_backend(
                        graph, backend='auto', k=k,
                        min_gpu_nodes=args.min_gpu_nodes
                    )
                    result = benchmark_case(
                        graph, measure, k, args.repeats, args.warmups,
                        args.atol, args.rtol
                    )
                    result.update({
                        'family': family,
                        'requested_nodes': nodes,
                        'nodes': len(graph),
                        'edges': graph.number_of_edges(),
                        'density': nx.density(graph),
                        'seed': seed,
                        'measure': measure,
                        'k': k,
                        'auto_backend': selection['selected'],
                        'auto_reason': selection['reason']
                    })
                    rows.append(result)
                    print(
                        '{} n={} seed={} {}: {:.2f}x, parity={}'.format(
                            family, len(graph), seed, measure,
                            result['speedup'], result['parity_passed']
                        )
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output.with_suffix('.metadata.json')
    metadata = system_metadata()
    metadata['arguments'] = vars(args)
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    failures = [row for row in rows if not row['parity_passed']]
    print('Wrote {} cases to {}'.format(len(rows), output))
    print('Numerical parity failures: {}'.format(len(failures)))
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare TIGER spectral measures on CPU and GPU'
    )
    parser.add_argument(
        '--nodes', nargs='+', type=int, default=[1000, 5000, 20000]
    )
    parser.add_argument(
        '--families', nargs='+',
        choices=['barabasi_albert', 'watts_strogatz', 'erdos_renyi'],
        default=['barabasi_albert', 'watts_strogatz', 'erdos_renyi']
    )
    parser.add_argument('--measures', nargs='+', choices=MEASURES, default=MEASURES)
    parser.add_argument('--mean-degree', type=int, default=8)
    parser.add_argument('--graph-seeds', type=int, default=3)
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument(
        '--exact', action='store_true',
        help='request full spectra; use only with suitably small node sizes'
    )
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--min-gpu-nodes', type=int, default=1000)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument(
        '--output', default='gpu-benchmark-results.csv'
    )
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(run_benchmarks(parse_args()))
