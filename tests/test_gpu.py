import networkx as nx
import numpy as np
import pytest

from graph_tiger.measures import run_measure
from graph_tiger.utils import (
    get_adjacency_spectrum,
    get_laplacian_spectrum,
    gpu_available,
    gpu_status,
    select_backend
)


def test_gpu_status_has_stable_fields():
    status = gpu_status()
    assert set(status) == {
        'available', 'device_count', 'device_name', 'free_memory',
        'total_memory', 'reason'
    }
    assert status['available'] == gpu_available()


def test_backend_selection_keeps_small_auto_workloads_on_cpu():
    graph = nx.path_graph(20)
    selection = select_backend(
        graph, backend='auto', k=2, min_gpu_nodes=100
    )
    assert selection['selected'] == 'cpu'
    assert not selection['suitable']


def test_backend_selection_rejects_invalid_name():
    with pytest.raises(ValueError):
        select_backend(nx.path_graph(4), backend='accelerator', k=2)


def test_cpu_partial_spectra_match_expected_values():
    graph = nx.path_graph(120)
    adjacency = np.sort(get_adjacency_spectrum(
        graph, k=2, eigvals_only=True, backend='cpu'
    ))
    laplacian = get_laplacian_spectrum(
        graph, k=2, eigvals_only=True, backend='cpu'
    )
    assert adjacency.shape == (2,)
    assert laplacian.shape == (2,)
    assert laplacian[0] == pytest.approx(0, abs=1e-8)
    assert laplacian[1] > 0


def test_legacy_use_gpu_false_matches_cpu_backend():
    graph = nx.barabasi_albert_graph(120, 3, seed=17)
    legacy = run_measure(graph, 'spectral_radius', use_gpu=False)
    explicit = run_measure(graph, 'spectral_radius', backend='cpu')
    assert legacy == explicit


@pytest.mark.skipif(not gpu_available(), reason='CUDA device unavailable')
@pytest.mark.parametrize('measure,k', [
    ('spectral_radius', 1),
    ('spectral_gap', 2),
    ('natural_connectivity', 30),
    ('spectral_scaling', 30),
    ('generalized_robustness_index', 30),
    ('algebraic_connectivity', 2),
    ('number_spanning_trees', 30),
    ('effective_resistance', 30)
])
def test_cpu_gpu_measure_parity(measure, k):
    graph = nx.barabasi_albert_graph(250, 4, seed=17)
    cpu = run_measure(graph, measure, k=k, backend='cpu')
    gpu = run_measure(graph, measure, k=k, backend='gpu')

    if cpu is None or gpu is None:
        assert cpu is None and gpu is None
    elif np.isinf(cpu) or np.isinf(gpu):
        assert np.isinf(cpu) and np.isinf(gpu)
    else:
        assert gpu == pytest.approx(cpu, rel=1e-3, abs=1e-2)
