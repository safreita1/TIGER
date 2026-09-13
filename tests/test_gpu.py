import networkx as nx
import numpy as np
import pytest
from types import SimpleNamespace

from graph_tiger.measures import run_measure
from graph_tiger import utils as tiger_utils
from graph_tiger.utils import (
    get_adjacency_spectrum,
    get_laplacian_spectrum,
    gpu_available,
    gpu_status,
    networkx_backend_kwargs,
    select_backend,
    select_networkx_backend,
    system_gpu_status
)
from experiments.robustness import gpu_benchmarks


def test_gpu_status_has_stable_fields():
    status = gpu_status()
    assert set(status) == {
        'available', 'device_count', 'device_name', 'free_memory',
        'total_memory', 'reason'
    }
    assert status['available'] == gpu_available()


def test_cpu_dispatch_omits_backend_for_predispatch_networkx(monkeypatch):
    graph = nx.path_graph(3)
    tiger_utils.networkx_gpu_status.cache_clear()
    with monkeypatch.context() as context:
        context.delattr(nx, 'config')
        assert networkx_backend_kwargs(graph, backend='cpu') == {}
    tiger_utils.networkx_gpu_status.cache_clear()


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


def test_explicit_gpu_rejects_cpu_only_connectivity_measures():
    graph = nx.cycle_graph(4)
    for measure in ['node_connectivity', 'edge_connectivity']:
        with pytest.raises(ValueError):
            run_measure(graph, measure, backend='gpu')


def test_measured_auto_policy_selects_only_clear_gpu_wins(monkeypatch):
    status = {
        'available': True,
        'device_count': 1,
        'device_name': 'Test GPU',
        'free_memory': 32 * 1024 ** 3,
        'total_memory': 32 * 1024 ** 3,
        'reason': 'CUDA device is ready'
    }
    monkeypatch.setattr(tiger_utils, 'gpu_status', lambda: status)

    assert select_backend(
        nx.path_graph(250), backend='auto', k=1,
        operation='average_distance'
    )['selected'] == 'gpu'
    assert select_backend(
        nx.path_graph(20000), backend='auto', k=1,
        operation='spectral_radius'
    )['selected'] == 'gpu'
    assert select_backend(
        nx.path_graph(20000), backend='auto', k=1,
        operation='sis'
    )['selected'] == 'cpu'
    assert select_backend(
        nx.path_graph(20000), backend='auto', k=30,
        operation='spectral_scaling'
    )['selected'] == 'cpu'


def test_measured_networkx_policy_and_override(monkeypatch):
    status = {
        'available': True,
        'device_count': 1,
        'device_name': 'Test GPU',
        'free_memory': 32 * 1024 ** 3,
        'total_memory': 32 * 1024 ** 3,
        'reason': 'nx-cugraph and a CUDA device are ready',
        'backend': 'cugraph'
    }
    monkeypatch.setattr(tiger_utils, 'networkx_gpu_status', lambda: status)

    assert select_networkx_backend(
        nx.path_graph(4999), backend='auto', operation='pagerank'
    )['selected'] == 'cpu'
    assert select_networkx_backend(
        nx.path_graph(5000), backend='auto', operation='pagerank'
    )['selected'] == 'gpu'
    assert select_networkx_backend(
        nx.path_graph(20), backend='auto', min_gpu_nodes=0,
        operation='pagerank'
    )['selected'] == 'gpu'


def test_system_gpu_status_uses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(tiger_utils.shutil, 'which', lambda name: 'nvidia-smi')
    result = SimpleNamespace(
        stdout='NVIDIA Test GPU, 600.1, 24576\n', stderr='', returncode=0
    )
    monkeypatch.setattr(tiger_utils.subprocess, 'run', lambda *args, **kwargs: result)

    status = system_gpu_status()

    assert status['available']
    assert status['devices'][0]['name'] == 'NVIDIA Test GPU'
    assert status['devices'][0]['memory_mib'] == 24576


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
    assert np.all(np.diff(laplacian) >= 0)
    assert laplacian[0] >= -1e-8


def test_gpu_shortest_path_statistics_match_closed_forms():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')

    graph = nx.relabel_nodes(
        nx.path_graph(4), {index: 'node-{}'.format(index) for index in range(4)}
    )
    statistics = tiger_utils.get_shortest_path_statistics(
        graph, backend='gpu', block_size=2
    )

    assert statistics['distance_sum'] == 20
    assert statistics['inverse_distance_sum'] == pytest.approx(26 / 3)
    assert statistics['diameter'] == 3
    assert statistics['reachable_pairs'] == 12


def test_gpu_largest_component_size_preserves_arbitrary_labels():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')

    graph = nx.Graph([('left-a', 'left-b'), ('right-a', 'right-b')])
    graph.add_node('right-c')

    assert tiger_utils.get_largest_component_size(
        graph, backend='gpu'
    ) == 2


def test_largest_component_auto_stays_on_cpu(monkeypatch):
    graph = nx.disjoint_union(nx.path_graph(4), nx.path_graph(2))

    def unexpected_gpu_selection(*args, **kwargs):
        raise AssertionError('auto should not inspect the GPU for this measure')

    monkeypatch.setattr(tiger_utils, 'select_backend', unexpected_gpu_selection)
    assert tiger_utils.get_largest_component_size(
        graph, backend='auto', min_gpu_nodes=0
    ) == 4


def test_gpu_shortest_path_measures_preserve_disconnected_contracts():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')

    graph = nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))
    assert run_measure(graph, 'diameter', backend='gpu') is None
    assert run_measure(graph, 'average_distance', backend='gpu') is None
    assert run_measure(
        graph, 'average_inverse_distance', backend='gpu'
    ) == run_measure(graph, 'average_inverse_distance', backend='cpu')


def test_gpu_remaining_measures_handle_empty_graph():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')

    graph = nx.Graph()
    assert run_measure(graph, 'diameter', backend='gpu') is None
    assert run_measure(graph, 'average_distance', backend='gpu') is None
    assert run_measure(graph, 'average_inverse_distance', backend='gpu') == 0
    assert run_measure(graph, 'largest_connected_component', backend='gpu') == 0


def test_laplacian_spectrum_restores_a_missing_zero_mode(monkeypatch):
    monkeypatch.setattr(
        tiger_utils,
        'eigsh',
        lambda *args, **kwargs: np.array([0.1, 0.2])
    )

    spectrum = get_laplacian_spectrum(
        nx.path_graph(120), k=2, eigvals_only=True, backend='cpu'
    )

    assert spectrum == pytest.approx([0.0, 0.1])


def test_largest_magnitude_parity_allows_a_tied_sign_boundary(monkeypatch):
    spectra = {
        'cpu': np.array([-6.062558, 5.951263, 9.151679]),
        'gpu': np.array([-6.062558, -5.948622, 9.151679])
    }

    def fake_spectrum(graph, k, which, eigvals_only, backend):
        return spectra[backend]

    monkeypatch.setattr(
        gpu_benchmarks, 'get_adjacency_spectrum', fake_spectrum
    )
    absolute, relative = gpu_benchmarks.raw_spectrum_error(
        nx.path_graph(3), 'spectral_scaling', 3
    )

    assert absolute == pytest.approx(0.002641)
    assert relative < 1e-3


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
