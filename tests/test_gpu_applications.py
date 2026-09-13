import networkx as nx
import pytest

from graph_tiger.attacks import run_attack_method
from graph_tiger.attacks import Attack
from graph_tiger.cascading import Cascading
from graph_tiger.defenses import Defense, run_defense_method
from graph_tiger.diffusion import Diffusion
from graph_tiger.influence import Influence
from graph_tiger.measures import run_measure
from graph_tiger.utils import gpu_available, networkx_gpu_status


@pytest.mark.parametrize('model', ['SIS', 'SIR'])
def test_diffusion_cpu_gpu_parity(model):
    if not gpu_available():
        pytest.skip('CUDA device unavailable')
    graph = nx.barabasi_albert_graph(250, 4, seed=17)
    params = {
        'model': model, 'runs': 1, 'steps': 12, 'b': 0.2,
        'd': 0.1, 'c': 0.05, 'seed': 17
    }
    cpu = Diffusion(graph, backend='cpu', **params)
    gpu = Diffusion(graph, backend='gpu', **params)

    assert gpu.run_single_sim() == cpu.run_single_sim()
    assert gpu.infected == cpu.infected
    assert gpu.vaccinated == cpu.vaccinated


def test_progressive_influence_cpu_gpu_parity():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')
    graph = nx.barabasi_albert_graph(250, 4, seed=17)
    params = {
        'model': 'independent_cascade', 'seeds': {0},
        'probability': 0.3, 'runs': 1, 'steps': 8, 'seed': 17
    }
    cpu = Influence(graph, backend='cpu', **params)
    gpu = Influence(graph, backend='gpu', **params)

    assert gpu.run_single_sim() == cpu.run_single_sim()
    assert gpu.state == cpu.state


def test_linear_threshold_cpu_gpu_parity():
    if not gpu_available():
        pytest.skip('CUDA device unavailable')
    graph = nx.barabasi_albert_graph(250, 4, seed=17)
    params = {
        'model': 'linear_threshold', 'seeds': {0, 1},
        'threshold': 2, 'weight': None, 'runs': 1, 'steps': 8,
        'seed': 17
    }
    cpu = Influence(graph, backend='cpu', **params)
    gpu = Influence(graph, backend='gpu', **params)

    assert gpu.run_single_sim() == cpu.run_single_sim()
    assert gpu.state == cpu.state
    assert gpu.influence == pytest.approx(cpu.influence)


@pytest.mark.parametrize('tie_break', ['priority', 'random'])
def test_competitive_influence_cpu_gpu_parity(tie_break):
    if not gpu_available():
        pytest.skip('CUDA device unavailable')
    graph = nx.barabasi_albert_graph(250, 4, seed=17)
    params = {
        'model': 'competitive_cascade',
        'message_seeds': {'claim': {0}, 'correction': {1}},
        'probability': 0.3, 'tracked_state': 'correction',
        'tie_break': tie_break, 'priority': ['correction', 'claim'],
        'runs': 1, 'steps': 8, 'seed': 17
    }
    cpu = Influence(graph, backend='cpu', **params)
    gpu = Influence(graph, backend='gpu', **params)

    assert gpu.run_single_sim() == cpu.run_single_sim()
    assert gpu.state == cpu.state


def test_networkx_centrality_attack_and_defense_parity():
    if not networkx_gpu_status()['available']:
        pytest.skip('nx-cugraph unavailable')
    graph = nx.barabasi_albert_graph(500, 4, seed=17)

    cpu_attack = run_attack_method(
        graph, 'ib_node', 8, approx=50, seed=7, backend='cpu'
    )
    gpu_attack = run_attack_method(
        graph, 'ib_node', 8, approx=50, seed=7, backend='gpu'
    )
    cpu_defense = run_defense_method(
        graph, 'add_edge_pr', 8, backend='cpu'
    )
    gpu_defense = run_defense_method(
        graph, 'add_edge_pr', 8, backend='gpu'
    )

    assert gpu_attack == cpu_attack
    assert gpu_defense == cpu_defense


def test_networkx_approximate_edge_betweenness_parity():
    if not networkx_gpu_status()['available']:
        pytest.skip('nx-cugraph unavailable')
    graph = nx.barabasi_albert_graph(500, 4, seed=17)

    cpu = run_measure(
        graph, 'average_edge_betweenness', k=50,
        seed=7, backend='cpu'
    )
    gpu = run_measure(
        graph, 'average_edge_betweenness', k=50,
        seed=7, backend='gpu'
    )

    assert gpu == pytest.approx(cpu, rel=1e-5, abs=1e-2)


def test_remaining_measure_cpu_gpu_parity():
    if not networkx_gpu_status()['available']:
        pytest.skip('nx-cugraph unavailable')

    graph = nx.barabasi_albert_graph(120, 3, seed=17)
    for measure in [
            'diameter', 'average_distance', 'average_inverse_distance',
            'average_clustering_coefficient', 'largest_connected_component']:
        cpu = run_measure(graph, measure, backend='cpu')
        gpu = run_measure(graph, measure, backend='gpu')
        assert gpu == pytest.approx(cpu)


def test_remaining_measures_flow_through_attack_and_defense():
    if not networkx_gpu_status()['available']:
        pytest.skip('nx-cugraph unavailable')

    graph = nx.barabasi_albert_graph(120, 3, seed=17)
    measures = [
        'diameter', 'average_distance', 'average_inverse_distance',
        'average_clustering_coefficient', 'largest_connected_component'
    ]
    for measure in measures:
        for simulation_type, parameters in [
                (Attack, {'attack': 'id_node', 'steps': 2}),
                (Defense, {
                    'attack': 'id_node', 'k_a': 2,
                    'defense': 'id_node', 'k_d': 1, 'steps': 2
                })]:
            cpu = simulation_type(
                graph, runs=1, robust_measure=measure,
                backend='cpu', seed=17, **parameters
            )
            gpu = simulation_type(
                graph, runs=1, robust_measure=measure,
                backend='gpu', seed=17, **parameters
            )
            assert gpu.run_single_sim() == cpu.run_single_sim()


def test_motter_lai_cpu_gpu_state_parity():
    if not networkx_gpu_status()['available']:
        pytest.skip('nx-cugraph unavailable')
    graph = nx.barabasi_albert_graph(150, 3, seed=17)
    params = {
        'model': 'motter_lai', 'runs': 1, 'steps': 3,
        'r': 0.2, 'k_a': 2, 'attack': 'id_node',
        'c': 150, 'seed': 17
    }
    cpu = Cascading(graph, backend='cpu', **params)
    gpu = Cascading(graph, backend='gpu', **params)

    assert gpu.run_single_sim() == cpu.run_single_sim()
    assert gpu.failed == cpu.failed
    assert gpu.sim_info.keys() == cpu.sim_info.keys()


def test_gpu_backend_rejects_asynchronous_voter_model():
    with pytest.raises(ValueError):
        Influence(
            nx.path_graph(2), model='voter', initial_state={0: 0, 1: 1},
            tracked_state=1, runs=1, steps=1, backend='gpu'
        )
