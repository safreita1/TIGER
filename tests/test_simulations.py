import numpy as np

from graphs import karate
from cascading import Cascading
from diffusion import Diffusion


def test_sis_model():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'seed': 1,
        'plot_transition': False,
        'gif_animation': False
    }

    graph = karate()

    ds = Diffusion(graph, params)
    increased_diffusion = ds.run_simulation()

    params['diffusion'] = None
    params['method'] = None
    params['k'] = 0

    ds = Diffusion(graph, params)
    baseline_diffusion = ds.run_simulation()

    params['diffusion'] = 'min'
    params['method'] = 'ns_node'
    params['k'] = 4

    ds = Diffusion(graph, params)
    decreased_diffusion = ds.run_simulation()

    assert sum(decreased_diffusion) < sum(baseline_diffusion) < sum(increased_diffusion)


def test_sir_model():
    params = {
        'model': 'SIR',
        'b': 0.00208,
        'd': 0.01,
        'c': 0.1,
        'runs': 10,
        'steps': 5000,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 40,

        'seed': 1,
        'plot_transition': False,
        'gif_animation': False
    }

    graph = karate()

    ds = Diffusion(graph, params)
    increased_diffusion = ds.run_simulation()

    params['diffusion'] = None
    params['method'] = None
    params['k'] = 0

    ds = Diffusion(graph, params)
    baseline_diffusion = ds.run_simulation()

    params['diffusion'] = 'min'
    params['method'] = 'ns_node'
    params['k'] = 4

    ds = Diffusion(graph, params)
    decreased_diffusion = ds.run_simulation()

    assert sum(decreased_diffusion) < sum(baseline_diffusion) < sum(increased_diffusion)


def test_cascading():
    params = {
        'runs': 10,
        'steps': 30,

        'l': 0.8,
        'r': 0.5,
        'capacity_approx': np.inf,

        'k_a': 4,
        'attack': 'rnd_node',

        'k_d': 0,
        'defense': None,

        'robust_measure': 'largest_connected_component',

        'seed': 1,
        'plot_transition': False,
        'gif_animation': False
    }

    graph = karate()

    cf = Cascading(graph, params)
    attacked = cf.run_simulation()

    params['k_a'] = 0
    params['attack'] = None

    cf = Cascading(graph, params)
    baseline = cf.run_simulation()

    params['k_a'] = 4
    params['attack'] = 'rnd_node'

    params['k_d'] = 4
    params['defense'] = 'pr_node'

    cf = Cascading(graph, params)
    defended = cf.run_simulation()

    assert sum(attacked) < sum(defended) <= sum(baseline)


def main():
    test_sis_model()
    test_sir_model()
    test_cascading()


if __name__ == '__main__':
    main()
