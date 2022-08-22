from graph_tiger.graphs import karate
from graph_tiger.diffusion import Diffusion


def run_test(params):
    graph = karate()
    ds = Diffusion(graph, **params)
    ds.run_simulation()


def test_animation():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'plot_transition': False,
        'gif_animation': True
    }

    run_test(params)


def test_transition():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'plot_transition': True,
        'gif_animation': False
    }

    run_test(params)


def test_gif_snaps():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'plot_transition': False,
        'gif_animation': True,
        'gif_snaps': True
    }

    run_test(params)


def test_force_atlas():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'edge_style': None,
        'node_style': 'force_atlas',
        'fa_iter': 200,
        'plot_transition': True,
        'gif_animation': False
    }

    run_test(params)


def test_curved_edges():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'edge_style': 'curved',
        'node_style': 'force_atlas',
        'fa_iter': 200,
        'plot_transition': True,
        'gif_animation': False
    }

    run_test(params)


def test_edge_bundling():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 200,
        'plot_transition': True,
        'gif_animation': False
    }

    run_test(params)


def main():
    test_animation()
    test_transition()
    test_gif_snaps()
    test_force_atlas()
    test_curved_edges()
    test_edge_bundling()


if __name__ == '__main__':
    main()
