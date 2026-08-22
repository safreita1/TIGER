import os
import platform
import tempfile
from pathlib import Path

from graph_tiger.graphs import karate
from graph_tiger.diffusion import Diffusion


def run_test(params):
    cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            graph = karate()
            ds = Diffusion(graph, **params)
            results = ds.run_simulation()

            assert len(results) == params['steps'] + 1

            if params['plot_transition']:
                assert len(list(Path(directory).rglob('*.pdf'))) > 0

            if params['gif_animation'] and platform.system() != 'Windows':
                assert len(list(Path(directory).rglob('*.mp4'))) == 1

            if params.get('gif_snaps') and platform.system() != 'Windows':
                assert len(list(Path(directory).rglob('gif_snaps/*.pdf'))) > 0
        finally:
            os.chdir(cwd)


def test_animation():
    params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 1,
        'steps': 2,
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
        'runs': 1,
        'steps': 2,
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
        'runs': 1,
        'steps': 2,
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
        'model': 'SIR',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 1,
        'steps': 2,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'edge_style': None,
        'node_style': 'force_atlas',
        'fa_iter': 20,
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
        'runs': 1,
        'steps': 2,
        'seed': 1,

        'diffusion': 'max',
        'method': 'add_edge_random',
        'k': 15,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 20,
        'plot_transition': True,
        'gif_animation': False
    }

    run_test(params)


def main():
    test_animation()
    test_transition()
    test_gif_snaps()
    test_force_atlas()
    test_edge_bundling()


if __name__ == '__main__':
    main()
