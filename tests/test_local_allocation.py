import os
import tempfile

import networkx as nx
import numpy as np

from graph_tiger.cascading import Cascading


def make_cascade(graph, loads, capacities, failures, allocation='degree', **kwargs):
    return Cascading(graph, model='local_load_sharing', allocation=allocation,
                     initial_load=loads, capacities=capacities,
                     initial_failures=failures, runs=1, steps=12, seed=17, **kwargs)


def test_greedy_and_proportional_allocations():
    graph = nx.Graph([('F', 'a'), ('F', 'b'), ('F', 'c')])
    expected = {
        12: {'greedy': [0, 5, 7], 'proportional': [2.4, 4, 5.6]},
        18: {'greedy': [4, 6, 8], 'proportional': [3.6, 6, 8.4]}
    }
    for demand, policies in expected.items():
        for allocation, values in policies.items():
            cascade = make_cascade(graph, dict(F=demand, a=10, b=10, c=10),
                                   dict(F=demand, a=13, b=15, c=17), ['F'], allocation)
            failed = cascade.run_local_load_sharing_step()
            actual = [cascade.last_transfers['F', node] for node in 'abc']
            assert np.allclose(actual, values)
            assert failed == (set('abc') if demand == 18 else set())
            assert np.isclose(sum(cascade.load.values()) + cascade.lost_load, demand + 30)


def test_degree_preference_examples():
    graph = nx.Graph([('F', 'a'), ('F', 'b'), ('F', 'c'),
                      ('b', 'x'), ('c', 'y'), ('c', 'z')])
    loads = {node: 0 for node in graph}
    loads['F'] = 12
    for beta, values in [(0, [4, 4, 4]), (1, [2, 4, 6]),
                         (2, [12 / 14, 48 / 14, 108 / 14]), (10000, [0, 0, 12])]:
        cascade = make_cascade(graph, loads, {node: 100 for node in graph}, ['F'], beta=beta)
        cascade.run_local_load_sharing_step()
        assert np.allclose([cascade.last_transfers['F', node] for node in 'abc'], values)


def test_max_flow_coordinates_shared_recipients():
    graph = nx.Graph([('A', 'U'), ('A', 'V'), ('B', 'U')])
    for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
        cascade = make_cascade(graph, dict(A=5, B=7, U=0, V=0),
                               dict(A=5, B=7, U=7, V=5), ['A', 'B'], allocation)
        failed = cascade.run_local_load_sharing_step()
        assert failed == (set() if allocation == 'max_flow' else {'U'})
        assert np.isclose(sum(cascade.last_transfers.values()), 12)
        if allocation == 'max_flow':
            assert cascade.last_transfers == {('A', 'U'): 0, ('A', 'V'): 5, ('B', 'U'): 7}
        assert cascade.lost_load == 0


def test_max_flow_does_not_discard_overflow():
    graph = nx.Graph([('A', 'U'), ('A', 'V'), ('B', 'U')])
    cascade = make_cascade(graph, dict(A=5, B=9, U=0, V=0),
                           dict(A=5, B=9, U=7, V=5), ['A', 'B'], 'max_flow')
    assert cascade.run_local_load_sharing_step() == {'U'}
    assert cascade.load['U'] == 9
    assert cascade.load['V'] == 5
    assert cascade.lost_load == 0


def test_local_policies_accumulate_shared_headroom():
    graph = nx.Graph([('A', 'U'), ('B', 'U')])
    for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
        cascade = make_cascade(graph, dict(A=6, B=6, U=0),
                               dict(A=6, B=6, U=8), ['A', 'B'], allocation)
        assert cascade.run_local_load_sharing_step() == {'U'}
        assert cascade.load['U'] == 12


def test_zero_headroom_and_stranded_work():
    graph = nx.Graph([('F', 'a'), ('F', 'b')])
    graph.add_node('isolated')
    for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
        for demand in [0, 6]:
            cascade = make_cascade(graph, dict(F=demand, a=2, b=2, isolated=4),
                                   dict(F=demand, a=2, b=2, isolated=4), ['F', 'isolated'], allocation)
            failed = cascade.run_local_load_sharing_step()
            assert failed == ({'a', 'b'} if demand else set())
            assert cascade.last_transfers == {('F', 'a'): demand / 2, ('F', 'b'): demand / 2}
            assert cascade.lost_load == 4
            cascade.run_local_load_sharing_step()
            assert np.isclose(sum(cascade.load.values()) + cascade.lost_load, demand + 8)
            cascade.track_simulation(2)
            assert cascade.sim_info[2]['lost_load'] == cascade.sim_info[2]['shed_load']


def test_supplied_inputs_are_copied_and_reset():
    graph = nx.path_graph(3)
    loads = {0: 2, 1: 3, 2: 4}
    capacities = {0: 4, 1: 10, 2: 8}
    original = graph.copy()
    cascade = make_cascade(graph, loads, capacities, [0], r=5)
    loads[0] = 900
    capacities[1] = 900
    assert cascade.load[0] == 2
    assert cascade.capacity[1] == 10
    cascade.run_single_sim()
    cascade.reset_simulation()
    assert cascade.load == {0: 2, 1: 3, 2: 4}
    assert cascade.capacity == {0: 4, 1: 10, 2: 8}
    assert cascade.failed == {0}
    assert cascade.lost_load == 0
    assert cascade.last_transfers == {}
    assert nx.utils.graphs_equal(graph, original)
    cascade.shed_load = 3
    assert cascade.lost_load == 3
    cascade.lost_load = 4
    assert cascade.shed_load == 4


def test_default_capacity_and_empty_initiating_set():
    graph = nx.path_graph(3)
    loads = {0: 2, 1: 3, 2: 4}
    cascade = Cascading(graph, model='local_load_sharing', initial_load=loads,
                        initial_failures=[], r=0.5)
    assert cascade.capacity == {node: 1.5 * value for node, value in loads.items()}
    assert cascade.failed == set()


def test_local_allocation_validation():
    graph = nx.path_graph(3)
    cases = [dict(allocation='wrong'), dict(initial_load={0: 1}),
             dict(capacities={0: 1, 1: 2, 2: 3, 3: 4}),
             dict(initial_load={0: 1, 1: float('nan'), 2: 3}),
             dict(capacities={0: 1, 1: float('inf'), 2: 3}),
             dict(capacities={0: 1, 1: -1, 2: 3}), dict(initial_load={0: 1, 1: '2', 2: 3}),
             dict(initial_failures=[9]), dict(initial_failures='abc'), dict(beta=float('nan'))]
    for options in cases:
        try:
            Cascading(graph, model='local_load_sharing', attack=None, **options)
        except ValueError:
            pass
        else:
            assert False, options
    for model in ['motter_lai', 'crucitti', 'legacy_redistribution']:
        try:
            Cascading(graph, model=model, allocation='max_flow')
        except ValueError:
            pass
        else:
            assert False, model


def test_local_allocation_custom_labels_and_ties():
    nodes = [('tuple', 0), 'x', 9]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from([(nodes[0], node) for node in nodes[1:]])
    for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
        cascade = make_cascade(graph, dict(zip(nodes, [3, 0, 0])),
                               dict(zip(nodes, [3, 3, 3])), [nodes[0]], allocation)
        cascade.run_local_load_sharing_step()
        first = cascade.last_transfers.copy()
        assert sum(first.values()) == 3
        cascade.reset_simulation()
        cascade.run_local_load_sharing_step()
        assert cascade.last_transfers == first
        if allocation == 'greedy':
            assert first[nodes[0], 'x'] == 3


def test_local_allocation_empty_graph():
    for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
        cascade = make_cascade(nx.Graph(), {}, {}, [], allocation)
        assert cascade.run_single_sim() == [0] * 13


def test_random_local_workload_conservation():
    rng = np.random.RandomState(41)
    for seed in range(30):
        graph = nx.gnp_random_graph(9, 0.3, seed=seed)
        loads = {node: float(rng.randint(0, 12)) for node in graph}
        capacities = {node: loads[node] + float(rng.randint(0, 12)) for node in graph}
        for allocation in ['degree', 'greedy', 'proportional', 'max_flow']:
            cascade = make_cascade(graph, loads, capacities, [0, 1, 2], allocation)
            for step in range(10):
                before = cascade.load.copy()
                failed = cascade.failed.copy()
                cascade.run_local_load_sharing_step()
                assert not (failed - cascade.processed)
                for source in {i for i, j in cascade.last_transfers}:
                    sent = sum(value for (i, j), value in cascade.last_transfers.items() if i == source)
                    assert np.isclose(sent, before[source])
                assert all(j not in failed and graph.has_edge(i, j) for i, j in cascade.last_transfers)
                assert np.isclose(sum(cascade.load.values()) + cascade.lost_load, sum(loads.values()))
                assert all(value >= 0 for value in cascade.load.values())


def test_max_flow_matches_independent_min_cut():
    rng = np.random.RandomState(12)
    sources = [0, 1, 2]
    for seed in range(30):
        graph = nx.gnp_random_graph(9, 0.3, seed=seed)
        loads = {node: float(rng.randint(0, 12)) for node in graph}
        capacities = {node: loads[node] + float(rng.randint(0, 12)) for node in graph}
        cascade = make_cascade(graph, loads, capacities, sources, 'max_flow')
        eligible = {i: [j for j in graph[i] if j not in sources] for i in sources}
        spare = {j: capacities[j] - loads[j] for j in graph if j not in sources}
        allocations = cascade._local_capacity_flow(sources, eligible, spare)
        total = sum(sum(values.values()) for values in allocations.values())
        bound = min(sum(loads[i] for k, i in enumerate(sources) if not mask & (1 << k)) +
                    sum(spare[j] for j in set().union(*(set(eligible[i]) for k, i in enumerate(sources)
                                                       if mask & (1 << k))))
                    for mask in range(8))
        assert np.isclose(total, bound)
        for j in spare:
            assert sum(values.get(j, 0) for values in allocations.values()) <= spare[j] + 1e-9


def main():
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            for name, test in list(globals().items()):
                if name.startswith('test_'):
                    test()
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    main()
