import importlib.util
import shutil
import subprocess
from functools import lru_cache

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


_BACKENDS = {'auto', 'cpu', 'gpu'}

_AUTO_GPU_THRESHOLDS = {
    'average_distance': 250,
    'average_inverse_distance': 250,
    'diameter': 250,
    'average_vertex_betweenness': 500,
    'average_edge_betweenness': 5000,
    'average_clustering_coefficient': 5000,
    'pagerank': 5000,
    'eigenvector': 5000,
    'motter_lai': 1000,
    'spectral_radius': 20000,
    'spectral_gap': 20000,
    'algebraic_connectivity': 20000,
    'natural_connectivity': {True: 250, False: None},
    'number_spanning_trees': {True: 250, False: 20000},
    'effective_resistance': {True: 250, False: 20000},
    'spectral_scaling': None,
    'generalized_robustness_index': None,
    'largest_connected_component': None,
    'sis': None,
    'sir': None,
    'independent_cascade': None,
    'linear_threshold': None,
    'competitive_cascade': None,
    'netshield': None
}


def automatic_gpu_threshold(operation, exact=False):
    """Return the measured auto-selection threshold for one operation.

    ``None`` means automatic execution stays on the CPU. Unknown operations
    retain the historical 1,000-node threshold for compatibility.
    """

    if operation is None:
        return 1000
    policy = _AUTO_GPU_THRESHOLDS.get(str(operation).lower(), 1000)
    if isinstance(policy, dict):
        return policy[bool(exact)]
    return policy


def system_gpu_status():
    """Detect NVIDIA hardware through ``nvidia-smi`` without importing CuPy."""

    status = {
        'available': False,
        'devices': [],
        'reason': 'nvidia-smi was not found'
    }
    command = shutil.which('nvidia-smi')
    if command is None:
        return status

    try:
        result = subprocess.run(
            [command, '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode != 0:
            status['reason'] = result.stderr.strip() or 'nvidia-smi failed'
            return status

        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(',')]
            if len(fields) != 3:
                continue
            status['devices'].append({
                'name': fields[0],
                'driver_version': fields[1],
                'memory_mib': int(float(fields[2]))
            })
        if status['devices']:
            status['available'] = True
            status['reason'] = 'NVIDIA driver and GPU detected'
        else:
            status['reason'] = 'nvidia-smi found no NVIDIA GPUs'
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        status['reason'] = '{}: {}'.format(type(error).__name__, error)

    return status


@lru_cache(maxsize=1)
def gpu_status():
    """Return information about the first CUDA device visible to CuPy."""

    status = {
        'available': False,
        'device_count': 0,
        'device_name': None,
        'free_memory': None,
        'total_memory': None,
        'reason': 'CuPy is not installed'
    }
    if importlib.util.find_spec('cupy') is None:
        return status

    try:
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            status['reason'] = 'CuPy found no CUDA devices'
            return status

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties.get('name', 'CUDA device')
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()

        cp.asarray([0.0])
        cp.cuda.Stream.null.synchronize()
        status.update({
            'available': True,
            'device_count': device_count,
            'device_name': name,
            'free_memory': int(free_memory),
            'total_memory': int(total_memory),
            'reason': 'CUDA device is ready'
        })
    except Exception as error:
        status['reason'] = '{}: {}'.format(type(error).__name__, error)

    return status


def gpu_available():
    """Return whether CuPy can execute work on a visible CUDA device."""

    return gpu_status()['available']


@lru_cache(maxsize=1)
def networkx_gpu_status():
    """Return whether NetworkX can dispatch algorithms to nx-cugraph."""

    status = gpu_status().copy()
    status['backend'] = 'cugraph'
    if not status['available']:
        return status

    try:
        backends = getattr(nx.config, 'backends', {})
    except AttributeError:
        status['available'] = False
        status['reason'] = 'NetworkX does not support backend dispatch'
        return status

    if 'cugraph' not in backends:
        status['available'] = False
        status['reason'] = 'nx-cugraph is not installed'
        return status

    status['reason'] = 'nx-cugraph and a CUDA device are ready'
    return status


def select_networkx_backend(
        graph, backend='auto', min_gpu_nodes=None, operation=None):
    """Resolve CPU NetworkX or its optional nx-cugraph backend.

    :param graph: NetworkX graph supplied to a dispatchable algorithm
    :param backend: auto, cpu, or gpu
    :param min_gpu_nodes: optional user override for the automatic threshold
    :param operation: algorithm name used by the measured automatic policy
    :return: dictionary describing the selection
    """

    if backend not in _BACKENDS:
        raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")
    if min_gpu_nodes is not None and min_gpu_nodes < 0:
        raise ValueError('min_gpu_nodes must be nonnegative')

    threshold = (automatic_gpu_threshold(operation)
                 if min_gpu_nodes is None else min_gpu_nodes)

    status = networkx_gpu_status()
    result = {
        'requested': backend,
        'selected': 'cpu',
        'available': status['available'],
        'suitable': False,
        'reason': 'CPU requested',
        'nodes': len(graph),
        'edges': graph.number_of_edges(),
        'device_name': status['device_name']
    }

    if backend == 'cpu':
        return result
    if not status['available']:
        if backend == 'gpu':
            raise RuntimeError('GPU requested but unavailable: {}'.format(status['reason']))
        result['reason'] = status['reason']
        return result

    result['suitable'] = threshold is not None and len(graph) >= threshold
    if backend == 'gpu' or result['suitable']:
        result['selected'] = 'gpu'
        result['reason'] = ('GPU explicitly requested' if backend == 'gpu'
                            else 'nx-cugraph is available and the graph is large enough')
    else:
        if threshold is None:
            result['reason'] = 'automatic policy keeps {} on CPU'.format(operation)
        else:
            result['reason'] = 'graph has fewer than {} nodes'.format(threshold)

    return result


def networkx_backend(
        graph, backend='auto', min_gpu_nodes=None, operation=None):
    """Return the explicit NetworkX backend name for one graph algorithm."""

    selected = select_networkx_backend(
        graph, backend=backend, min_gpu_nodes=min_gpu_nodes,
        operation=operation
    )['selected']
    return 'cugraph' if selected == 'gpu' else 'networkx'


def networkx_backend_kwargs(
        graph, backend='auto', min_gpu_nodes=None, operation=None):
    """Return backend-dispatch keywords compatible with the installed NetworkX.

    NetworkX releases before backend dispatch do not accept a ``backend``
    keyword. CPU execution therefore omits it on those releases, while an
    explicit GPU request still fails through :func:`select_networkx_backend`
    with a useful availability error.
    """

    selected = networkx_backend(
        graph, backend, min_gpu_nodes, operation=operation
    )
    if selected == 'networkx' and not hasattr(nx, 'config'):
        return {}
    return {'backend': selected}


def counter_random(xp, size, seed, offset=0):
    """Return reproducible uniforms computed identically by NumPy and CuPy."""

    values = xp.arange(size, dtype=xp.uint64)
    values = values + xp.uint64(seed) + xp.uint64(offset)
    values = values + xp.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)
    values = values ^ (values >> xp.uint64(31))
    return (values >> xp.uint64(11)).astype(xp.float64) * (1.0 / (1 << 53))


def _memory_required(graph, exact, k):
    """Conservative matrix and eigensolver workspace estimate in bytes."""

    n = len(graph)
    if exact:
        return 4 * n * n * np.dtype(float).itemsize

    nnz = 2 * graph.number_of_edges()
    csr_bytes = nnz * (np.dtype(float).itemsize + np.dtype(np.int64).itemsize)
    csr_bytes += (n + 1) * np.dtype(np.int64).itemsize
    eigenpairs = min(max(0, int(k)), max(0, len(graph) - 1))
    eigenvectors = len(graph) * eigenpairs * np.dtype(float).itemsize
    return 4 * (csr_bytes + eigenvectors)


def select_backend(
        graph, backend='auto', k=np.inf, min_gpu_nodes=None, operation=None):
    """Resolve a requested compute backend for a spectral calculation.

    CPU always selects SciPy. GPU requires a working CUDA device and raises if
    one is unavailable. Auto uses the GPU only when the graph is large enough
    and the estimated working set fits comfortably in free device memory.

    :param graph: NetworkX graph
    :param backend: auto, cpu, or gpu
    :param k: requested number of eigenpairs; infinity denotes a full spectrum
    :param min_gpu_nodes: optional user override for the automatic threshold
    :param operation: calculation name used by the measured automatic policy
    :return: dictionary describing the selection
    """

    if backend not in _BACKENDS:
        raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")
    if min_gpu_nodes is not None and min_gpu_nodes < 0:
        raise ValueError('min_gpu_nodes must be nonnegative')

    status = gpu_status().copy()
    n = len(graph)
    exact = np.isinf(k) or k >= n
    threshold = (automatic_gpu_threshold(operation, exact=exact)
                 if min_gpu_nodes is None else min_gpu_nodes)
    required = _memory_required(graph, exact, k)
    result = {
        'requested': backend,
        'selected': 'cpu',
        'available': status['available'],
        'suitable': False,
        'reason': 'CPU requested',
        'nodes': n,
        'edges': graph.number_of_edges(),
        'exact': bool(exact),
        'estimated_gpu_bytes': int(required),
        'device_name': status['device_name'],
        'free_gpu_bytes': status['free_memory']
    }

    if backend == 'cpu':
        return result

    if not status['available']:
        if backend == 'gpu':
            raise RuntimeError('GPU requested but unavailable: {}'.format(status['reason']))
        result['reason'] = status['reason']
        return result

    memory_ok = required <= 0.5 * status['free_memory']
    large_enough = threshold is not None and n >= threshold
    result['suitable'] = memory_ok and large_enough

    if backend == 'gpu':
        if not memory_ok:
            raise MemoryError(
                'estimated GPU working set ({}) exceeds half of free device memory ({})'.format(
                    required, status['free_memory']
                )
            )
        result['selected'] = 'gpu'
        result['reason'] = 'GPU explicitly requested'
        return result

    if not large_enough:
        if threshold is None:
            result['reason'] = 'automatic policy keeps {} on CPU'.format(operation)
        else:
            result['reason'] = 'graph has fewer than {} nodes'.format(threshold)
    elif not memory_ok:
        result['reason'] = 'estimated working set exceeds half of free GPU memory'
    else:
        result['selected'] = 'gpu'
        result['reason'] = 'GPU is available and the workload is large enough'

    return result


def _legacy_backend(backend, use_gpu):
    if use_gpu is None:
        return backend
    if backend != 'cpu':
        raise ValueError('specify backend or use_gpu, not both')
    return 'gpu' if use_gpu else 'cpu'


def get_sparse_graph(graph):
    """Return the adjacency matrix in SciPy CSR format."""

    if len(graph) == 0:
        return sparse.csr_matrix((0, 0), dtype=float)
    if hasattr(nx, 'to_scipy_sparse_array'):
        return nx.to_scipy_sparse_array(
            graph, format='csr', dtype=float, nodelist=list(graph.nodes)
        )

    return nx.to_scipy_sparse_matrix(
        graph, format='csr', dtype=float, nodelist=list(graph.nodes)
    )


def get_largest_component_size(
        graph, backend='auto', min_gpu_nodes=1000):
    """Return the largest undirected component size on CPU or GPU.

    Explicit GPU execution remains available for parity checks and future
    crossover calibration.  ``auto`` deliberately stays on NetworkX because
    measured GPU end-to-end time is slower through 20,000 nodes.
    """

    if len(graph) == 0:
        return 0

    selected = 'cpu' if backend == 'auto' else select_backend(
        graph, backend=backend, k=1, min_gpu_nodes=min_gpu_nodes
    )['selected']
    if selected == 'cpu':
        return len(max(nx.connected_components(graph), key=len))

    import cupy as cp
    matrix = get_sparse_graph(graph).tocoo()
    sources = cp.asarray(matrix.row, dtype=cp.int64)
    targets = cp.asarray(matrix.col, dtype=cp.int64)
    labels = cp.arange(len(graph), dtype=cp.int64)
    while True:
        updated = labels.copy()
        cp.minimum.at(updated, targets, labels[sources])
        updated = updated[updated]
        changed = bool(cp.any(updated != labels).item())
        labels = updated
        if not changed:
            break
    return int(cp.bincount(labels).max().item())


def get_shortest_path_statistics(
        graph, backend='auto', min_gpu_nodes=None, block_size=1024,
        operation='average_distance'):
    """Reduce unweighted all-pairs distances without returning path dictionaries.

    The returned sums count ordered source-target pairs and exclude self-pairs.
    GPU execution processes sources in bounded dense blocks while retaining the
    sparse graph and every reduction on the device.
    """

    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError('block_size must be a positive integer')

    n = len(graph)
    if n == 0:
        return {
            'distance_sum': 0,
            'inverse_distance_sum': 0.0,
            'diameter': 0,
            'reachable_pairs': 0,
            'backend': 'cpu'
        }

    selected = select_backend(
        graph, backend=backend, k=1, min_gpu_nodes=min_gpu_nodes,
        operation=operation
    )['selected']
    if selected == 'cpu':
        distance_sum = 0
        inverse_sum = 0.0
        diameter = 0
        reachable_pairs = 0
        for source, distances in nx.all_pairs_shortest_path_length(graph):
            for target, distance in distances.items():
                if source == target:
                    continue
                distance_sum += distance
                inverse_sum += 1.0 / distance
                diameter = max(diameter, distance)
                reachable_pairs += 1
        return {
            'distance_sum': distance_sum,
            'inverse_distance_sum': inverse_sum,
            'diameter': diameter,
            'reachable_pairs': reachable_pairs,
            'backend': selected
        }

    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix

    status = gpu_status()
    bytes_per_source = max(1, 12 * n)
    memory_block = max(1, int(0.25 * status['free_memory'] / bytes_per_source))
    block_size = min(int(block_size), n, memory_block)

    adjacency = csr_matrix(get_sparse_graph(graph), dtype=cp.float32).transpose().tocsr()
    adjacency.data.fill(1)
    distance_sum = 0
    inverse_sum = 0.0
    diameter = 0
    reachable_pairs = 0

    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        width = stop - start
        frontier = cp.zeros((n, width), dtype=cp.float32)
        frontier[cp.arange(start, stop), cp.arange(width)] = 1
        visited = frontier.astype(cp.bool_)
        depth = 0

        while True:
            reached = adjacency.dot(frontier)
            discovered = (reached > 0) & ~visited
            count = int(discovered.sum().item())
            if count == 0:
                break
            depth += 1
            distance_sum += depth * count
            inverse_sum += count / depth
            diameter = max(diameter, depth)
            visited |= discovered
            frontier = discovered.astype(cp.float32)

        reachable_pairs += int(visited.sum().item()) - width

    return {
        'distance_sum': distance_sum,
        'inverse_distance_sum': inverse_sum,
        'diameter': diameter,
        'reachable_pairs': reachable_pairs,
        'backend': selected
    }


def _gpu_dense_spectrum(matrix, eigvals_only):
    import cupy as cp

    matrix_gpu = cp.asarray(matrix)
    if eigvals_only:
        result = cp.linalg.eigvalsh(matrix_gpu)
    else:
        result = cp.linalg.eigh(matrix_gpu)

    if isinstance(result, tuple):
        return tuple(cp.asnumpy(item) for item in result)
    return cp.asnumpy(result)


def _gpu_sparse_spectrum(matrix, k, which, eigvals_only, tol=0):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

    matrix_gpu = csr_matrix(matrix)
    result = cupy_eigsh(
        matrix_gpu,
        k=k,
        which=which,
        tol=tol,
        return_eigenvectors=not eigvals_only
    )

    if isinstance(result, tuple):
        return tuple(cp.asnumpy(item) for item in result)
    return cp.asnumpy(result)


def _restore_laplacian_zero_modes(graph, eigpairs, k, eigvals_only):
    """Return the known Laplacian nullspace plus computed nonzero modes."""

    if eigvals_only:
        values = np.asarray(eigpairs)
        vectors = None
    else:
        values, vectors = eigpairs
        values = np.asarray(values)
        vectors = np.asarray(vectors)

    zero_count = min(nx.number_connected_components(graph), k)
    near_zero = np.flatnonzero(np.abs(values) <= 1e-10)
    remove_count = min(zero_count, len(near_zero))
    remove = set(near_zero[np.argsort(np.abs(values[near_zero]))[:remove_count]])
    nonzero = np.array(
        [index for index in np.argsort(values) if index not in remove],
        dtype=int
    )
    nonzero = nonzero[:max(0, k - zero_count)]
    restored_values = np.concatenate((
        np.zeros(zero_count, dtype=float), values[nonzero]
    ))

    if eigvals_only:
        return restored_values

    node_index = {node: index for index, node in enumerate(graph.nodes)}
    null_vectors = np.zeros((len(graph), zero_count), dtype=float)
    for column, component in enumerate(nx.connected_components(graph)):
        if column >= zero_count:
            break
        indices = [node_index[node] for node in component]
        null_vectors[indices, column] = 1.0 / np.sqrt(len(indices))
    restored_vectors = np.column_stack((
        null_vectors, vectors[:, nonzero]
    ))
    return restored_values, restored_vectors


def get_adjacency_spectrum(
        graph, k=np.inf, eigvals_only=False, which='LA', backend='cpu',
        min_gpu_nodes=None, use_gpu=None, operation=None):
    """Get the top k eigenpairs of the adjacency matrix.

    :param graph: undirected NetworkX graph
    :param k: number of eigenpairs; infinity requests the full spectrum
    :param eigvals_only: return eigenvalues without eigenvectors
    :param which: eigenpairs requested by the sparse solver
    :param backend: cpu, gpu, or auto
    :param min_gpu_nodes: auto-selection threshold
    :param use_gpu: backward-compatible Boolean alias for backend
    :param operation: measure name used by the automatic backend policy
    """

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    backend = _legacy_backend(backend, use_gpu)
    exact = np.isinf(k) or k >= n
    dense = exact or n < 100
    selection_k = np.inf if dense else k
    selected = select_backend(
        graph, backend=backend, k=selection_k,
        min_gpu_nodes=min_gpu_nodes, operation=operation
    )['selected']

    if dense:
        matrix = nx.to_numpy_array(
            graph, nodelist=list(graph.nodes), dtype=float
        )
        if selected == 'gpu':
            return _gpu_dense_spectrum(matrix, eigvals_only)
        return eigh(matrix, eigvals_only=eigvals_only)

    k = min(int(k), n - 1)
    if k <= 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((n, 0))

    matrix = get_sparse_graph(graph)
    if selected == 'gpu':
        return _gpu_sparse_spectrum(matrix, k, which, eigvals_only)

    if n < 100:
        dense = matrix.toarray()
        values, vectors = eigh(dense)
        if which in {'LM', 'SM'}:
            order = np.abs(values).argsort()
        else:
            order = values.argsort()
        if which in {'LA', 'LM'}:
            order = order[::-1]
        order = order[:k]
        if eigvals_only:
            return values[order]
        return values[order], vectors[:, order]

    return eigsh(
        matrix, k=k, which=which, return_eigenvectors=not eigvals_only
    )


def get_laplacian_spectrum(
        graph, k=np.inf, which='SM', tol=1E-8, eigvals_only=True,
        backend='cpu', min_gpu_nodes=None, use_gpu=None, operation=None):
    """Get the bottom k eigenpairs of the Laplacian matrix.

    :param graph: undirected NetworkX graph
    :param k: number of eigenpairs; infinity requests the full spectrum
    :param which: eigenpairs requested by the sparse solver
    :param tol: sparse-solver tolerance
    :param eigvals_only: return eigenvalues without eigenvectors
    :param backend: cpu, gpu, or auto
    :param min_gpu_nodes: auto-selection threshold
    :param use_gpu: backward-compatible Boolean alias for backend
    :param operation: measure name used by the automatic backend policy
    """

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    backend = _legacy_backend(backend, use_gpu)
    exact = np.isinf(k) or k >= n
    dense = exact or n < 100
    selection_k = np.inf if dense else k
    selected = select_backend(
        graph, backend=backend, k=selection_k,
        min_gpu_nodes=min_gpu_nodes, operation=operation
    )['selected']

    if dense:
        matrix = nx.laplacian_matrix(
            graph, nodelist=list(graph.nodes)
        ).toarray().astype(float)
        if selected == 'gpu':
            eigpairs = _gpu_dense_spectrum(matrix, eigvals_only)
        else:
            eigpairs = eigh(matrix, eigvals_only=eigvals_only)
    else:
        matrix = get_laplacian(graph)
        k = min(int(k), n - 1)
        if selected == 'gpu':
            # CuPy does not support SciPy's ``SM`` selector.  A graph
            # Laplacian is positive semidefinite, so ``SA`` is equivalent.
            sparse_which = 'SA' if which == 'SM' else which
            eigpairs = _gpu_sparse_spectrum(
                matrix, k, sparse_which, eigvals_only, tol=tol
            )
        else:
            eigpairs = eigsh(
                matrix, k=k, which=which, tol=tol,
                return_eigenvectors=not eigvals_only
            )
        if which == 'SM':
            eigpairs = _restore_laplacian_zero_modes(
                graph, eigpairs, k, eigvals_only
            )

    if eigvals_only:
        return np.sort(eigpairs)

    values, vectors = eigpairs
    order = values.argsort()
    return values[order], vectors[:, order]


def get_laplacian(graph):
    """Return the graph Laplacian in SciPy CSR format."""

    adjacency = get_sparse_graph(graph)
    degree = np.asarray(adjacency.sum(axis=1)).flatten()
    diagonal = sparse.diags(
        degree, offsets=0, shape=(len(graph), len(graph)), format='csr'
    )

    return diagonal - adjacency
