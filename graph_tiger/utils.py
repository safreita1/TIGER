import importlib.util
from functools import lru_cache

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


_BACKENDS = {'auto', 'cpu', 'gpu'}


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


def _memory_required(graph, exact):
    """Conservative matrix and eigensolver workspace estimate in bytes."""

    n = len(graph)
    if exact:
        return 4 * n * n * np.dtype(float).itemsize

    nnz = 2 * graph.number_of_edges()
    csr_bytes = nnz * (np.dtype(float).itemsize + np.dtype(np.int64).itemsize)
    csr_bytes += (n + 1) * np.dtype(np.int64).itemsize
    return 4 * csr_bytes


def select_backend(graph, backend='auto', k=np.inf, min_gpu_nodes=1000):
    """Resolve a requested compute backend for a spectral calculation.

    CPU always selects SciPy. GPU requires a working CUDA device and raises if
    one is unavailable. Auto uses the GPU only when the graph is large enough
    and the estimated working set fits comfortably in free device memory.

    :param graph: NetworkX graph
    :param backend: auto, cpu, or gpu
    :param k: requested number of eigenpairs; infinity denotes a full spectrum
    :param min_gpu_nodes: minimum graph order considered worthwhile in auto mode
    :return: dictionary describing the selection
    """

    if backend not in _BACKENDS:
        raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")
    if min_gpu_nodes < 0:
        raise ValueError('min_gpu_nodes must be nonnegative')

    status = gpu_status().copy()
    n = len(graph)
    exact = np.isinf(k) or k >= n
    required = _memory_required(graph, exact)
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
    large_enough = n >= min_gpu_nodes
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
        result['reason'] = 'graph has fewer than {} nodes'.format(min_gpu_nodes)
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


def get_adjacency_spectrum(
        graph, k=np.inf, eigvals_only=False, which='LA', backend='cpu',
        min_gpu_nodes=1000, use_gpu=None):
    """Get the top k eigenpairs of the adjacency matrix.

    :param graph: undirected NetworkX graph
    :param k: number of eigenpairs; infinity requests the full spectrum
    :param eigvals_only: return eigenvalues without eigenvectors
    :param which: eigenpairs requested by the sparse solver
    :param backend: cpu, gpu, or auto
    :param min_gpu_nodes: auto-selection threshold
    :param use_gpu: backward-compatible Boolean alias for backend
    """

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    backend = _legacy_backend(backend, use_gpu)
    exact = np.isinf(k) or k >= n
    selected = select_backend(
        graph, backend=backend, k=k, min_gpu_nodes=min_gpu_nodes
    )['selected']

    if exact:
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
        backend='cpu', min_gpu_nodes=1000, use_gpu=None):
    """Get the bottom k eigenpairs of the Laplacian matrix.

    :param graph: undirected NetworkX graph
    :param k: number of eigenpairs; infinity requests the full spectrum
    :param which: eigenpairs requested by the sparse solver
    :param tol: sparse-solver tolerance
    :param eigvals_only: return eigenvalues without eigenvectors
    :param backend: cpu, gpu, or auto
    :param min_gpu_nodes: auto-selection threshold
    :param use_gpu: backward-compatible Boolean alias for backend
    """

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    backend = _legacy_backend(backend, use_gpu)
    exact = np.isinf(k) or k >= n
    selected = select_backend(
        graph, backend=backend, k=k, min_gpu_nodes=min_gpu_nodes
    )['selected']

    if exact:
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
            eigpairs = _gpu_sparse_spectrum(
                matrix, k, which, eigvals_only, tol=tol
            )
        else:
            eigpairs = eigsh(
                matrix, k=k, which=which, tol=tol,
                return_eigenvectors=not eigvals_only
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
