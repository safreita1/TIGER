import importlib.util
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def gpu_available():
    """
    Return whether CuPy is importable without relying on pip internals.
    """

    return importlib.util.find_spec('cupy') is not None


def get_sparse_graph(graph):
    """
    Returns a sparse adjacency matrix in CSR format

    :param graph: undirected NetworkX graph
    :return: Scipy sparse adjacency matrix
    """

    if len(graph) == 0:
        return sparse.csr_matrix((0, 0), dtype=float)
    if hasattr(nx, 'to_scipy_sparse_array'):
        return nx.to_scipy_sparse_array(graph, format='csr', dtype=float, nodelist=list(graph.nodes))

    return nx.to_scipy_sparse_matrix(graph, format='csr', dtype=float, nodelist=list(graph.nodes))


def get_adjacency_spectrum(graph, k=np.inf, eigvals_only=False, which='LA', use_gpu=False):
    """
    Gets the top k eigenpairs of the adjacency matrix

    Exact spectra are returned when ``k`` is infinite or at least the graph order.

    :param graph: undirected NetworkX graph
    :param k: number of top k eigenpairs to obtain
    :param eigvals_only: get only the eigenvalues i.e., no eigenvectors
    :param which: the type of k eigenvectors and eigenvalues to find
    :param use_gpu: use CuPy for an explicitly requested partial spectrum
    :return: the eigenpair information
    """

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    exact = np.isinf(k) or k >= n
    if exact or n < 100:
        A = nx.to_numpy_array(graph, nodelist=list(graph.nodes), dtype=float)
        return eigh(A, eigvals_only=eigvals_only)

    A = get_sparse_graph(graph)
    k = min(int(k), n - 1)

    if k <= 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((n, 0))

    if gpu_available() and use_gpu:
        import cupy as cp
        import cupyx.scipy.sparse.linalg as cp_linalg

        A_gpu = cp.sparse.csr_matrix(A)
        eigpairs = cp_linalg.eigsh(A_gpu, k=k, which=which,
                                  return_eigenvectors=not eigvals_only)

        if type(eigpairs) is tuple:
            eigpairs = list(eigpairs)
            eigpairs[0], eigpairs[1] = cp.asnumpy(eigpairs[0]), cp.asnumpy(eigpairs[1])
            return tuple(eigpairs)

        return cp.asnumpy(eigpairs)

    if use_gpu:
        print('Warning: GPU requested, but not available')

    return eigsh(A, k=k, which=which, return_eigenvectors=not eigvals_only)


def get_laplacian_spectrum(graph, k=np.inf, which='SM', tol=1E-8, eigvals_only=True, use_gpu=False):
    """
    Gets the bottom k eigenpairs of the Laplacian matrix

    Exact spectra are returned when ``k`` is infinite or at least the graph order.

    :param graph: undirected NetworkX graph
    :param k: number of bottom k eigenpairs to obtain
    :param which: the type of k eigenvectors and eigenvalues to find
    :param tol: the precision at which to stop computing partial eigenpairs
    :param eigvals_only: get only the eigenvalues i.e., no eigenvectors
    :param use_gpu: retained for API compatibility; Laplacian GPU is unavailable
    :return: the eigenpair information
    """

    if use_gpu:
        print('Warning: GPU requested, but not available for Laplacian measures')

    n = len(graph)
    if n == 0:
        if eigvals_only:
            return np.array([])
        return np.array([]), np.empty((0, 0))

    exact = np.isinf(k) or k >= n
    if exact or n < 100:
        L = nx.laplacian_matrix(graph, nodelist=list(graph.nodes)).toarray().astype(float)
        eigpairs = eigh(L, eigvals_only=eigvals_only)
    else:
        L = get_laplacian(graph)
        eigpairs = eigsh(L, k=min(int(k), n - 1), which=which, tol=tol,
                         return_eigenvectors=not eigvals_only)

    if eigvals_only:
        return np.sort(eigpairs)

    lam, vectors = eigpairs
    idx = lam.argsort()
    return lam[idx], vectors[:, idx]


def get_laplacian(graph):
    """
    Gets the Laplacian matrix in sparse CSR format

    :param graph: undirected NetworkX graph
    :return: Scipy sparse Laplacian matrix
    """

    A = get_sparse_graph(graph)
    degree = np.asarray(A.sum(axis=1)).flatten()
    D = sparse.diags(degree, offsets=0, shape=(len(graph), len(graph)), format='csr')

    return D - A
