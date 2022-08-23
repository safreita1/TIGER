import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def gpu_available():
    from pip._internal.utils.misc import get_installed_distributions

    gpu = False
    installed_packages = [package.project_name for package in get_installed_distributions()]

    if any("cupy" in s for s in installed_packages):
        gpu = True

    return gpu


def get_sparse_graph(graph):
    """
    Returns a sparse adjacency matrix in CSR format

    :param graph: undirected NetworkX graph
    :return: Scipy sparse adjacency matrix
    """

    return nx.to_scipy_sparse_matrix(graph, format='csr', dtype=float, nodelist=graph.nodes)


def get_adjacency_spectrum(graph, k=np.inf, eigvals_only=False, which='LA', use_gpu=False):
    """
    Gets the top k eigenpairs of the adjacency matrix

    :param graph: undirected NetworkX graph
    :param k: number of top k eigenpairs to obtain
    :param eigvals_only: get only the eigenvalues i.e., no eigenvectors
    :param which:  the type of k eigenvectors and eigenvalues to find
    :return: the eigenpair information
    """

    # get all eigenpairs for small graphs
    if len(graph) < 100:
        A = nx.adjacency_matrix(graph).todense()
        eigpairs = eigh(A, eigvals_only=eigvals_only)

    else:
        A = nx.to_scipy_sparse_matrix(graph, format='csr', dtype=np.float, nodelist=graph.nodes)

        if gpu_available() and use_gpu:
            import cupy as cp
            import cupyx.scipy.sparse.linalg as cp_linalg

            A_gpu = cp.sparse.csr_matrix(A)
            eigpairs = cp_linalg.eigsh(A_gpu, k=min(k, len(graph) - 3), which=which, return_eigenvectors=not eigvals_only)

            if type(eigpairs) is tuple:
                eigpairs = list(eigpairs)
                eigpairs[0], eigpairs[1] = cp.asnumpy(eigpairs[0]), cp.asnumpy(eigpairs[1])
            else:
                eigpairs = cp.asnumpy(eigpairs)

        else:
            if use_gpu: print('Warning: GPU requested, but not available')
            eigpairs = eigsh(A, k=min(k, len(graph) - 1), which=which, return_eigenvectors=not eigvals_only)

    return eigpairs


def get_laplacian_spectrum(graph, k=np.inf, which='SM', tol=1E-2, eigvals_only=True, use_gpu=False):
    """
    Gets the bottom k eigenpairs of the Laplacian matrix

    :param graph: undirected NetworkX graph
    :param k: number of bottom k eigenpairs to obtain
    :param which:  he type of k eigenvectors and eigenvalues to find
    :param tol: the precision at which to stop computing the eigenpairs
    :param eigvals_only: get only the eigenvalues i.e., no eigenvectors

    :return: the eigenpair information
    """

    if use_gpu: print('Warning: GPU requested, but not available for Laplacian measures')

    # get all eigenvalues for small graphs
    if len(graph) < 100:
        lam = nx.laplacian_spectrum(graph)
    else:
        L = get_laplacian(graph)
        lam = eigsh(L, k=min(k, len(graph) - 1), which=which, tol=tol, return_eigenvectors=not eigvals_only)

    lam = np.sort(lam)  # sort ascending

    return lam


def get_laplacian(graph):
    """
    Gets the Laplacian matrix in sparse CSR format

    :param graph: undirected NetworkX graph
    :return: Scipy sparse Laplacian matrix
    """
    A = nx.to_scipy_sparse_matrix(graph, format='csr', dtype=np.float, nodelist=graph.nodes)
    D = sparse.spdiags(data=A.sum(axis=1).flatten(), diags=[0], m=len(graph), n=len(graph), format='csr')
    L = D - A

    return L
