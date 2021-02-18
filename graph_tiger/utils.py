import bezier
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def get_sparse_graph(graph):
    """
    Returns a sparse adjacency matrix in CSR format

    :param graph: undirected NetworkX graph
    :return: Scipy sparse adjacency matrix
    """

    return nx.to_scipy_sparse_matrix(graph, format='csr', dtype=np.float, nodelist=graph.nodes)


def get_adjacency_spectrum(graph, k=np.inf, eigvals_only=False, which='LA'):
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
        eigpairs = eigsh(A, k=min(k, len(graph) - 1), which=which, return_eigenvectors=not eigvals_only)

    return eigpairs


def get_laplacian_spectrum(graph, k=np.inf, which='SM', tol=1E-2, eigvals_only=True):
    """
    Gets the bottom k eigenpairs of the Laplacian matrix

    :param graph: undirected NetworkX graph
    :param k: number of bottom k eigenpairs to obtain
    :param which:  he type of k eigenvectors and eigenvalues to find
    :param tol: the precision at which to stop computing the eigenpairs
    :param eigvals_only: get only the eigenvalues i.e., no eigenvectors

    :return: the eigenpair information
    """

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


def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
    """
    Internal function to enable Bezier curved edges. Code originally from 'beyondbeneath' @ https://github.com/beyondbeneath/bezier-curved-edges-networkx
    """
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l) == 0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:, 0]) + np.vectorize(hash)(edges[:, 1]), 2) == 0, -1, 1)

    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse=True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:, 0, :]
    coords_node2 = coords[:, 1, :]

    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:, 0] > coords_node2[:, 0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]

    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1 - coords_node2) ** 2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:, 1] - coords_node1[:, 1]) / (coords_node2[:, 0] - coords_node1[:, 0])
    m2 = -1 / m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist / np.sqrt(1 + m1 ** 2)
    v1 = np.array([np.ones(l), m1])
    coords_node1_displace = coords_node1 + (v1 * t1).T
    coords_node2_displace = coords_node2 - (v1 * t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist / np.sqrt(1 + m2 ** 2)
    v2 = np.array([np.ones(len(edges)), m2])
    coords_node1_ctrl = coords_node1_displace + (rnd * v2 * t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd * v2 * t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:, i, :].T
        curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0, 1, bezier_precision)).T)

    # Return an array of these curves
    curves = np.array(curveplots)
    return curves