graph_tiger.utils
=================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.6.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-utils"><details class="api-entry" id="api-utils-gpu_available"><summary><code>gpu_available()</code></summary><div class="api-body"><p>Return whether CuPy is importable without relying on pip internals.</p><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def gpu_available():
       &quot;&quot;&quot;
       Return whether CuPy is importable without relying on pip internals.
       &quot;&quot;&quot;

       return importlib.util.find_spec('cupy') is not None</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/utils.py#L9">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-utils-get_sparse_graph"><summary><code>get_sparse_graph(graph)</code></summary><div class="api-body"><p>Returns a sparse adjacency matrix in CSR format</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> Scipy sparse adjacency matrix</p><details class="source-code"><summary>View source code</summary><pre><code>def get_sparse_graph(graph):
       &quot;&quot;&quot;
       Returns a sparse adjacency matrix in CSR format

       :param graph: undirected NetworkX graph
       :return: Scipy sparse adjacency matrix
       &quot;&quot;&quot;

       if len(graph) == 0:
           return sparse.csr_matrix((0, 0), dtype=float)
       if hasattr(nx, 'to_scipy_sparse_array'):
           return nx.to_scipy_sparse_array(graph, format='csr', dtype=float, nodelist=list(graph.nodes))

       return nx.to_scipy_sparse_matrix(graph, format='csr', dtype=float, nodelist=list(graph.nodes))</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/utils.py#L17">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-utils-get_adjacency_spectrum"><summary><code>get_adjacency_spectrum(graph, k=np.inf, eigvals_only=False, which='LA', use_gpu=False)</code></summary><div class="api-body"><p>Gets the top k eigenpairs of the adjacency matrix

   Exact spectra are returned when k is infinite or at least the graph order.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of top k eigenpairs to obtain</td></tr><tr><td><code>eigvals_only</code></td><td>get only the eigenvalues i.e., no eigenvectors</td></tr><tr><td><code>which</code></td><td>the type of k eigenvectors and eigenvalues to find</td></tr><tr><td><code>use_gpu</code></td><td>use CuPy for an explicitly requested partial spectrum</td></tr></tbody></table></div><p><strong>Returns</strong> the eigenpair information</p><details class="source-code"><summary>View source code</summary><pre><code>def get_adjacency_spectrum(graph, k=np.inf, eigvals_only=False, which='LA', use_gpu=False):
       &quot;&quot;&quot;
       Gets the top k eigenpairs of the adjacency matrix

       Exact spectra are returned when ``k`` is infinite or at least the graph order.

       :param graph: undirected NetworkX graph
       :param k: number of top k eigenpairs to obtain
       :param eigvals_only: get only the eigenvalues i.e., no eigenvectors
       :param which: the type of k eigenvectors and eigenvalues to find
       :param use_gpu: use CuPy for an explicitly requested partial spectrum
       :return: the eigenpair information
       &quot;&quot;&quot;

       n = len(graph)
       if n == 0:
           if eigvals_only:
               return np.array([])
           return np.array([]), np.empty((0, 0))

       exact = np.isinf(k) or k &gt;= n
       if exact or n &lt; 100:
           A = nx.to_numpy_array(graph, nodelist=list(graph.nodes), dtype=float)
           return eigh(A, eigvals_only=eigvals_only)

       A = get_sparse_graph(graph)
       k = min(int(k), n - 1)

       if k &lt;= 0:
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

       return eigsh(A, k=k, which=which, return_eigenvectors=not eigvals_only)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/utils.py#L33">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-utils-get_laplacian_spectrum"><summary><code>get_laplacian_spectrum(graph, k=np.inf, which='SM', tol=1e-08, eigvals_only=True, use_gpu=False)</code></summary><div class="api-body"><p>Gets the bottom k eigenpairs of the Laplacian matrix

   Exact spectra are returned when k is infinite or at least the graph order.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of bottom k eigenpairs to obtain</td></tr><tr><td><code>which</code></td><td>the type of k eigenvectors and eigenvalues to find</td></tr><tr><td><code>tol</code></td><td>the precision at which to stop computing partial eigenpairs</td></tr><tr><td><code>eigvals_only</code></td><td>get only the eigenvalues i.e., no eigenvectors</td></tr><tr><td><code>use_gpu</code></td><td>retained for API compatibility; Laplacian GPU is unavailable</td></tr></tbody></table></div><p><strong>Returns</strong> the eigenpair information</p><details class="source-code"><summary>View source code</summary><pre><code>def get_laplacian_spectrum(graph, k=np.inf, which='SM', tol=1E-8, eigvals_only=True, use_gpu=False):
       &quot;&quot;&quot;
       Gets the bottom k eigenpairs of the Laplacian matrix

       Exact spectra are returned when ``k`` is infinite or at least the graph order.

       :param graph: undirected NetworkX graph
       :param k: number of bottom k eigenpairs to obtain
       :param which: the type of k eigenvectors and eigenvalues to find
       :param tol: the precision at which to stop computing partial eigenpairs
       :param eigvals_only: get only the eigenvalues i.e., no eigenvectors
       :param use_gpu: retained for API compatibility; Laplacian GPU is unavailable
       :return: the eigenpair information
       &quot;&quot;&quot;

       if use_gpu:
           print('Warning: GPU requested, but not available for Laplacian measures')

       n = len(graph)
       if n == 0:
           if eigvals_only:
               return np.array([])
           return np.array([]), np.empty((0, 0))

       exact = np.isinf(k) or k &gt;= n
       if exact or n &lt; 100:
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
       return lam[idx], vectors[:, idx]</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/utils.py#L87">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-utils-get_laplacian"><summary><code>get_laplacian(graph)</code></summary><div class="api-body"><p>Gets the Laplacian matrix in sparse CSR format</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> Scipy sparse Laplacian matrix</p><details class="source-code"><summary>View source code</summary><pre><code>def get_laplacian(graph):
       &quot;&quot;&quot;
       Gets the Laplacian matrix in sparse CSR format

       :param graph: undirected NetworkX graph
       :return: Scipy sparse Laplacian matrix
       &quot;&quot;&quot;

       A = get_sparse_graph(graph)
       degree = np.asarray(A.sum(axis=1)).flatten()
       D = sparse.diags(degree, offsets=0, shape=(len(graph), len(graph)), format='csr')

       return D - A</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/utils.py#L128">View this source on GitHub</a>.</p></div></details></div>
