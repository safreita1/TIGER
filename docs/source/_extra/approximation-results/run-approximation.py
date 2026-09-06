import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "public-0.6.0"), str(ROOT / "experiment-deps")]
import csv
import json
import random
import platform
from time import perf_counter
import importlib.metadata
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graph_tiger.measures import run_measure

OUT = ROOT / "site" / "approximation-results"
OUT.mkdir(parents=True, exist_ok=True)
names = ["average_vertex_betweenness", "average_edge_betweenness",
         "natural_connectivity", "number_spanning_trees", "effective_resistance"]
ks = [10, 30, 60, 100, 200, 300]
rows = []
for seed in range(5):
    G = nx.powerlaw_cluster_graph(300, 3, 0.3, seed=seed)
    for name in names:
        exact = run_measure(G, name)
        for k in ks:
            random.seed(seed)
            np.random.seed(seed)
            start = perf_counter()
            estimate = run_measure(G, name, k=k)
            seconds = perf_counter() - start
            if estimate is None or not np.isfinite(estimate):
                raise RuntimeError(f"Nonfinite result: {seed}, {name}, {k}")
            rows.append(dict(seed=seed, measure=name, k=k, exact=float(exact),
                             estimate=float(estimate), absolute_error=float(abs(estimate-exact)),
                             seconds=seconds))
        print(f"Graph {seed+1}/5: {name}", flush=True)
with (OUT/"measurements.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metadata={"graph":"NetworkX powerlaw_cluster_graph","n":300,"m":3,"p":0.3,
          "seeds":list(range(5)),"k":ks,"time_unit":"seconds",
          "error":"mean of per-graph absolute errors against full calculation",
          "python":platform.python_version(),"platform":platform.platform(),
          "versions":{p:importlib.metadata.version(p) for p in ["graph-tiger","networkx","numpy","scipy","matplotlib"]},
          "threads":1,"timing":"one timed call per graph, measure, and k; machine dependent"}
(OUT/"experiment.json").write_text(json.dumps(metadata,indent=2))
summary = {name: {field: [float(np.mean([r[field] for r in rows if r["measure"] == name and r["k"] == k])) for k in ks] for field in ["absolute_error", "seconds"]} for name in names}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
import runpy
runpy.run_path(str(ROOT / "render-approximation.py"), run_name="__main__")
