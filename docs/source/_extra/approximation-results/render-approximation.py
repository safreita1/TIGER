"""Render approximation plots from saved measurements without rerunning timings."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiment-deps"))
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = ROOT / "site" / "approximation-results"
with (OUT / "measurements.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
names = ["average_vertex_betweenness", "average_edge_betweenness",
         "natural_connectivity", "number_spanning_trees", "effective_resistance"]
titles = ["Node betweenness", "Edge betweenness", "Natural connectivity",
          "Spanning trees", "Effective resistance"]
ks = [10, 30, 60, 100, 200, 300]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#879b98", "text.color": "#18343b",
                     "axes.labelcolor": "#18343b", "xtick.color": "#40575d",
                     "ytick.color": "#40575d", "svg.fonttype": "path",
                     "savefig.facecolor": "white"})
fig, axes = plt.subplots(5, 2, figsize=(12, 20))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.04, top=0.94, hspace=0.95, wspace=0.25)
for idx, (name, title) in enumerate(zip(names, titles)):
    for col, (field, label, color) in enumerate([
        ("absolute_error", "Mean absolute error", "#087f73"),
        ("seconds", "Mean runtime (seconds)", "#3979aa")]):
        values = [np.mean([float(r[field]) for r in rows
                           if r["measure"] == name and int(r["k"]) == k]) for k in ks]
        def draw(ax):
            ax.plot(ks, values, color=color, marker="o", markersize=5, lw=2)
            ax.set_xlabel("k", fontsize=13)
            ax.set_ylabel(label, rotation=0, ha="left", va="bottom", fontsize=13)
            ax.yaxis.set_label_coords(0, 1.15)
            ax.text(0, 1.39, title, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left")
            ax.set_xticks([10, 100, 200, 300])
            ax.grid(axis="y", color="#e5ece9", lw=0.7)
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize(11)
            ax.margins(x=0.04)
        draw(axes[idx, col])
        panel, ax = plt.subplots(figsize=(5.5, 3.7))
        panel.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.67)
        draw(ax)
        panel.canvas.draw()
        renderer = panel.canvas.get_renderer()
        for artist in [ax.yaxis.label, ax.xaxis.label, ax.yaxis.get_offset_text(), *ax.texts]:
            bounds = artist.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > panel.bbox.width or bounds.y1 > panel.bbox.height:
                raise RuntimeError(f"Clipped plot label: {name}, {field}")
        panel.savefig(OUT / f"{name}-{field}.svg")
        plt.close(panel)
fig.savefig(OUT / "approximation-comparison.png", dpi=150)
fig.savefig(OUT / "approximation-comparison.pdf")
plt.close(fig)
print("Rendered 10 panels with horizontal y-axis titles; all label bounds passed.")
