"""Render CPU-versus-GPU benchmark results.

The output PDF contains a summary and one diagnostic page per robustness
measure so crossover points and numerical agreement remain visible.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def add_summary(pdf, data):
    grouped = data.groupby('measure', sort=False)
    labels = []
    speedups = []
    passed = []
    for measure, rows in grouped:
        labels.append(measure.replace('_', ' '))
        speedups.append(rows['speedup'].median())
        passed.append(rows['parity_passed'].astype(bool).all())

    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = ['#2979a8' if value else '#c84b31' for value in passed]
    positions = np.arange(len(labels))
    axis.barh(positions, speedups, color=colors)
    axis.axvline(1, color='#555555', linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel('Median CPU time / median GPU time')
    axis.set_title('GPU speedup across all benchmark cases')
    axis.invert_yaxis()
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def add_measure_page(pdf, measure, data):
    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = {
        'barabasi_albert': '#2979a8',
        'watts_strogatz': '#d57a2a',
        'erdos_renyi': '#2c8c6b'
    }

    for family, rows in data.groupby('family'):
        summary = rows.groupby('nodes', as_index=False).median(numeric_only=True)
        color = colors.get(family)
        label = family.replace('_', ' ')
        axes[0].plot(
            summary['nodes'], summary['cpu_median_seconds'],
            marker='o', color=color, linestyle='--', label='{} CPU'.format(label)
        )
        axes[0].plot(
            summary['nodes'], summary['gpu_median_seconds'],
            marker='o', color=color, label='{} GPU'.format(label)
        )
        axes[1].plot(
            summary['nodes'], summary['speedup'],
            marker='o', color=color, label=label
        )
        axes[2].plot(
            summary['nodes'], summary['relative_error'],
            marker='o', color=color, label=label
        )

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Nodes')
    axes[0].set_ylabel('Median seconds')
    axes[0].set_title('End-to-end runtime')

    axes[1].set_xscale('log')
    axes[1].axhline(1, color='#555555', linewidth=1)
    axes[1].set_xlabel('Nodes')
    axes[1].set_ylabel('CPU / GPU time')
    axes[1].set_title('Speedup')

    axes[2].set_xscale('log')
    if (data['relative_error'].replace([np.inf, -np.inf], np.nan).dropna() > 0).any():
        axes[2].set_yscale('log')
    axes[2].set_xlabel('Nodes')
    axes[2].set_ylabel('Relative error')
    axes[2].set_title('Numerical agreement')

    axes[1].legend(fontsize=8)
    figure.suptitle(measure.replace('_', ' ').title())
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def render(input_path, output_path):
    data = pd.read_csv(input_path)
    if data.empty:
        raise ValueError('benchmark result file is empty')

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        add_summary(pdf, data)
        for measure, rows in data.groupby('measure', sort=False):
            add_measure_page(pdf, measure, rows)
    print('Wrote {}'.format(output))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render TIGER CPU-versus-GPU benchmark results'
    )
    parser.add_argument('input', nargs='?', default='gpu-benchmark-results.csv')
    parser.add_argument('--output', default='gpu-benchmark-results.pdf')
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    render(arguments.input, arguments.output)
