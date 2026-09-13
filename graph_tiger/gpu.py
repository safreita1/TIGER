"""GPU installation and runtime diagnostics for TIGER."""

import argparse
import json

from graph_tiger.utils import (
    gpu_status,
    networkx_gpu_status,
    system_gpu_status
)


def gpu_report():
    """Return NVIDIA hardware, CuPy, and nx-cugraph readiness information."""

    return {
        'system': system_gpu_status(),
        'cupy': gpu_status(),
        'networkx': networkx_gpu_status()
    }


def main(args=None):
    """Print a GPU readiness report and return zero when CuPy can execute."""

    parser = argparse.ArgumentParser(
        description='Check NVIDIA, CuPy, and nx-cugraph readiness for TIGER.'
    )
    parser.add_argument('--json', action='store_true', dest='as_json')
    options = parser.parse_args(args)
    report = gpu_report()

    if options.as_json:
        print(json.dumps(report, indent=2))
    else:
        system = report['system']
        print('NVIDIA hardware: {}'.format(system['reason']))
        for device in system['devices']:
            print('  {name}; driver {driver_version}; {memory_mib} MiB'.format(
                **device
            ))
        print('CuPy runtime: {}'.format(report['cupy']['reason']))
        print('NetworkX GPU backend: {}'.format(report['networkx']['reason']))
        if system['available'] and not report['cupy']['available']:
            print('Install one matching extra: graph-tiger[gpu-cu12] or '
                  'graph-tiger[gpu-cu13].')

    return 0 if report['cupy']['available'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
