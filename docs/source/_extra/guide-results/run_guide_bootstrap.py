"""Shared plotting and output helpers for the documentation experiment scripts."""
import runpy
from pathlib import Path
globals().update({k:v for k,v in runpy.run_path(str(Path(__file__).with_name('run-guide-studies.py'))).items() if not k.startswith('__')})
