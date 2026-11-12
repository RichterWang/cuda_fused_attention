#!/usr/bin/env python3
"""Generate the report figures (results/figures/*.png and *.svg).

The implementation lives in plot/make_figures.py; this wrapper exists because the
course submission layout asks for scripts/plot_results.py.

    MPLCONFIGDIR=build/.mplcache /path/to/python scripts/plot_results.py

Needs matplotlib (available in the infra_dev conda environment on the
measurement host). Reads results/raw/flash_attention/summary.csv,
results/raw/v0_stage_timing.csv and the ablation speedup CSVs; writes into
results/figures/.
"""
import os
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(ROOT, 'plot', 'make_figures.py'), run_name='__main__')
