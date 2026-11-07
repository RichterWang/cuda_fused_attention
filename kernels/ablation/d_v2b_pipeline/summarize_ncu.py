#!/usr/bin/env python3
import csv
from pathlib import Path

folder = Path('kernels/ablation/result/d_v2b_pipeline')
metrics = ['launch__registers_per_thread', 'launch__shared_mem_per_block_dynamic',
           'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_registers',
           'sm__warps_active.avg.pct_of_peak_sustained_active']
with (folder / 'ncu_summary.csv').open('w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['report', *metrics])
    for path in sorted((folder / 'ncu').glob('*.csv')):
        with path.open() as f:
            rows = list(csv.reader(f))
        if len(rows) < 3:
            continue
        header, units, data = rows[0], rows[1], rows[-1]
        writer.writerow([path.stem, *[f'{data[header.index(m)]} {units[header.index(m)]}'.strip() if m in header else '' for m in metrics]])
