#!/usr/bin/env python3
"""Summarize independent timing samples. Run from the repository root."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

root = Path('kernels/ablation/result')
for group in ('a_overall', 'b_v_access', 'c_v2a_pipeline', 'd_v2b_pipeline'):
    rows = list(csv.DictReader((root / group / 'raw.csv').open()))
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[x] for x in ('variant','batch','heads','seq_len','head_dim','causal'))
        groups[key].append(float(row['latency_ms']))
    with (root / group / 'summary.csv').open('w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['variant','batch','heads','seq_len','head_dim','causal','samples','median_ms','q1_ms','q3_ms','iqr_ms'])
        for key, values in groups.items():
            if len(values) < 2:
                continue
            q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
            out.writerow([*key, len(values), statistics.median(values), q1, q3, q3-q1])
    # Standalone SVG avoids requiring a plotting package.
    keys = list(groups)
    width, height = 1000, 100 + len(keys)*25
    maximum = max((statistics.median(v) for v in groups.values()), default=1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="10" y="25" font-family="sans-serif" font-size="16">Group {group}: median latency (ms)</text>']
    for i, key in enumerate(keys):
        value = statistics.median(groups[key]); y = 45 + i*25
        label = f'{key[0]} N={key[3]} D={key[4]} causal={key[5]}'
        parts += [f'<text x="10" y="{y+14}" font-family="sans-serif" font-size="11">{label}</text>', f'<rect x="310" y="{y}" width="{value/maximum*590:.1f}" height="16" fill="#3d6fa6"/>', f'<text x="{315+value/maximum*590:.1f}" y="{y+13}" font-family="sans-serif" font-size="11">{value:.4f}</text>']
    parts.append('</svg>')
    (root / group / 'latency.svg').write_text('\n'.join(parts))
    correctness_rows = list(csv.DictReader((root / 'correctness.csv').open()))
    wanted = {key[0] for key in keys}
    with (root / group / 'correctness.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=correctness_rows[0].keys())
        writer.writeheader()
        writer.writerows(r for r in correctness_rows if r['variant'] in wanted)
    baseline = {'a_overall':'v0', 'b_v_access':'v2a', 'c_v2a_pipeline':'v2a', 'd_v2b_pipeline':'v2b'}[group]
    with (root / group / 'speedup.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['variant','baseline','batch','heads','seq_len','head_dim','causal','median_speedup','latency_reduction_pct'])
        for key, values in groups.items():
            base = (baseline, *key[1:])
            if base not in groups: continue
            median = statistics.median(values); reference = statistics.median(groups[base])
            writer.writerow([key[0], baseline, *key[1:], reference / median, 100 * (1 - median / reference)])
