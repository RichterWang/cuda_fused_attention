#!/usr/bin/env python3
"""Figures for the attention study. Requires matplotlib (infra_dev env).

    MPLCONFIGDIR=/tmp/mplcache /home/minstrel/miniconda3/envs/infra_dev/bin/python plot/make_figures.py

Inputs are the committed CSVs (results/raw/flash_attention/*.csv and
kernels/ablation/result/*/speedup.csv plus the V0 stage table); outputs land in
results/figures as both PNG (200 dpi) and SVG.
"""
import csv, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'results', 'raw', 'flash_attention', 'summary.csv')
V0CSV = os.path.join(ROOT, 'results', 'raw', 'v0_stage_timing.csv')
ABL = os.path.join(ROOT, 'kernels', 'ablation', 'result')
OUT = os.path.join(ROOT, 'results', 'figures')

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False,
})

PALETTE = {'v0_unfused': '#9e9e9e', 'v1_fused_row': '#e8a33d', 'v2a_fused_tiled': '#5b8fd6',
           'v2b_fused_tiled': '#2f6fb5', 'v3_tensorcore': '#c0392b'}
NICE = {'v0_unfused': 'V0 unfused', 'v1_fused_row': 'V1 row-fused', 'v2a_fused_tiled': 'V2a tiled',
        'v2b_fused_tiled': 'V2b tiled', 'v3_tensorcore': 'V3 tensor core'}

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)))
    plt.close(fig)
    return name

def load_summary():
    d = {}
    with open(RAW) as f:
        for r in csv.DictReader(f):
            if r['status'] != 'ok':
                continue
            d[(r['kernel'], int(r['batch']), int(r['heads']), int(r['seq_len']),
               int(r['head_dim']), int(r['causal']))] = dict(
                ms=float(r['median_ms']), iqr=float(r['iqr_ms']), tf=float(r['tflops']))
    return d

def load_speedups(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# --------------------------------------------------------------------------
def fig1(S):
    shapes = [((4, 8, 512, 64, 0), 'B4 N512\nD64 dense'),
              ((2, 8, 2048, 64, 0), 'B2 N2048\nD64 dense'),
              ((2, 8, 2048, 64, 1), 'B2 N2048\nD64 causal'),
              ((1, 8, 2048, 128, 1), 'B1 N2048\nD128 causal')]
    kernels = ['v0_unfused', 'v1_fused_row', 'v2a_fused_tiled', 'v2b_fused_tiled', 'v3_tensorcore']
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(shapes)); w = 0.16
    for i, k in enumerate(kernels):
        vals = [S[(k,) + sh]['ms'] for sh, _ in shapes]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=NICE[k], color=PALETTE[k],
                      edgecolor='white', linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12, ('%.1f' if v > 1 else '%.3f') % v,
                    ha='center', va='bottom', fontsize=6.5, rotation=90, color='#333')
    ax.set_yscale('log'); ax.set_ylim(0.008, 260)
    ax.set_ylabel('median latency [ms, log scale]')
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in shapes], fontsize=9)
    ax.set_title('Device-side latency by implementation (lower is better)')
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=9)
    return save(fig, 'fig1_evolution')

def fig2(S):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ns = [128, 512, 1024, 2048, 4096]
    series = [('vs V2b, D=64', 'v2b_fused_tiled', 64, '-', 'o', '#c0392b'),
              ('vs V2b, D=128', 'v2b_fused_tiled', 128, '-', 's', '#8e44ad'),
              ('vs V0, D=64', 'v0_unfused', 64, '--', 'o', '#c0392b'),
              ('vs V0, D=128', 'v0_unfused', 128, '--', 's', '#8e44ad')]
    for label, base, d, ls, mk, colour in series:
        ys = []
        for n in ns:
            v3 = S.get(('v3_tensorcore', 1, 8, n, d, 0)); bs = S.get((base, 1, 8, n, d, 0))
            ys.append(bs['ms'] / v3['ms'] if v3 and bs else np.nan)
        ax.plot(ns, ys, ls, marker=mk, ms=5, color=colour, label=label, lw=1.8)
    ax.axhline(1.0, color='#888', lw=1, ls=':')
    ax.axvline(512, color='#bbb', lw=0.8, ls='--')
    ax.text(530, 30, 'grid reaches half a wave', fontsize=8, color='#777')
    ax.set_xscale('log', base=2); ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('sequence length N'); ax.set_ylabel('speedup [x]')
    ax.set_ylim(0, 95)
    ax.set_title('V3 speedup over V2b and V0 versus sequence length  (B=1, H=8, dense)')
    ax.legend(ncol=2, loc='upper left', fontsize=9)
    return save(fig, 'fig2_speedup_vs_n')

def fig3(S):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for d, colour, mk, lbl in ((64, '#c0392b', 'o', 'D=64 (3 blocks/SM, 138 slots)'),
                               (128, '#8e44ad', 's', 'D=128 (2 blocks/SM, 92 slots)')):
        xs, ys = [], []
        for (k, b, h, n, dd, c), v in S.items():
            if k != 'v3_tensorcore' or dd != d: continue
            xs.append(((n + 63) // 64) * b * h / (46 * (3 if d == 64 else 2)))
            ys.append(v['tf'])
        ax.scatter(xs, ys, s=28, color=colour, marker=mk, alpha=0.8, edgecolor='white',
                   linewidth=0.5, label=lbl)
    ax.axvline(1.0, color='#c0392b', lw=1, ls='--')
    ax.text(1.05, 31, 'one full wave', fontsize=8, color='#c0392b')
    ax.set_xscale('log'); ax.set_xlim(0.08, 20); ax.set_ylim(0, 33)
    ax.set_xlabel('grid fill [waves = launched blocks / resident slots, log scale]')
    ax.set_ylabel('achieved throughput [TFLOPS]')
    ax.set_title('V3 throughput versus grid occupancy, all 60 benchmark shapes')
    ax.annotate('N=128, B=1: 16 blocks,\n30 of 46 SMs idle',
                xy=(0.12, 3.0), xytext=(0.16, 9.5), fontsize=8, color='#555',
                arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))
    ax.legend(loc='lower right', fontsize=9)
    return save(fig, 'fig3_grid_fill')

def fig4():
    val, runs = {}, []
    with open(V0CSV) as f:
        for r in csv.DictReader(f):
            sh = r['shape']
            if sh not in val:
                val[sh] = {}
                runs.append(sh)
            val[sh][r['metric']] = (float(r['median_ms']), float(r['iqr_ms']))
    def label(sh):
        m = re.match(r'b(\d+)h(\d+)n(\d+)d(\d+)', sh)
        return 'B%s H%s N%s D%s' % m.groups() if m else sh
    labels = [label(sh) for sh in runs]
    stage_names = ['QK', 'softmax', 'PV']
    colours = {'QK': '#c0392b', 'softmax': '#e8a33d', 'PV': '#5b8fd6'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={'width_ratios': [1.3, 1]})

    ys = np.arange(len(runs))[::-1]
    for y, sh in zip(ys, runs):
        left = 0.0
        tot = sum(val[sh][n][0] for n in stage_names)
        for name in stage_names:
            ms = val[sh][name][0]
            a.barh(y, ms, left=left, color=colours[name], height=0.5, edgecolor='white')
            pct = 100 * ms / tot
            if name == 'QK':
                a.text(left + ms / 2, y, 'QK %.0f%%' % pct, ha='center', va='center',
                       color='white', fontsize=10)
            elif name == 'PV':
                a.text(left + ms / 2, y, 'PV %.0f%%' % pct, ha='center', va='center',
                       color='white', fontsize=9)
            else:
                a.annotate('softmax %.1f%%' % pct, xy=(left + ms / 2, y - 0.26),
                           xytext=(left - 2.0, y - 0.62), ha='center', fontsize=8, color='#333',
                           arrowprops=dict(arrowstyle='-', color='#999', lw=0.7))
            left += ms
        a.text(left + 0.6, y, '%.1f ms' % left, va='center', fontsize=9)
    a.set_yticks(ys); a.set_yticklabels(labels, fontsize=9)
    a.set_xlim(0, 47); a.set_ylim(-1.0, len(runs) - 0.35)
    a.set_xlabel('cumulative time [ms]'); a.grid(axis='y', visible=False)
    a.set_title('composition: the three kernels cover the pipeline', fontsize=10)

    xs = np.arange(len(stage_names))
    ratios = [val[runs[1]][n][0] / val[runs[0]][n][0] for n in stage_names]
    b.bar(xs, ratios, color=[colours[n] for n in stage_names], width=0.55)
    b.axhline(1.0, color='#333', lw=1)
    for x, v in zip(xs, ratios):
        b.text(x, v + 0.03, '%.2fx' % v, ha='center', fontsize=9)
    b.set_xticks(xs); b.set_xticklabels(stage_names)
    b.set_ylim(0, 1.2); b.set_ylabel('stage time ratio\n(D128 run / D64 run)')
    b.set_title('equal FLOPs, half the rows', fontsize=10)
    b.text(0, 1.10, 'same FLOPs', ha='center', fontsize=8, color='#666')
    b.text(2, 1.10, 'same FLOPs', ha='center', fontsize=8, color='#666')
    b.text(1, 0.62, 'half the rows', ha='center', fontsize=8, color='#666')

    fig.suptitle('V0 stage breakdown: stages sum to the pipeline; over these two runs QK/PV scale with FLOPs and softmax with rows',
                 fontsize=11, y=1.05)
    fig.subplots_adjust(wspace=0.32)
    return save(fig, 'fig4_v0_stages')

def fig5():
    rows_b = load_speedups(os.path.join(ABL, 'b_v_access', 'speedup.csv'))
    rows_c = load_speedups(os.path.join(ABL, 'c_v2a_pipeline', 'speedup.csv'))
    rows_d = load_speedups(os.path.join(ABL, 'd_v2b_pipeline', 'speedup.csv'))
    def med(rows, pred):
        v = [float(r['median_speedup']) for r in rows if pred(r)]
        return float(np.median(v)) if v else float('nan')
    items = [
        ('B  V access', 'b01', med(rows_b, lambda r: r['variant'] == 'b01')),
        ('B  V access', 'b10', med(rows_b, lambda r: r['variant'] == 'b10')),
        ('B  V access', 'v2b', med(rows_b, lambda r: r['variant'] == 'v2b')),
        ('C  pipeline', 'D64 sync', med(rows_c, lambda r: r['variant'] == 'c1_sync' and r['head_dim'] == '64')),
        ('C  pipeline', 'D64 async', med(rows_c, lambda r: r['variant'] == 'c1_async' and r['head_dim'] == '64')),
        ('C  pipeline', 'D128 sync', med(rows_c, lambda r: r['variant'] == 'c1_sync' and r['head_dim'] == '128')),
        ('C  pipeline', 'D128 async', med(rows_c, lambda r: r['variant'] == 'c1_async' and r['head_dim'] == '128')),
        ('D  buffer', 'D64 single', med(rows_d, lambda r: r['variant'] == 'v2b_single' and r['head_dim'] == '64')),
        ('D  buffer', 'D128 single', med(rows_d, lambda r: r['variant'] == 'v2b_single' and r['head_dim'] == '128')),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(items))
    for i, (grp, name, v) in enumerate(items):
        colour = '#c0392b' if v > 1.05 else ('#9e9e9e' if v >= 0.97 else '#e8a33d')
        ax.bar(i, v - 1.0, bottom=1.0, color=colour, width=0.7)
        ax.text(i, v + (0.015 if v >= 1 else -0.03), '%.2fx' % v, ha='center',
                va='bottom' if v >= 1 else 'top', fontsize=9)
    ax.axhline(1.0, color='#333', lw=1)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n, _ in items], fontsize=8.5, rotation=20, ha='right')
    ax.set_ylabel('speedup vs group baseline [x]'); ax.set_ylim(0.88, 1.75)
    for xb, grp in ((1, 'B  V access'), (4.5, 'C  pipeline'), (7.5, 'D  buffer')):
        ax.text(xb, 1.70, grp, ha='center', fontsize=9.5, color='#555')
    ax.set_title('Controlled ablations, each relative to its own group baseline\n'
                 'single-factor controls (b01, b10, c1, v2b_single) plus the composed v2b', fontsize=10.5)
    return save(fig, 'fig5_ablation_summary')

def main():
    os.makedirs(OUT, exist_ok=True)
    S = load_summary()
    print(f'loaded {len(S)} summary rows')
    for f in (fig1(S), fig2(S), fig3(S), fig4(), fig5()):
        print(os.path.join(OUT, f + '.png'))

if __name__ == '__main__':
    main()
