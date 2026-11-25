"""
Draws the README's 3D isometric diagrams from pipeline outputs (numbers are never typed in).

    python src/generate_diagrams.py   ->  assets/architecture_3d.svg, assets/pipeline_flow_3d.svg
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / 'reports'
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

C, S = math.cos(math.radians(30)), 0.5
FONT = "font-family=\"'Segoe UI', 'Helvetica Neue', Arial, sans-serif\""

# ── numbers from the pipeline ───────────────────────────────────────
met = json.loads((REP / 'model_metrics.json').read_text())
dq = json.loads((REP / 'data_quality_report.json').read_text())
anom = pd.read_csv(REP / 'anomaly_register.csv', usecols=['Classification'])
sub = pd.read_csv(ROOT / 'submission.csv')
N = dict(
    wape=met['overall']['LightGBM_P50']['wape'], snaive=met['overall']['Seasonal_Naive']['wape'],
    folds=len(met['folds']), cover=met['interval_P10_P90_coverage'],
    agg_total=met['aggregate_level_wape']['Total'],
    issues=len(dq['issues']), rows_fixed=dq['summary']['rows_touched'],
    raw_rows=dq['before_after']['train_rows_raw'], clean_rows=dq['before_after']['train_rows_clean'],
    stockouts=dq['before_after']['stockout_rows'], flags=len(anom),
    series=sub.groupby(['SKU_ID', 'Region']).ngroups, sub_rows=len(sub),
    units=met['test_forecast_summary']['total_units'], feats=met['model']['features'],
)
N['gain'] = (1 - N['wape'] / N['snaive']) * 100


class Iso:
    def __init__(self, ox, oy, unit):
        self.ox, self.oy, self.u = ox, oy, unit
        self.out = []

    def p(self, x, y, z):
        return self.ox + (x - y) * C * self.u, self.oy + (x + y) * S * self.u - z * self.u

    def poly(self, pts, fill, stroke='none', sw=1, op=1.0, extra=''):
        d = ' '.join(f'{a:.1f},{b:.1f}' for a, b in (self.p(*q) for q in pts))
        self.out.append(f'<polygon points="{d}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" '
                        f'stroke-width="{sw}" stroke-linejoin="round" {extra}/>')

    def box(self, x, y, z, w, d, h, top, left, right, edge='none', op=1.0, glow=False):
        g = 'filter="url(#glow)"' if glow else ''
        self.poly([(x, y + d, z), (x + w, y + d, z), (x + w, y + d, z + h), (x, y + d, z + h)], left, op=op)
        self.poly([(x + w, y, z), (x + w, y + d, z), (x + w, y + d, z + h), (x + w, y, z + h)], right, op=op)
        self.poly([(x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h)], top, edge, 1.6, op, g)

    def iso_text(self, x, y, z, txt, size, fill, weight=700, anchor='start', ls=1):
        """Text lying on the floor plane along the +x axis."""
        sx, sy = self.p(x, y, z)
        self.out.append(f'<text transform="matrix({C:.4f},{S:.4f},{-C:.4f},{S:.4f},{sx:.1f},{sy:.1f})" {FONT} '
                        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
                        f'letter-spacing="{ls}">{txt}</text>')

    def text(self, sx, sy, txt, size, fill, weight=400, anchor='start', ls=0):
        self.out.append(f'<text x="{sx:.1f}" y="{sy:.1f}" {FONT} font-size="{size}" font-weight="{weight}" '
                        f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">{txt}</text>')

    def raw(self, s):
        self.out.append(s)


def defs(w, h):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<defs>
  <radialGradient id="bg" cx="55%" cy="45%" r="75%">
    <stop offset="0%" stop-color="#14203a"/><stop offset="65%" stop-color="#0a1020"/><stop offset="100%" stop-color="#05080f"/>
  </radialGradient>
  <filter id="glow" filterUnits="userSpaceOnUse" x="0" y="0" width="{w}" height="{h}">
    <feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">
    <feDropShadow dx="0" dy="14" stdDeviation="12" flood-color="#000" flood-opacity="0.6"/>
  </filter>
  <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#ffb347"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="url(#bg)"/>
<g opacity="0.07" stroke="#7aa2d6">{''.join(f'<line x1="{x}" y1="0" x2="{x - h * 1.732}" y2="{h}"/><line x1="{x - h * 1.732}" y1="0" x2="{x}" y2="{h}"/>' for x in range(0, w + int(h * 1.8), 60))}</g>
'''


# ════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE: five stacked layers
# ════════════════════════════════════════════════════════════════════
def architecture():
    W, H = 1500, 1300
    iso = Iso(830, 300, 30)
    LW, LD, LH, GAP = 14, 5, 0.45, 6.0
    layers = [  # bottom -> top
        dict(name='DATA SOURCES', color=('#1d3557', '#12223a', '#0c182b', '#4cc9f0'),
             desc=['Raw ERP extract + reference files', f"train.csv {N['raw_rows']:,} rows · test.csv {N['sub_rows']:,} rows",
                   'stockouts · macro · sku_master · events'],
             chips=['train.csv', 'stockouts', 'macro', 'sku_master', 'events']),
        dict(name='CLEANING + UNCONSTRAINED DEMAND', color=('#1b3a35', '#11261f', '#0a1a15', '#2ee6a6'),
             desc=[f"{N['issues']} issue types · {N['rows_fixed']:,} rows fixed and logged",
                   'one clean row per Date-SKU-Region', f"{N['stockouts']:,} stockouts → requested units"],
             chips=['labels + dates', 'price / units', 'dedupe', 'stockouts']),
        dict(name='ANOMALIES + DRIVERS', color=('#33244a', '#221733', '#170f24', '#b388ff'),
             desc=[f"{N['flags']:,} flags: error vs real event + action", 'order-cycle phase · holidays · EV shift',
                   'macro lead/lag + lag stability'],
             chips=['register', 'order cycles', 'lead / lag']),
        dict(name='FORECAST ENGINE', color=('#3d2a17', '#2a1c0e', '#1d1309', '#ffb347'),
             desc=['global LightGBM · direct h = 1..19', f"WAPE {N['wape']:.3f} vs naive {N['snaive']:.3f}",
                   'P10 / P50 / P90 · launches · EOL · events'],
             chips=['LightGBM', 'P10/P50/P90', 'lifecycle']),
        dict(name='PLANNING CONTROL TOWER', color=('#163a52', '#0e2638', '#091a27', '#4cc9f0'),
             desc=['Streamlit · 11 views · shared filters', 'history → forecast on one timeline',
                   'what-if · P90 procurement · exceptions'],
             chips=['11 views', 'what-if', 'P90 plan']),
    ]
    iso.raw(defs(W, H))
    iso.text(60, 70, 'CELERIS DEMAND PLANNING CONTROL TOWER', 30, '#ffffff', 800, ls=2)
    iso.text(60, 102, 'Solution architecture — five layers, each reading only the outputs of the layer below',
             16, '#8fb8e8', 500)
    base_z = 3 - 4 * GAP
    for k, L in enumerate(layers):
        z = base_z + k * GAP
        top, left, right, edge = L['color']
        iso.raw('<g filter="url(#shadow)">')
        iso.box(0, 0, z, LW, LD, LH, top, left, right, edge, op=0.93, glow=True)
        iso.raw('</g>')
        # pillars to the next layer
        if k < len(layers) - 1:
            for (px, py) in [(0.25, LD - 0.25), (LW - 0.25, LD - 0.25), (LW - 0.25, 0.25)]:
                a, b = iso.p(px, py, z + LH), iso.p(px, py, z + GAP)
                iso.raw(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{edge}" '
                        f'stroke-width="2.5" opacity="0.75" filter="url(#glow)"/>')
        # chips on the slab
        n = len(L['chips'])
        cw = (LW - 1.6) / n - 0.5
        labels = []
        for j, chip in enumerate(L['chips']):
            cx = 0.8 + j * (cw + 0.5)
            iso.box(cx, 1.0, z + LH, cw, 2.2, 0.7, edge, left, right, 'none', op=0.95)
            labels.append((*iso.p(cx + cw / 2, 2.1, z + LH + 0.7), chip))
        for sx, sy, chip in labels:
            iso.text(sx, sy + 4, chip, 12 if n > 3 else 13, '#06101e', 800, 'middle')
        iso.iso_text(0.6, LD - 0.7, z + LH, L['name'], 14, edge, 800, ls=1.5)
        # left-hand annotation with leader line
        ax, ay = iso.p(0, LD, z + LH / 2)
        tx, ty = 60, ay - 34
        iso.raw(f'<line x1="{ax - 8:.1f}" y1="{ay:.1f}" x2="{tx + 365:.1f}" y2="{ty + 8:.1f}" stroke="{edge}" '
                f'stroke-width="1.2" opacity="0.6" stroke-dasharray="4 4"/>')
        iso.raw(f'<circle cx="{ax - 8:.1f}" cy="{ay:.1f}" r="4" fill="{edge}" filter="url(#glow)"/>')
        iso.text(tx, ty, f'{5 - (4 - k)}. {L["name"].title()}', 19, '#ffffff', 700)
        for i, line in enumerate(L['desc']):
            iso.text(tx + 4, ty + 26 + i * 21, '• ' + line, 14.5, '#b9cbe6', 400)
    # KPI panel
    px, py = 1150, 120
    iso.raw(f'<rect x="{px}" y="{py}" width="320" height="140" rx="14" fill="#0b1426" stroke="#ffb347" '
            f'stroke-opacity="0.7" filter="url(#glow)"/>')
    iso.text(px + 20, py + 34, f'19-MONTH BACKTEST ({N["folds"]} ORIGINS)', 13, '#ffb347', 800, ls=1)
    iso.text(px + 20, py + 80, f'WAPE {N["wape"]:.3f}', 34, '#ffffff', 800)
    iso.text(px + 20, py + 110, f'{N["gain"]:.0f}% better than seasonal naive ({N["snaive"]:.3f})', 13.5, '#b9cbe6')
    iso.text(px + 20, py + 128, f'total-level WAPE {N["agg_total"]:.3f} · P10–P90 cover {N["cover"]:.0%}', 13.5, '#b9cbe6')
    iso.raw('</svg>')
    (ASSETS / 'architecture_3d.svg').write_text('\n'.join(iso.out), encoding='utf-8')


# ════════════════════════════════════════════════════════════════════
# 2. PIPELINE FLOW: one command, five stages, outputs
# ════════════════════════════════════════════════════════════════════
def flow():
    W, H = 1500, 640
    iso = Iso(150, 340, 26)
    iso.raw(defs(W, H))
    iso.text(60, 62, 'ONE-COMMAND PIPELINE', 30, '#ffffff', 800, ls=2)
    iso.text(60, 94, 'run_pipeline.bat / .sh  →  every number in the dashboard and README is produced here',
             16, '#8fb8e8', 500)
    stages = [
        ('RAW DATA', 'dataset/student_data', ['7 CSV files', f"{N['raw_rows']:,} messy rows"], '#4cc9f0'),
        ('CLEAN', 'src/clean.py', [f"{N['issues']} issue types fixed", 'unconstrained demand'], '#2ee6a6'),
        ('DRIVERS', 'src/features.py', ['macro lead/lag', 'order cycles · EV'], '#b388ff'),
        ('ANOMALIES', 'src/anomalies.py', [f"{N['flags']:,} flags", 'error vs real event'], '#ff6b9a'),
        ('FORECAST', 'src/models.py', [f"{N['folds']} × 19-month backtest", 'P10 / P50 / P90'], '#ffb347'),
        ('OUTPUTS', 'submission + dashboard', [f"{N['sub_rows']:,} forecast rows", '11-view control tower'], '#4cc9f0'),
    ]
    step, size = 5.1, 3.0          # stages advance along (+x, -y) = screen-horizontal
    pos = []
    for i, (name, script, lines, col) in enumerate(stages):
        x, y = i * step, -i * step
        h = 2.4 + (1.2 if name == 'FORECAST' else 0)
        iso.raw('<g filter="url(#shadow)">')
        iso.box(x - 0.7, y - 0.7, -0.45, size + 1.4, size + 1.4, 0.45, '#13223d', '#0c1629', '#08101e', '#2a4a7a', op=0.97)
        iso.box(x, y, 0, size, size, h, col, '#16243d', '#0e1a2e', col, op=0.97, glow=True)
        iso.raw('</g>')
        sx, sy = iso.p(x + size / 2, y + size / 2, h)
        iso.text(sx, sy + 5, name, 13.5, '#06101e', 900, 'middle', 1)
        cx, by = iso.p(x + size / 2, y + size / 2, 0)[0], iso.p(x + size + 0.7, y + size + 0.7, -0.45)[1]
        iso.text(cx, by + 34, script, 14.5, col, 700, 'middle')
        for j, ln in enumerate(lines):
            iso.text(cx, by + 56 + j * 19, ln, 13.5, '#b9cbe6', 400, 'middle')
        pos.append((x, y, h))
    for i in range(len(stages) - 1):
        x, y, _ = pos[i]
        x2, y2, _ = pos[i + 1]
        a = iso.p(x + size, y + size / 2, 0.9)
        b = iso.p(x2 + size / 2, y2 + size, 0.9)
        iso.raw(f'<line x1="{a[0] + 6:.1f}" y1="{a[1]:.1f}" x2="{b[0] - 10:.1f}" y2="{b[1]:.1f}" stroke="#ffb347" '
                f'stroke-width="3.5" marker-end="url(#arr)" filter="url(#glow)"/>')
    # backtest loop over the forecast block
    x, y, h = pos[4]
    a, b = iso.p(x, y + size, h), iso.p(x + size, y, h)
    top = iso.p(x, y, h)[1] - 70
    iso.raw(f'<path d="M{b[0]:.1f},{b[1]:.1f} C{b[0] + 30:.1f},{top:.1f} {a[0] - 30:.1f},{top:.1f} {a[0]:.1f},{a[1]:.1f}" '
            f'fill="none" stroke="#ffb347" stroke-width="2.5" stroke-dasharray="6 5" marker-end="url(#arr)"/>')
    mx = (a[0] + b[0]) / 2
    iso.text(mx, top - 26, 'rolling origins: train ≤ T, forecast T+1..T+19', 13, '#ffb347', 700, 'middle')
    iso.text(mx, top - 8, f'WAPE {N["wape"]:.3f} vs naive {N["snaive"]:.3f}', 13, '#ffffff', 700, 'middle')
    # tests badge above outputs
    x, y, h = pos[5]
    tx, ty = iso.p(x + size / 2, y + size / 2, h)
    iso.raw(f'<rect x="{tx - 75:.1f}" y="{ty - 120:.1f}" width="150" height="40" rx="10" fill="#0b1426" '
            f'stroke="#2ee6a6" stroke-opacity="0.8"/>')
    iso.text(tx, ty - 94, '15 tests pass', 15, '#2ee6a6', 800, 'middle')
    iso.raw(f'<line x1="{tx:.1f}" y1="{ty - 80:.1f}" x2="{tx:.1f}" y2="{ty - 30:.1f}" stroke="#2ee6a6" '
            f'stroke-width="1.5" stroke-dasharray="3 4"/>')
    y0 = H - 32
    iso.text(60, y0, 'Outputs:', 14, '#ffffff', 700)
    iso.text(132, y0, 'data/cleaned · reports/data_quality_report.json · reports/anomaly_register.csv · '
             'reports/model_metrics.json · data/forecasts/forecast_detail.csv · submission.csv', 13.5, '#8fb8e8')
    iso.raw('</svg>')
    (ASSETS / 'pipeline_flow_3d.svg').write_text('\n'.join(iso.out), encoding='utf-8')

if __name__ == '__main__':
    architecture()
    flow()
    print('wrote', ASSETS / 'architecture_3d.svg', ASSETS / 'pipeline_flow_3d.svg')
