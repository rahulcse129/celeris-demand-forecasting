import os
from pathlib import Path

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 900" width="100%" height="100%">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGrad" cx="50%" cy="45%" r="70%">
      <stop offset="0%" stop-color="#121826" />
      <stop offset="60%" stop-color="#080c14" />
      <stop offset="100%" stop-color="#030509" />
    </radialGradient>

    <!-- Platform Gradients -->
    <linearGradient id="p5Top" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e3a5f" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#0f2038" stop-opacity="0.8" />
    </linearGradient>
    <linearGradient id="p4Top" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#163832" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#0b241f" stop-opacity="0.8" />
    </linearGradient>
    <linearGradient id="p3Top" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3d2a45" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#24152a" stop-opacity="0.8" />
    </linearGradient>
    <linearGradient id="p2Top" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3d291d" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#26160d" stop-opacity="0.8" />
    </linearGradient>
    <linearGradient id="p1Top" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1a2636" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#0d1522" stop-opacity="0.8" />
    </linearGradient>

    <!-- Glow & Borders -->
    <linearGradient id="cyanBorder" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00f0ff" />
      <stop offset="50%" stop-color="#7000ff" />
      <stop offset="100%" stop-color="#00f0ff" />
    </linearGradient>
    <linearGradient id="greenBorder" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00ff88" />
      <stop offset="100%" stop-color="#00a86b" />
    </linearGradient>
    <linearGradient id="orangeBorder" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ff9900" />
      <stop offset="100%" stop-color="#ff4400" />
    </linearGradient>

    <!-- Drop Shadow Filter -->
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="18" stdDeviation="15" flood-color="#000000" flood-opacity="0.75" />
    </filter>
  </defs>

  <!-- Background -->
  <rect width="1200" height="900" fill="url(#bgGrad)" />

  <!-- Background Tech Grid Lines -->
  <g opacity="0.12" stroke="#4a6b94" stroke-width="1">
    <line x1="100" y1="200" x2="1100" y2="700" />
    <line x1="1100" y1="200" x2="100" y2="700" />
    <circle cx="600" cy="450" r="350" fill="none" stroke-dasharray="8 8" />
    <circle cx="600" cy="450" r="220" fill="none" stroke-dasharray="4 4" />
  </g>

  <!-- Header Branding -->
  <text x="600" y="55" font-family="'Segoe UI', Roboto, sans-serif" font-size="28" font-weight="900" fill="#ffffff" text-anchor="middle" letter-spacing="3">
    CELERIS DEMAND PLANNING CONTROL TOWER
  </text>
  <text x="600" y="85" font-family="'Segoe UI', Roboto, sans-serif" font-size="14" font-weight="600" fill="#00f0ff" text-anchor="middle" letter-spacing="4">
    3D ISOMETRIC ENTERPRISE ML ARCHITECTURE
  </text>

  <!-- ==================== LAYER 1: DATA INGESTION & HYGIENE (Y-Base: 710) ==================== -->
  <g transform="translate(0, 80)" filter="url(#softShadow)">
    <!-- Isometric Slab 1 -->
    <!-- Top Face -->
    <polygon points="600,640 920,730 600,820 280,730" fill="url(#p1Top)" stroke="#385170" stroke-width="2" />
    <!-- Left Face -->
    <polygon points="280,730 600,820 600,845 280,755" fill="#090e17" stroke="#385170" stroke-width="1.5" />
    <!-- Right Face -->
    <polygon points="600,820 920,730 920,755 600,845" fill="#131c2b" stroke="#385170" stroke-width="1.5" />

    <!-- Isometric Data Blocks on Layer 1 -->
    <g transform="translate(420, 700)">
      <polygon points="0,0 45,15 0,30 -45,15" fill="#20334d" stroke="#00f0ff" stroke-width="1.5" />
      <polygon points="-45,15 0,30 0,55 -45,40" fill="#101c2e" />
      <polygon points="0,30 45,15 45,40 0,55" fill="#18273d" />
      <text x="0" y="20" font-family="sans-serif" font-size="9" fill="#00f0ff" text-anchor="middle" font-weight="bold">ERP RAW</text>
    </g>
    <g transform="translate(600, 745)">
      <polygon points="0,0 45,15 0,30 -45,15" fill="#20334d" stroke="#00ff88" stroke-width="1.5" />
      <polygon points="-45,15 0,30 0,55 -45,40" fill="#101c2e" />
      <polygon points="0,30 45,15 45,40 0,55" fill="#18273d" />
      <text x="0" y="20" font-family="sans-serif" font-size="8" fill="#00ff88" text-anchor="middle" font-weight="bold">STOCKOUTS</text>
    </g>
    <g transform="translate(770, 700)">
      <polygon points="0,0 45,15 0,30 -45,15" fill="#20334d" stroke="#ff9900" stroke-width="1.5" />
      <polygon points="-45,15 0,30 0,55 -45,40" fill="#101c2e" />
      <polygon points="0,30 45,15 45,40 0,55" fill="#18273d" />
      <text x="0" y="20" font-family="sans-serif" font-size="9" fill="#ffaa00" text-anchor="middle" font-weight="bold">MACRO</text>
    </g>

    <!-- Layer 1 Annotation -->
    <line x1="280" y1="730" x2="160" y2="730" stroke="#00f0ff" stroke-width="1.5" stroke-dasharray="4 4" />
    <circle cx="160" cy="730" r="4" fill="#00f0ff" />
    <text x="145" y="725" font-family="sans-serif" font-size="13" font-weight="bold" fill="#00f0ff" text-anchor="end">LAYER 1: DATA INGESTION &amp; HYGIENE</text>
    <text x="145" y="745" font-family="sans-serif" font-size="11" fill="#99bbdd" text-anchor="end">29 Regions Reconciled | 220 Typo Normalization | Stockout Demand Restored</text>
  </g>

  <!-- Flow Connectors L1 -> L2 -->
  <g stroke="#00f0ff" stroke-width="2" opacity="0.6" stroke-dasharray="6 4">
    <line x1="450" y1="710" x2="450" y2="580" />
    <line x1="600" y1="755" x2="600" y2="625" />
    <line x1="750" y1="710" x2="750" y2="580" />
  </g>

  <!-- ==================== LAYER 2: ANOMALY & BUSINESS SIGNALS (Y-Base: 550) ==================== -->
  <g transform="translate(0, -60)" filter="url(#softShadow)">
    <polygon points="600,500 900,580 600,660 300,580" fill="url(#p2Top)" stroke="#ff7700" stroke-width="2" />
    <polygon points="300,580 600,660 600,680 300,600" fill="#140a05" stroke="#ff7700" stroke-width="1.5" />
    <polygon points="600,660 900,580 900,600 600,680" fill="#241309" stroke="#ff7700" stroke-width="1.5" />

    <!-- Isometric Badges on Layer 2 -->
    <g transform="translate(460, 560)">
      <circle cx="0" cy="0" r="18" fill="#ff4400" opacity="0.3" filter="url(#glow)" />
      <polygon points="0,-12 18,0 0,12 -18,0" fill="#ff5500" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="4" font-family="sans-serif" font-size="9" fill="#ffffff" text-anchor="middle" font-weight="bold">IQR</text>
    </g>
    <g transform="translate(600, 595)">
      <circle cx="0" cy="0" r="18" fill="#ffaa00" opacity="0.3" filter="url(#glow)" />
      <polygon points="0,-12 18,0 0,12 -18,0" fill="#ffaa00" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="4" font-family="sans-serif" font-size="8" fill="#000000" text-anchor="middle" font-weight="900">PRE-BUY</text>
    </g>
    <g transform="translate(740, 560)">
      <circle cx="0" cy="0" r="18" fill="#ff2255" opacity="0.3" filter="url(#glow)" />
      <polygon points="0,-12 18,0 0,12 -18,0" fill="#ff2255" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="4" font-family="sans-serif" font-size="8" fill="#ffffff" text-anchor="middle" font-weight="bold">SHOCKS</text>
    </g>

    <!-- Layer 2 Annotation -->
    <line x1="900" y1="580" x2="1030" y2="580" stroke="#ff9900" stroke-width="1.5" stroke-dasharray="4 4" />
    <circle cx="1030" cy="580" r="4" fill="#ff9900" />
    <text x="1045" y="575" font-family="sans-serif" font-size="13" font-weight="bold" fill="#ff9900" text-anchor="start">LAYER 2: ANOMALY MONITOR &amp; REGISTER</text>
    <text x="1045" y="595" font-family="sans-serif" font-size="11" fill="#ddbb99" text-anchor="start">7,125 Classified Events | Pre-Buy Spikes | COVID/Macro Shocks</text>
  </g>

  <!-- Flow Connectors L2 -> L3 -->
  <g stroke="#ff9900" stroke-width="2" opacity="0.6" stroke-dasharray="6 4">
    <line x1="480" y1="520" x2="480" y2="400" />
    <line x1="600" y1="550" x2="600" y2="430" />
    <line x1="720" y1="520" x2="720" y2="400" />
  </g>

  <!-- ==================== LAYER 3: FEATURE SYNTHESIS ENGINE (Y-Base: 380) ==================== -->
  <g transform="translate(0, -190)" filter="url(#softShadow)">
    <polygon points="600,370 880,445 600,520 320,445" fill="url(#p3Top)" stroke="#a855f7" stroke-width="2" />
    <polygon points="320,445 600,520 600,538 320,463" fill="#13081a" stroke="#a855f7" stroke-width="1.5" />
    <polygon points="600,520 880,445 880,463 600,538" fill="#200d2c" stroke="#a855f7" stroke-width="1.5" />

    <!-- Feature Nodes -->
    <g transform="translate(450, 430)">
      <polygon points="0,-10 25,0 0,10 -25,0" fill="#9333ea" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="3" font-family="sans-serif" font-size="8" fill="#ffffff" text-anchor="middle" font-weight="bold">CROSTON-SBA</text>
    </g>
    <g transform="translate(600, 465)">
      <polygon points="0,-10 25,0 0,10 -25,0" fill="#c084fc" stroke="#000000" stroke-width="1" />
      <text x="0" y="3" font-family="sans-serif" font-size="8" fill="#000000" text-anchor="middle" font-weight="bold">EV DYNAMICS</text>
    </g>
    <g transform="translate(750, 430)">
      <polygon points="0,-10 25,0 0,10 -25,0" fill="#9333ea" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="3" font-family="sans-serif" font-size="8" fill="#ffffff" text-anchor="middle" font-weight="bold">REPLACES_SKU</text>
    </g>

    <!-- Layer 3 Annotation -->
    <line x1="320" y1="445" x2="160" y2="445" stroke="#a855f7" stroke-width="1.5" stroke-dasharray="4 4" />
    <circle cx="160" cy="445" r="4" fill="#a855f7" />
    <text x="145" y="440" font-family="sans-serif" font-size="13" font-weight="bold" fill="#c084fc" text-anchor="end">LAYER 3: FEATURE SYNTHESIS ENGINE</text>
    <text x="145" y="460" font-family="sans-serif" font-size="11" fill="#ddbbff" text-anchor="end">75 Attributes | ADI x CV2 Classification | NPI Cannibalization</text>
  </g>

  <!-- Flow Connectors L3 -> L4 -->
  <g stroke="#00ff88" stroke-width="2" opacity="0.7" stroke-dasharray="6 4">
    <line x1="500" y1="350" x2="500" y2="240" />
    <line x1="600" y1="375" x2="600" y2="265" />
    <line x1="700" y1="350" x2="700" y2="240" />
  </g>

  <!-- ==================== LAYER 4: ENSEMBLE FORECASTING (Y-Base: 230) ==================== -->
  <g transform="translate(0, -320)" filter="url(#softShadow)">
    <polygon points="600,240 860,310 600,380 340,310" fill="url(#p4Top)" stroke="#00ff88" stroke-width="2" />
    <polygon points="340,310 600,380 600,398 340,328" fill="#061713" stroke="#00ff88" stroke-width="1.5" />
    <polygon points="600,380 860,310 860,328 600,398" fill="#0c2720" stroke="#00ff88" stroke-width="1.5" />

    <!-- Dual Booster Icons -->
    <g transform="translate(480, 300)">
      <polygon points="0,-14 30,0 0,14 -30,0" fill="#00b462" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="4" font-family="sans-serif" font-size="9" fill="#ffffff" text-anchor="middle" font-weight="bold">LightGBM (55%)</text>
    </g>
    <g transform="translate(720, 300)">
      <polygon points="0,-14 30,0 0,14 -30,0" fill="#10b981" stroke="#ffffff" stroke-width="1" />
      <text x="0" y="4" font-family="sans-serif" font-size="9" fill="#ffffff" text-anchor="middle" font-weight="bold">CatBoost (45%)</text>
    </g>
    <g transform="translate(600, 335)">
      <polygon points="0,-12 35,0 0,12 -35,0" fill="#047857" stroke="#00ff88" stroke-width="1.5" />
      <text x="0" y="4" font-family="sans-serif" font-size="9" fill="#00ff88" text-anchor="middle" font-weight="bold">MinT Reconciliation</text>
    </g>

    <!-- Layer 4 Annotation -->
    <line x1="860" y1="310" x2="1030" y2="310" stroke="#00ff88" stroke-width="1.5" stroke-dasharray="4 4" />
    <circle cx="1030" cy="310" r="4" fill="#00ff88" />
    <text x="1045" y="305" font-family="sans-serif" font-size="13" font-weight="bold" fill="#00ff88" text-anchor="start">LAYER 4: DUAL-BOOSTER ENSEMBLE</text>
    <text x="1045" y="325" font-family="sans-serif" font-size="11" fill="#bbffdd" text-anchor="start">WAPE: 0.3058 (26.5% Gain) | Bias: -0.0130 | P10-P50-P90</text>
  </g>

  <!-- Flow Connectors L4 -> L5 -->
  <g stroke="#00f0ff" stroke-width="2.5" opacity="0.8" stroke-dasharray="6 4">
    <line x1="520" y1="210" x2="520" y2="105" />
    <line x1="600" y1="230" x2="600" y2="125" />
    <line x1="680" y1="210" x2="680" y2="105" />
  </g>

  <!-- ==================== LAYER 5: EXECUTIVE CONTROL TOWER DASHBOARD ==================== -->
  <g transform="translate(0, -450)" filter="url(#softShadow)">
    <polygon points="600,110 840,175 600,240 360,175" fill="url(#p5Top)" stroke="#00f0ff" stroke-width="2.5" />
    <polygon points="360,175 600,240 600,258 360,193" fill="#061220" stroke="#00f0ff" stroke-width="1.5" />
    <polygon points="600,240 840,175 840,193 600,258" fill="#0f2238" stroke="#00f0ff" stroke-width="1.5" />

    <!-- Control Tower Hologram Screens (Floating) -->
    <g transform="translate(600, 155)">
      <!-- Central holographic dashboard screen -->
      <polygon points="-80,-40 80,-40 80,10 -80,10" fill="#0284c7" opacity="0.25" stroke="#00f0ff" stroke-width="1.5" />
      <line x1="-70" y1="-10" x2="70" y2="-10" stroke="#00f0ff" stroke-width="1" opacity="0.6" />
      <polyline points="-60,0 -30,-25 0,-15 30,-32 60,-5" fill="none" stroke="#ffaa00" stroke-width="2" />
      <text x="0" y="-28" font-family="sans-serif" font-size="8" fill="#ffffff" text-anchor="middle" font-weight="bold">19-MO TRAJECTORY</text>
    </g>

    <g transform="translate(460, 185)">
      <polygon points="-35,-20 35,-20 35,5 -35,5" fill="#0f766e" opacity="0.3" stroke="#14b8a6" stroke-width="1" />
      <text x="0" y="-8" font-family="sans-serif" font-size="7" fill="#14b8a6" text-anchor="middle" font-weight="bold">WHAT-IF</text>
    </g>
    <g transform="translate(740, 185)">
      <polygon points="-35,-20 35,-20 35,5 -35,5" fill="#b45309" opacity="0.3" stroke="#f59e0b" stroke-width="1" />
      <text x="0" y="-8" font-family="sans-serif" font-size="7" fill="#f59e0b" text-anchor="middle" font-weight="bold">P90 BUFFER</text>
    </g>

    <!-- Layer 5 Annotation -->
    <line x1="360" y1="175" x2="160" y2="175" stroke="#00f0ff" stroke-width="1.5" stroke-dasharray="4 4" />
    <circle cx="160" cy="175" r="4" fill="#00f0ff" />
    <text x="145" y="170" font-family="sans-serif" font-size="13" font-weight="bold" fill="#00f0ff" text-anchor="end">LAYER 5: CONTROL TOWER DASHBOARD</text>
    <text x="145" y="190" font-family="sans-serif" font-size="11" fill="#cceeff" text-anchor="end">10 Interactive Views | Live Simulation Sliders | 1-Click Export</text>
  </g>

  <!-- Footer Banner -->
  <rect x="250" y="855" width="700" height="32" rx="16" fill="#111827" stroke="#374151" stroke-width="1" />
  <text x="600" y="876" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" fill="#9ca3af" text-anchor="middle" font-weight="600">
    MKS AI &amp; Data Hackathon 2026 • Celeris Automobili Industrial • 58,219 Predictions Generated
  </text>
</svg>
"""

out_svg = Path("dashboard/assets/architecture_3d_isometric.svg")
out_svg.parent.mkdir(parents=True, exist_ok=True)
with open(out_svg, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"Isometric SVG Architecture generated at: {out_svg}")
