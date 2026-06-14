"""Generate animated SVGs that *explain* strands-diffusers — reproducibly.

These are pure CSS-animated SVGs (no JS, no SMIL): tiny, crisp, loop forever,
honor `prefers-reduced-motion`, and adapt to light/dark. They illustrate the
four ideas the docs lean on:

  1. denoise.svg   — what diffusion *is*: noise resolves into a sample
  2. hub.svg       — one tool, every modality (inputs in → outputs out)
  3. wfm.svg       — the hero: a robot world model (frame+prompt → video+actions)
  4. discover.svg  — the catalog is read from diffusers at runtime

Run:  python examples/gen_animations.py
Out:  docs/assets/anim/*.svg
"""
from __future__ import annotations
import math, random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "anim"
OUT.mkdir(parents=True, exist_ok=True)

VIOLET, INDIGO, BLUE, CYAN = "#8B5CFF", "#5C7CFF", "#2563eb", "#22D3EE"

DEFS = f'''  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{VIOLET}"/>
      <stop offset="0.5" stop-color="{INDIGO}"/>
      <stop offset="1" stop-color="{CYAN}"/>
    </linearGradient>
    <linearGradient id="gh" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{VIOLET}"/>
      <stop offset="0.5" stop-color="{INDIGO}"/>
      <stop offset="1" stop-color="{CYAN}"/>
    </linearGradient>
  </defs>'''


def wrap(w, h, label, body, css):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" fill="none" role="img" aria-label="{label}">
{DEFS}
  <style>
    .lbl{{font:600 13px -apple-system,BlinkMacSystemFont,system-ui,sans-serif;fill:#86868b}}
    .mono{{font:600 12px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#86868b}}
    .cap{{font:700 14px -apple-system,system-ui,sans-serif}}
{css}
    @media (prefers-reduced-motion: reduce){{
      * {{ animation: none !important; }}
      .ron {{ opacity: 1 !important; }}
      .roff {{ opacity: .25 !important; }}
    }}
  </style>
{body}
</svg>
'''


# ── 1. denoise.svg ────────────────────────────────────────────────────────────
def denoise():
    W, H = 480, 200
    rnd = random.Random(7)
    # scattered "noise" field
    noise = []
    for _ in range(64):
        x = rnd.uniform(40, 440); y = rnd.uniform(30, 170)
        r = rnd.uniform(1.0, 2.6); o = rnd.uniform(0.25, 0.7)
        noise.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" opacity="{o:.2f}"/>')
    noise_svg = '\n      '.join(noise)
    # the clean "sample": logo-style denoise curve + ordered particles
    clean = '''<path d="M44 150 C 120 60, 240 90, 320 70 C 380 56, 410 60, 440 52"
            stroke="url(#gh)" stroke-width="5" stroke-linecap="round" fill="none"/>
      <circle cx="44" cy="150" r="6" fill="#8B5CFF"/>
      <circle cx="320" cy="70" r="4" fill="#5C7CFF"/>
      <circle cx="440" cy="52" r="5" fill="#22D3EE"/>
      <circle cx="392" cy="58" r="2.6" fill="#22D3EE" opacity="0.7"/>
      <circle cx="356" cy="62" r="2.0" fill="#22D3EE" opacity="0.55"/>'''
    body = f'''  <text class="lbl roff noiseLbl" x="44" y="22">noise · t=T</text>
  <text class="lbl ron sampleLbl" x="372" y="22">sample · t=0</text>
  <g class="noise" fill="#86868b">
      {noise_svg}
  </g>
  <g class="clean">
      {clean}
  </g>
  <rect class="sweep" x="0" y="26" width="3" height="150" fill="url(#gh)" opacity="0.9" rx="1.5"/>'''
    css = '''    .noise{animation:dN 5.5s ease-in-out infinite}
    .clean{opacity:0;animation:dC 5.5s ease-in-out infinite}
    .noiseLbl{animation:dN 5.5s ease-in-out infinite}
    .sampleLbl{opacity:0;animation:dC 5.5s ease-in-out infinite}
    .sweep{animation:dS 5.5s ease-in-out infinite}
    @keyframes dN{0%,12%{opacity:.95}48%,78%{opacity:0}96%,100%{opacity:.95}}
    @keyframes dC{0%,12%{opacity:0}48%,78%{opacity:1}96%,100%{opacity:0}}
    @keyframes dS{0%,8%{transform:translateX(40px)}48%{transform:translateX(440px)}
      62%{opacity:.9}78%,100%{transform:translateX(440px);opacity:0}}'''
    (OUT / "denoise.svg").write_text(wrap(W, H, "Diffusion: noise resolves into a sample", body, css))
    print("  -> anim/denoise.svg")


# ── 2. hub.svg ──────────────────────────────────────────────────────────────
def hub():
    W, H = 560, 300
    cx, cy = 280, 150
    ins = ["text", "image", "video", "robot state"]
    outs = ["image", "video", "audio", "actions", "3D"]
    def col(items, x, anchor, side):
        n = len(items); gap = 200 / (n - 1); top = cy - 100
        rows = []
        for i, t in enumerate(items):
            y = top + i * gap
            rows.append((t, y))
        return rows
    in_rows = col(ins, 70, "start", "L")
    out_rows = col(outs, 490, "end", "R")

    paths, flows, labels = [], [], []
    for i, (t, y) in enumerate(in_rows):
        d = f"M120 {y:.0f} C 190 {y:.0f}, 200 {cy}, 232 {cy}"
        pid = f"il{i}"
        paths.append(f'<path id="{pid}" d="{d}" stroke="#3a3a3e" stroke-width="1.5" fill="none" opacity="0.5"/>')
        flows.append(f'<circle r="3" fill="url(#gh)"><animateMotion dur="2.4s" begin="{i*0.3:.1f}s" repeatCount="indefinite" path="{d}"/></circle>')
        labels.append(f'<text class="mono" x="112" y="{y+4:.0f}" text-anchor="end">{t}</text>')
    for i, (t, y) in enumerate(out_rows):
        d = f"M328 {cy} C 360 {cy}, 370 {y:.0f}, 440 {y:.0f}"
        pid = f"ol{i}"
        paths.append(f'<path id="{pid}" d="{d}" stroke="#3a3a3e" stroke-width="1.5" fill="none" opacity="0.5"/>')
        flows.append(f'<circle r="3" fill="url(#gh)"><animateMotion dur="2.4s" begin="{1.0+i*0.3:.1f}s" repeatCount="indefinite" path="{d}"/></circle>')
        labels.append(f'<text class="mono" x="448" y="{y+4:.0f}" text-anchor="start">{t}</text>')

    body = f'''  {''.join(paths)}
  {''.join(flows)}
  {''.join(labels)}
  <g class="pulse">
    <rect x="232" y="{cy-26}" width="96" height="52" rx="16"
          fill="rgba(124,92,255,0.10)" stroke="url(#gh)" stroke-width="2"/>
  </g>
  <text class="cap" x="{cx}" y="{cy-2}" text-anchor="middle" fill="url(#gh)">use_</text>
  <text class="cap" x="{cx}" y="{cy+16}" text-anchor="middle" fill="url(#gh)">diffusers</text>
  <text class="lbl" x="{cx}" y="40" text-anchor="middle">one tool</text>
  <text class="lbl" x="120" y="280" text-anchor="middle">inputs</text>
  <text class="lbl" x="440" y="280" text-anchor="middle">outputs</text>'''
    css = '''    .pulse{transform-origin:280px 150px;animation:hp 3s ease-in-out infinite}
    @keyframes hp{0%,100%{opacity:.85}50%{opacity:1;transform:scale(1.04)}}'''
    (OUT / "hub.svg").write_text(wrap(W, H, "use_diffusers: one tool, every modality", body, css))
    print("  -> anim/hub.svg")


# ── 3. wfm.svg (hero) ─────────────────────────────────────────────────────────
def wfm():
    W, H = 600, 280
    # left: camera frame + prompt  →  Cosmos  →  world video + action path
    # action trajectory points (an end-effector path)
    pts = [(430,150),(452,120),(486,108),(512,128),(540,116),(560,92)]
    path_d = "M" + " L".join(f"{x} {y}" for x, y in pts)
    dots = '\n      '.join(f'<circle cx="{x}" cy="{y}" r="2.4" fill="#22D3EE" opacity="0.8"/>' for x, y in pts)
    body = f'''  <!-- input: observation frame -->
  <rect x="36" y="92" width="96" height="72" rx="10" fill="rgba(124,92,255,0.08)"
        stroke="#3a3a3e" stroke-width="1.5"/>
  <circle cx="70" cy="128" r="9" fill="none" stroke="url(#gh)" stroke-width="2.5"/>
  <rect x="86" y="120" width="30" height="20" rx="4" fill="none" stroke="url(#gh)" stroke-width="2"/>
  <text class="lbl" x="84" y="182" text-anchor="middle">first frame</text>
  <!-- input: prompt -->
  <rect x="36" y="40" width="180" height="34" rx="9" fill="rgba(34,211,238,0.06)"
        stroke="#3a3a3e" stroke-width="1.5"/>
  <text class="mono" x="50" y="61" fill="#a1a1a6">“put the pot left…”</text>

  <!-- flow into cosmos -->
  <path id="w1" d="M132 128 C 180 128, 196 140, 232 140" stroke="#3a3a3e" stroke-width="1.5" fill="none"/>
  <circle r="3" fill="url(#gh)"><animateMotion dur="2s" repeatCount="indefinite" path="M132 128 C 180 128, 196 140, 232 140"/></circle>
  <path id="w0" d="M216 57 C 230 57, 224 120, 240 132" stroke="#3a3a3e" stroke-width="1.5" fill="none" opacity="0.6"/>
  <circle r="2.6" fill="url(#gh)"><animateMotion dur="2s" begin="0.4s" repeatCount="indefinite" path="M216 57 C 230 57, 224 120, 240 132"/></circle>

  <!-- cosmos world model -->
  <g class="pulse">
    <rect x="236" y="108" width="84" height="64" rx="16" fill="rgba(92,124,255,0.12)"
          stroke="url(#gh)" stroke-width="2"/>
  </g>
  <text class="cap" x="278" y="136" text-anchor="middle" fill="url(#gh)">world</text>
  <text class="cap" x="278" y="154" text-anchor="middle" fill="url(#gh)">model</text>

  <!-- two outputs -->
  <path d="M320 134 C 356 134, 366 70, 410 70" stroke="#3a3a3e" stroke-width="1.5" fill="none"/>
  <circle r="3" fill="url(#gh)"><animateMotion dur="2s" begin="0.9s" repeatCount="indefinite" path="M320 134 C 356 134, 366 70, 410 70"/></circle>
  <path d="M320 150 C 356 150, 366 150, 410 150" stroke="#3a3a3e" stroke-width="1.5" fill="none"/>
  <circle r="3" fill="url(#gh)"><animateMotion dur="2s" begin="1.1s" repeatCount="indefinite" path="M320 150 C 356 150, 366 150, 410 150"/></circle>

  <!-- output 1: world video (scanline) -->
  <rect x="412" y="44" width="150" height="52" rx="9" fill="rgba(124,92,255,0.06)"
        stroke="#3a3a3e" stroke-width="1.5"/>
  <rect class="scan" x="412" y="44" width="22" height="52" rx="9" fill="url(#gh)" opacity="0.30"/>
  <text class="lbl" x="487" y="116" text-anchor="middle">world video · .mp4</text>

  <!-- output 2: action trajectory -->
  <path class="trace" d="{path_d}" stroke="url(#gh)" stroke-width="2.5" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
  {dots}
  <circle class="ee" r="4.5" fill="#fff" stroke="url(#gh)" stroke-width="2">
    <animateMotion dur="3s" repeatCount="indefinite" path="{path_d}"/>
  </circle>
  <text class="lbl" x="495" y="182" text-anchor="middle">action chunk · .json</text>

  <text class="lbl" x="300" y="240" text-anchor="middle">one call → a future you can watch <tspan fill="#a1a1a6">and</tspan> actions you can run</text>'''
    css = '''    .pulse{transform-origin:278px 140px;animation:hp 3s ease-in-out infinite}
    @keyframes hp{0%,100%{opacity:.9}50%{opacity:1;transform:scale(1.05)}}
    .scan{animation:sc 2.6s ease-in-out infinite}
    @keyframes sc{0%{transform:translateX(0)}100%{transform:translateX(128px)}}
    .trace{stroke-dasharray:260;stroke-dashoffset:260;animation:tr 3s ease-in-out infinite}
    @keyframes tr{0%{stroke-dashoffset:260}60%,100%{stroke-dashoffset:0}}'''
    (OUT / "wfm.svg").write_text(wrap(W, H, "Robot world model: frame and prompt become a world video and an action chunk", body, css))
    print("  -> anim/wfm.svg")


# ── 4. discover.svg ───────────────────────────────────────────────────────────
def discover():
    W, H = 460, 260
    names = ["StableDiffusionXLPipeline","LTXPipeline","Cosmos3OmniPipeline",
             "AudioLDMPipeline","ShapEPipeline","FluxPipeline",
             "WanPipeline","HunyuanVideoPipeline"]
    rows = []
    y0 = 60; gap = 22
    for i, n in enumerate(names):
        y = y0 + i * gap
        rows.append(
            f'<text class="mono row r{i}" x="48" y="{y}">{n}</text>')
    rows_svg = '\n  '.join(rows)
    # scanning highlight bar
    body = f'''  <text class="lbl" x="48" y="32">diffusers._import_structure  →  read at runtime</text>
  <rect class="scanbar" x="40" y="{y0-14}" width="300" height="18" rx="5"
        fill="rgba(124,92,255,0.16)"/>
  {rows_svg}
  <text class="cap" x="48" y="{y0+len(names)*gap+18}" fill="url(#gh)">300+ pipelines</text>
  <text class="lbl" x="190" y="{y0+len(names)*gap+18}">· nothing hardcoded</text>'''
    # each row pulses bright as the scanbar passes it
    keyframes = []
    n = len(names); span = 0.8/n
    css_rows = ""
    for i in range(n):
        s = 0.1 + i*span
        css_rows += (f'    .r{i}{{animation:rowHi 6s ease-in-out infinite;animation-delay:{i*0.35:.2f}s}}\n')
    css = ('''    .row{fill:#86868b}
    .scanbar{animation:scb 6s ease-in-out infinite}
    @keyframes scb{0%,8%{transform:translateY(0);opacity:0}
      12%{opacity:1}88%{opacity:1}
      88%,100%{transform:translateY(154px);opacity:0}}
    @keyframes rowHi{0%,100%{fill:#86868b}}
'''
    )
    # simpler: highlight each row in sequence with its own keyframe
    css = '''    .row{fill:#86868b}
'''
    for i in range(n):
        d = i * (0.78/n)
        css += (f'    .r{i}{{animation:rh{i} 6s linear infinite}}\n'
                f'    @keyframes rh{i}{{0%,{8+i*9}%{{fill:#86868b}}{12+i*9}%,{20+i*9}%{{fill:#fff;font-weight:700}}{26+i*9}%,100%{{fill:#86868b}}}}\n')
    css += '''    .scanbar{animation:scb 6s linear infinite}
    @keyframes scb{0%,8%{transform:translateY(0);opacity:0}10%{opacity:.9}86%{opacity:.9}90%,100%{transform:translateY(158px);opacity:0}}'''
    (OUT / "discover.svg").write_text(wrap(W, H, "The pipeline catalog is read from diffusers at runtime", body, css))
    print("  -> anim/discover.svg")



# ════════════════════════════════════════════════════════════════════════════
#  ROUND 2 — robot modes, per-modality micro-animations, README banner
# ════════════════════════════════════════════════════════════════════════════

def _flow(d, dur, begin=0.0, r=3, op=1.0):
    return (f'<circle r="{r}" fill="url(#gh)" opacity="{op}">'
            f'<animateMotion dur="{dur}s" begin="{begin}s" repeatCount="indefinite" '
            f'path="{d}" calcMode="spline" keyTimes="0;1" keySplines="0.4 0 0.2 1"/></circle>')

def _wire(d, op=0.5):
    return f'<path d="{d}" stroke="#3a3a3e" stroke-width="1.5" fill="none" opacity="{op}"/>'


# ── 5. robot_modes.svg — the three Cosmos questions, side by side ─────────────
def robot_modes():
    W, H = 640, 300
    def cam(x, y, w=58, h=42):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                f'fill="rgba(124,92,255,0.08)" stroke="#3a3a3e" stroke-width="1.4"/>'
                f'<circle cx="{x+16}" cy="{y+h/2}" r="6" fill="none" stroke="url(#gh)" stroke-width="2"/>'
                f'<rect x="{x+26}" y="{y+h/2-7}" width="20" height="14" rx="3" fill="none" stroke="url(#gh)" stroke-width="1.6"/>')
    def chip(x, y, t, w=60, h=30):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                f'fill="rgba(34,211,238,0.06)" stroke="#3a3a3e" stroke-width="1.4"/>'
                f'<text class="mono" x="{x+w/2}" y="{y+h/2+4}" text-anchor="middle">{t}</text>')
    def core(x, y, t1, t2, w=70, h=50, cls="p0"):
        return (f'<g class="pulse {cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" '
                f'fill="rgba(92,124,255,0.12)" stroke="url(#gh)" stroke-width="1.8"/></g>'
                f'<text class="cap2" x="{x+w/2}" y="{y+h/2-1}" text-anchor="middle" fill="url(#gh)">{t1}</text>'
                f'<text class="cap2" x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" fill="url(#gh)">{t2}</text>')

    # three vertical lanes
    lane = 213
    body = []
    titles = [("policy","what should the robot do?"),
              ("forward dynamics","what happens if I act?"),
              ("inverse dynamics","what actions made this?")]
    for i,(t,sub) in enumerate(titles):
        x0 = i*lane
        body.append(f'<text class="cap" x="{x0+lane/2}" y="26" text-anchor="middle" fill="url(#gh)">{t}</text>')
        body.append(f'<text class="lbl" x="{x0+lane/2}" y="44" text-anchor="middle">{sub}</text>')
        if i: body.append(f'<line x1="{x0}" y1="60" x2="{x0}" y2="270" stroke="#2a2a2e" stroke-width="1"/>')

    # ---- lane 0: policy — frame+prompt -> core -> video + actions ----
    x=0
    body.append(cam(x+18, 96))
    body.append(f'<text class="lbl" x="{x+47}" y="156" text-anchor="middle">frame</text>')
    body.append(chip(x+16, 168, "prompt", 64))
    d1=f"M{x+76} 117 C {x+96} 117, {x+96} 130, {x+118} 132"; body.append(_wire(d1)); body.append(_flow(d1,2.0,0.0))
    d2=f"M{x+80} 183 C {x+100} 183, {x+100} 150, {x+118} 146"; body.append(_wire(d2)); body.append(_flow(d2,2.0,0.3))
    body.append(core(x+118, 110, "world","model", 64, 56, "p0"))
    do1=f"M{x+182} 124 C {x+196} 124, {x+196} 112, {x+205} 112"; body.append(_wire(do1)); body.append(_flow(do1,1.8,0.9))
    do2=f"M{x+182} 150 C {x+196} 150, {x+196} 162, {x+205} 162"; body.append(_wire(do2)); body.append(_flow(do2,1.8,1.1))

    # ---- lane 1: forward dynamics — frame + actions -> core -> video ----
    x=lane
    body.append(cam(x+16, 96))
    body.append(f'<text class="lbl" x="{x+45}" y="156" text-anchor="middle">frame</text>')
    # mini action squiggle as input
    body.append(f'<path d="M{x+18} 184 q 10 -12 20 0 t 20 0" stroke="url(#gh)" stroke-width="2" fill="none" class="trace2"/>')
    body.append(f'<text class="lbl" x="{x+45}" y="206" text-anchor="middle">actions</text>')
    d1=f"M{x+74} 117 C {x+94} 117, {x+94} 130, {x+116} 132"; body.append(_wire(d1)); body.append(_flow(d1,2.0,0.0))
    d2=f"M{x+62} 184 C {x+96} 184, {x+96} 150, {x+116} 146"; body.append(_wire(d2)); body.append(_flow(d2,2.0,0.3))
    body.append(core(x+116, 110, "forward","dyn.", 66, 56, "p1"))
    do1=f"M{x+182} 138 C {x+196} 138, {x+196} 138, {x+205} 138"; body.append(_wire(do1)); body.append(_flow(do1,1.8,0.9))

    # ---- lane 2: inverse dynamics — video -> core -> actions ----
    x=2*lane
    # film strip input
    for k in range(3):
        body.append(f'<rect x="{x+20+k*16}" y="104" width="14" height="28" rx="2" fill="rgba(124,92,255,0.08)" stroke="#3a3a3e" stroke-width="1.1"/>')
    body.append(f'<rect class="scan2" x="{x+20}" y="104" width="10" height="28" rx="2" fill="url(#gh)" opacity="0.3"/>')
    body.append(f'<text class="lbl" x="{x+44}" y="150" text-anchor="middle">video</text>')
    d1=f"M{x+70} 118 C {x+92} 118, {x+92} 134, {x+114} 138"; body.append(_wire(d1)); body.append(_flow(d1,2.0,0.0))
    body.append(core(x+114, 112, "inverse","dyn.", 66, 56, "p2"))
    do1=f"M{x+180} 140 C {x+192} 140, {x+192} 140, {x+200} 140"; body.append(_wire(do1)); body.append(_flow(do1,1.8,0.9))
    # output: action path traced
    apts=[(x+200,150),(x+208,134),(x+218,142),(x+228,128),(x+238,138)]
    apath="M"+" L".join(f"{px} {py}" for px,py in apts)
    body.append(f'<path class="trace2" d="{apath}" stroke="url(#gh)" stroke-width="2.2" fill="none" stroke-linecap="round"/>')

    # shared output labels
    body.append(f'<text class="lbl" x="{205+6}" y="104" text-anchor="start">video</text>')
    body.append(f'<text class="lbl" x="{205+6}" y="172" text-anchor="start">actions</text>')
    body.append(f'<text class="lbl" x="{lane+205}" y="130" text-anchor="start">video</text>')
    body.append(f'<text class="lbl" x="{2*lane+196}" y="170" text-anchor="start">actions</text>')

    css = '''    .cap2{font:700 12px -apple-system,system-ui,sans-serif}
    .pulse{animation:rmp 3s ease-in-out infinite}
    .p1{animation-delay:.5s}.p2{animation-delay:1s}
    @keyframes rmp{0%,100%{opacity:.85}50%{opacity:1}}
    .trace2{stroke-dasharray:120;stroke-dashoffset:120;animation:rmt 2.6s ease-in-out infinite}
    @keyframes rmt{0%{stroke-dashoffset:120}55%,100%{stroke-dashoffset:0}}
    .scan2{animation:rms 2s linear infinite}
    @keyframes rms{0%{transform:translateX(0)}100%{transform:translateX(32px)}}'''
    (OUT/"robot_modes.svg").write_text(wrap(W,H,"Three Cosmos modes: policy, forward dynamics, inverse dynamics", "\n  ".join(body), css))
    print("  -> anim/robot_modes.svg")


# ── 6. m_image.svg — an image tile resolving from noise (micro) ──────────────
def m_image():
    W,H=200,200
    rnd=random.Random(11)
    # pixel grid that "resolves": noisy squares fade, a clean gradient image appears
    cells=[]
    g=10; cell=(W-40)/g
    for r in range(g):
        for c in range(g):
            x=20+c*cell; y=20+r*cell
            o=rnd.uniform(0.15,0.6)
            cells.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="#86868b" opacity="{o:.2f}"/>')
    noise="\n      ".join(cells)
    body=f'''  <g class="mn">{noise}</g>
  <rect class="mc" x="20" y="20" width="{W-40}" height="{H-40}" rx="10" fill="url(#g)"/>
  <circle class="mc" cx="70" cy="78" r="16" fill="#fff" opacity="0.85"/>
  <path class="mc" d="M40 150 L80 110 L110 140 L140 100 L160 150 Z" fill="#0b0b0d" opacity="0.35"/>
  <rect x="20" y="20" width="{W-40}" height="{H-40}" rx="10" fill="none" stroke="url(#gh)" stroke-width="1.5" opacity="0.5"/>'''
    css='''    .mn{animation:miN 4s ease-in-out infinite}
    .mc{opacity:0;animation:miC 4s ease-in-out infinite}
    @keyframes miN{0%,10%{opacity:1}55%,90%{opacity:0}100%{opacity:1}}
    @keyframes miC{0%,10%{opacity:0}55%,90%{opacity:1}100%{opacity:0}}'''
    (OUT/"m_image.svg").write_text(wrap(W,H,"an image resolving from noise",body,css))
    print("  -> anim/m_image.svg")


# ── 7. m_audio.svg — a waveform drawing itself ───────────────────────────────
def m_audio():
    W,H=320,140
    cy=H/2; n=120
    pts=[]
    rnd=random.Random(5)
    for i in range(n+1):
        x=10+i*(W-20)/n
        env=math.sin(i/n*math.pi)            # amplitude envelope
        y=cy + math.sin(i*0.5)*28*env*rnd.uniform(0.5,1.0)
        pts.append(f"{x:.1f} {y:.1f}")
    d="M"+" L".join(pts)
    # mirrored bars under it for a richer look
    bars=[]
    for i in range(0,n+1,3):
        x=10+i*(W-20)/n
        env=math.sin(i/n*math.pi)
        h=abs(math.sin(i*0.5))*26*env+2
        bars.append(f'<rect x="{x:.1f}" y="{cy-h:.1f}" width="2" height="{2*h:.1f}" rx="1" fill="url(#gh)" opacity="0.25"/>')
    body=f'''  <g class="bars">{''.join(bars)}</g>
  <path class="wave" d="{d}" stroke="url(#gh)" stroke-width="2.4" fill="none" stroke-linecap="round"/>
  <circle class="head" r="4" fill="#fff" stroke="url(#gh)" stroke-width="2">
    <animateMotion dur="3s" repeatCount="indefinite" path="{d}"/>
  </circle>
  <text class="lbl" x="{W/2}" y="{H-6}" text-anchor="middle">.wav · sample rate from the model</text>'''
    css='''    .wave{stroke-dasharray:900;stroke-dashoffset:900;animation:wd 3s ease-in-out infinite}
    @keyframes wd{0%{stroke-dashoffset:900}70%,100%{stroke-dashoffset:0}}
    .bars{animation:bp 3s ease-in-out infinite}
    @keyframes bp{0%,30%{opacity:.1}70%,100%{opacity:.35}}'''
    (OUT/"m_audio.svg").write_text(wrap(W,H,"audio waveform drawing itself",body,css))
    print("  -> anim/m_audio.svg")


# ── 8. m_mesh.svg — a rotating wireframe (3D) ────────────────────────────────
def m_mesh():
    W,H=200,200; cx,cy=100,100; R=60
    # icosphere-ish: two offset rings + verticals to fake rotation via skewX
    rings=[]
    for ry,rx,op in [(22,58,0.9),(40,52,0.55),(12,60,0.7)]:
        rings.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="url(#gh)" stroke-width="1.6" opacity="{op}"/>')
    # longitudes
    lon=[]
    for rx in [60,42,20]:
        lon.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="60" fill="none" stroke="url(#gh)" stroke-width="1.3" opacity="0.5"/>')
    verts=[]
    rnd=random.Random(3)
    for _ in range(14):
        a=rnd.uniform(0,2*math.pi); rr=rnd.uniform(0.4,1.0)*R
        verts.append(f'<circle cx="{cx+math.cos(a)*rr:.1f}" cy="{cy+math.sin(a)*rr*0.5:.1f}" r="1.8" fill="#22D3EE" opacity="0.8"/>')
    body=f'''  <g class="spin">
    <circle cx="{cx}" cy="{cy}" r="{R}" fill="rgba(124,92,255,0.05)"/>
    {''.join(rings)}
    {''.join(lon)}
    {''.join(verts)}
  </g>
  <text class="lbl" x="{cx}" y="{H-8}" text-anchor="middle">.ply / .obj mesh</text>'''
    css='''    .spin{transform-origin:100px 100px;animation:sp 8s linear infinite}
    @keyframes sp{0%{transform:rotateY(0) scaleX(1)}25%{transform:scaleX(.3)}
      50%{transform:scaleX(-1)}75%{transform:scaleX(-.3)}100%{transform:scaleX(1)}}'''
    (OUT/"m_mesh.svg").write_text(wrap(W,H,"a rotating 3D wireframe mesh",body,css))
    print("  -> anim/m_mesh.svg")


# ── 9. m_video.svg — a film strip advancing with a denoise scanline ──────────
def m_video():
    W,H=320,150
    body=[f'<text class="lbl" x="{W/2}" y="{H-8}" text-anchor="middle">frames → .mp4</text>']
    # film strip frames sliding
    body.append('<g class="strip">')
    fw=70; gap=10; n=6
    for i in range(n):
        x=i*(fw+gap)
        body.append(f'<rect x="{x}" y="20" width="{fw}" height="80" rx="6" fill="rgba(124,92,255,0.07)" stroke="#3a3a3e" stroke-width="1.2"/>')
        # tiny evolving content: a dot moving across frames
        body.append(f'<circle cx="{x+12+i*9}" cy="{74-i*7}" r="5" fill="url(#gh)" opacity="0.8"/>')
        # sprocket holes
        for sx in (x+10,x+fw-10):
            body.append(f'<rect x="{sx-3}" y="24" width="6" height="5" rx="1" fill="#3a3a3e"/>')
            body.append(f'<rect x="{sx-3}" y="91" width="6" height="5" rx="1" fill="#3a3a3e"/>')
    body.append('</g>')
    # denoise scanline sweeping the visible window
    body.append(f'<rect class="vscan" x="0" y="20" width="26" height="80" fill="url(#gh)" opacity="0.18" rx="6"/>')
    css='''    .strip{animation:vs 6s linear infinite}
    @keyframes vs{0%{transform:translateX(0)}100%{transform:translateX(-160px)}}
    .vscan{animation:vsc 2.4s ease-in-out infinite}
    @keyframes vsc{0%{transform:translateX(0)}100%{transform:translateX(300px)}}'''
    (OUT/"m_video.svg").write_text(wrap(W,H,"video frames exported to mp4","\n  ".join(body),css))
    print("  -> anim/m_video.svg")


# ── 10. banner.svg — wide README hero ────────────────────────────────────────
def banner():
    W,H=1280,360
    rnd=random.Random(42)
    # left: noise resolving;  center: use_diffusers;  right: 5 modality glyphs
    noise=[]
    for _ in range(90):
        x=rnd.uniform(40,360); y=rnd.uniform(60,300)
        r=rnd.uniform(1.2,3.0); o=rnd.uniform(0.2,0.6)
        noise.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" opacity="{o:.2f}"/>')
    noise_svg="\n      ".join(noise)
    # input lines
    ins=[("text",120),("image",165),("video",210),("robot state",255)]
    inlines=[]
    for t,y in ins:
        d=f"M380 {y} C 470 {y}, 500 180, 560 180"
        inlines.append(_wire(d,0.45)); inlines.append(_flow(d,2.6,rnd.uniform(0,1)))
        inlines.append(f'<text class="bmono" x="372" y="{y+4}" text-anchor="end">{t}</text>')
    # output glyphs on the right
    outs=["image","video","audio","actions","3D"]
    outlines=[]
    oy0=80
    for i,t in enumerate(outs):
        y=oy0+i*48
        d=f"M720 180 C 790 180, 800 {y}, 900 {y}"
        outlines.append(_wire(d,0.45)); outlines.append(_flow(d,2.6,1.0+i*0.25))
        outlines.append(f'<text class="bmono" x="912" y="{y+4}" text-anchor="start">{t}</text>')

    body=f'''  <rect x="0" y="0" width="{W}" height="{H}" fill="none"/>
  <g class="bnoise" fill="#86868b">
      {noise_svg}
  </g>
  <path class="bcurve" d="M60 280 C 160 120, 300 180, 360 150"
        stroke="url(#gh)" stroke-width="5" fill="none" stroke-linecap="round"/>
  {''.join(inlines)}
  {''.join(outlines)}
  <g class="bpulse">
    <rect x="560" y="148" width="160" height="64" rx="20"
          fill="rgba(124,92,255,0.12)" stroke="url(#gh)" stroke-width="2.2"/>
  </g>
  <text class="bcap" x="640" y="178" text-anchor="middle" fill="url(#gh)">use_diffusers</text>
  <text class="bsub" x="640" y="200" text-anchor="middle">one @tool</text>
  <text class="btitle" x="640" y="56" text-anchor="middle">strands-diffusers</text>
  <text class="blbl" x="640" y="330" text-anchor="middle">300+ diffusion pipelines · every modality · even robot world models</text>'''
    css='''    .btitle{font:800 34px -apple-system,system-ui,sans-serif;fill:url(#gh)}
    .bcap{font:800 22px -apple-system,system-ui,sans-serif}
    .bsub{font:600 13px -apple-system,system-ui,sans-serif;fill:#86868b}
    .bmono{font:600 15px ui-monospace,Menlo,monospace;fill:#a1a1a6}
    .blbl{font:600 16px -apple-system,system-ui,sans-serif;fill:#86868b}
    .bpulse{transform-origin:640px 180px;animation:bp 3.4s ease-in-out infinite}
    @keyframes bp{0%,100%{opacity:.9}50%{opacity:1;transform:scale(1.03)}}
    .bnoise{animation:bn 6s ease-in-out infinite}
    @keyframes bn{0%,15%{opacity:.9}55%{opacity:.25}100%{opacity:.9}}
    .bcurve{stroke-dasharray:340;stroke-dashoffset:340;animation:bc 6s ease-in-out infinite}
    @keyframes bc{0%,12%{stroke-dashoffset:340}55%,100%{stroke-dashoffset:0}}'''
    (OUT/"banner.svg").write_text(wrap(W,H,"strands-diffusers: one tool, 300+ pipelines, every modality",body,css))
    print("  -> anim/banner.svg")




if __name__ == "__main__":
    print("Generating animated SVGs -> docs/assets/anim/")
    denoise(); hub(); wfm(); discover()
    print("Generating round-2 animations...")
    robot_modes(); m_image(); m_audio(); m_mesh(); m_video(); banner()
    print("done.")
