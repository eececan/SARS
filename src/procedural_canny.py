"""
Step 3c: Procedural canny + depth generation.

Why this module exists
----------------------
Photographs of parallel temples (Maison Carrée = hexastyle, 6 columns;
Pula = tetrastyle, 4 columns) have the WRONG column count for the
Templum Divi Augusti at Ankara, which is octastyle (8 columns front,
15 per flank). Using their canny maps as ControlNet conditioning produced
reconstructions with 6 columns labelled as Ankara — geometrically a
relabeling of the wrong temple.

This module synthesises canny + depth maps directly from the dimensions
extracted by `analyze_plans.py`. Every line in the conditioning traces
to a documented number (column count, intercolumniation, dimensions),
so the resulting reconstruction is archaeologically defensible.

Inputs
------
A dict of plan dimensions, typically from `load_plan_dims_aggregated()`:
    columns_front, columns_side, temple_length_m, temple_width_m,
    column_diameter_m, intercolumniation_m, cella_length_m, cella_width_m

Outputs
-------
PIL.Image canny (white edges on black, 1024x1024)
PIL.Image depth (greyscale, near=white far=black, 1024x1024)

These plug directly into ControlNet as canny + depth conditioning.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Default Ankara temple dimensions (scholarship-derived fallback)
# Hänlein-Schäfer 1985 (Veneratio Augusti); Mitchell & Waelkens 1998
# (Pisidian Antioch); Botteri et al. (Ancyra survey).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "columns_front": 8,
    "columns_side": 15,
    "temple_length_m": 36.0,
    "temple_width_m": 17.0,
    "column_diameter_m": 1.5,
    "intercolumniation_m": 2.0,
    "cella_length_m": 22.0,
    "cella_width_m": 11.0,
}

# Vitruvian Corinthian proportions
COL_HEIGHT_DIAMETERS = 9.5
CAPITAL_HEIGHT_DIAMETERS = 1.0
CAPITAL_WIDTH_RATIO = 1.3
BASE_HEIGHT_DIAMETERS = 0.4
BASE_WIDTH_RATIO = 1.2
ENTABLATURE_RATIO = 0.25      # of column height
PEDIMENT_SLOPE = 1 / 9         # height / base ratio
KREPIDOMA_STEPS = 3
KREPIDOMA_STEP_HEIGHT_M = 0.3

LINE_W = 2
CANVAS = 1024


# ---------------------------------------------------------------------------
# Plan-dimension loading
# ---------------------------------------------------------------------------

def load_plan_dims_aggregated(analysis_dir: str) -> dict:
    """Merge dimensions from every `pass: plans_extraction` analysis.
    Take the median where multiple analyses disagree, fall back to
    DEFAULTS for fields nobody extracted.
    """
    fields = (
        "columns_front", "columns_side",
        "temple_length_m", "temple_width_m",
        "cella_length_m", "cella_width_m",
        "column_diameter_m", "intercolumniation_m",
    )
    collected: dict[str, list[float]] = {f: [] for f in fields}

    for jf in Path(analysis_dir).glob("*_analysis.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("pass") != "plans_extraction" or "error" in data:
            continue
        for f in fields:
            v = data.get(f)
            if isinstance(v, (int, float)) and v > 0:
                collected[f].append(float(v))

    dims = dict(DEFAULTS)
    for f, vs in collected.items():
        if vs:
            dims[f] = float(np.median(vs))

    if collected["columns_front"]:
        dims["columns_front"] = int(round(dims["columns_front"]))
    if collected["columns_side"]:
        dims["columns_side"] = int(round(dims["columns_side"]))

    return dims


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (CANVAS, CANVAS), 0)
    return img, ImageDraw.Draw(img)


def _scale_for(temple_w_m: float, total_h_m: float, margin: float = 0.08) -> float:
    """Pick pixel-per-meter scale so the temple fits the canvas with margin."""
    avail_w = CANVAS * (1 - 2 * margin)
    avail_h = CANVAS * (1 - 2 * margin)
    return min(avail_w / temple_w_m, avail_h / total_h_m)


def _draw_column(draw, x: int, y_bottom: int, y_top: int, diam_px: int):
    """Vertical column shaft from y_bottom (base) to y_top (capital bottom)."""
    half = diam_px // 2
    draw.line([(x - half, y_bottom), (x - half, y_top)], fill=255, width=LINE_W)
    draw.line([(x + half, y_bottom), (x + half, y_top)], fill=255, width=LINE_W)


def _draw_corinthian_capital(draw, x: int, y_bottom: int, diam_px: int):
    """Capital atop the shaft. Bottom at y_bottom, flares to wider abacus on top."""
    shaft_half = diam_px // 2
    cap_h = int(diam_px * CAPITAL_HEIGHT_DIAMETERS)
    abacus_half = int(diam_px * CAPITAL_WIDTH_RATIO / 2)
    y_top = y_bottom - cap_h
    # Bell sides (flare)
    draw.line([(x - shaft_half, y_bottom), (x - abacus_half, y_top)], fill=255, width=LINE_W)
    draw.line([(x + shaft_half, y_bottom), (x + abacus_half, y_top)], fill=255, width=LINE_W)
    # Abacus (top of capital — slight projection)
    draw.line([(x - abacus_half, y_top), (x + abacus_half, y_top)], fill=255, width=LINE_W)
    # Acanthus suggestion: a short horizontal across the bell middle
    mid_y = y_bottom - cap_h // 2
    draw.line([(x - int(diam_px * 0.55), mid_y), (x + int(diam_px * 0.55), mid_y)],
              fill=255, width=LINE_W)
    return y_top


def _draw_attic_base(draw, x: int, y_top: int, diam_px: int):
    """Base under the shaft. Top at y_top, flares wider at the bottom."""
    shaft_half = diam_px // 2
    base_h = int(diam_px * BASE_HEIGHT_DIAMETERS)
    base_half = int(diam_px * BASE_WIDTH_RATIO / 2)
    y_bottom = y_top + base_h
    draw.line([(x - shaft_half, y_top), (x - base_half, y_bottom)], fill=255, width=LINE_W)
    draw.line([(x + shaft_half, y_top), (x + base_half, y_bottom)], fill=255, width=LINE_W)
    draw.line([(x - base_half, y_bottom), (x + base_half, y_bottom)], fill=255, width=LINE_W)
    return y_bottom


def _draw_krepidoma(draw, cx: int, top_y: int, base_y: int, top_half_w: int):
    """Three stepped horizontal levels under the stylobate. Each lower step
    is wider than the one above (overhang). Top of krepidoma at top_y."""
    step_h = max(1, (base_y - top_y) // KREPIDOMA_STEPS)
    overhang_px = max(6, top_half_w // 30)
    for i in range(KREPIDOMA_STEPS):
        y_top = top_y + i * step_h
        y_bot = top_y + (i + 1) * step_h if i < KREPIDOMA_STEPS - 1 else base_y
        half_w = top_half_w + (i + 1) * overhang_px
        x_left = cx - half_w
        x_right = cx + half_w
        draw.line([(x_left, y_top), (x_right, y_top)], fill=255, width=LINE_W)
        if i == KREPIDOMA_STEPS - 1:
            draw.line([(x_left, y_bot), (x_right, y_bot)], fill=255, width=LINE_W)
        # Risers
        draw.line([(x_left, y_top), (x_left, y_bot)], fill=255, width=LINE_W)
        draw.line([(x_right, y_top), (x_right, y_bot)], fill=255, width=LINE_W)


def _draw_entablature(draw, x_left: int, x_right: int, y_bottom: int, height_px: int):
    """Architrave + frieze + cornice horizontals. Bottom edge at y_bottom."""
    y_top = y_bottom - height_px
    third = max(1, height_px // 3)
    # Bottom (architrave bottom = capital top — already drawn by abacus, redraw for safety)
    draw.line([(x_left, y_bottom), (x_right, y_bottom)], fill=255, width=LINE_W)
    # Architrave / frieze divider
    draw.line([(x_left, y_bottom - third), (x_right, y_bottom - third)], fill=255, width=LINE_W)
    # Frieze / cornice divider
    draw.line([(x_left, y_bottom - 2 * third), (x_right, y_bottom - 2 * third)], fill=255, width=LINE_W)
    # Top of cornice
    draw.line([(x_left, y_top), (x_right, y_top)], fill=255, width=LINE_W)
    # Cornice projection (slight overhang)
    overhang = max(4, (x_right - x_left) // 80)
    draw.line([(x_left - overhang, y_top), (x_right + overhang, y_top)], fill=255, width=LINE_W)
    return y_top


def _draw_pediment(draw, x_left: int, x_right: int, y_bottom: int):
    """Triangular pediment sitting on top of the entablature."""
    width = x_right - x_left
    height = int(width * PEDIMENT_SLOPE)
    cx = (x_left + x_right) // 2
    y_top = y_bottom - height
    # Raking cornices
    draw.line([(x_left, y_bottom), (cx, y_top)], fill=255, width=LINE_W)
    draw.line([(cx, y_top), (x_right, y_bottom)], fill=255, width=LINE_W)
    # Inner tympanum line (slightly inset, suggests the recessed field)
    inset = height // 5
    cx_t = cx
    y_top_t = y_top + inset
    x_left_t = x_left + int(inset / PEDIMENT_SLOPE)
    x_right_t = x_right - int(inset / PEDIMENT_SLOPE)
    if x_right_t > x_left_t:
        draw.line([(x_left_t, y_bottom - inset // 2), (cx_t, y_top_t)], fill=255, width=1)
        draw.line([(cx_t, y_top_t), (x_right_t, y_bottom - inset // 2)], fill=255, width=1)
    return y_top


# ---------------------------------------------------------------------------
# Front elevation
# ---------------------------------------------------------------------------

def render_front_elevation(dims: dict | None = None) -> tuple[Image.Image, Image.Image]:
    d = {**DEFAULTS, **(dims or {})}
    n = int(d["columns_front"])
    diam_m = d["column_diameter_m"]
    intercol_m = d["intercolumniation_m"]

    # Stylobate width = n columns + (n-1) intercolumniations + 0.5 col margin each side
    cols_width_m = n * diam_m + (n - 1) * intercol_m
    stylobate_w_m = cols_width_m + diam_m  # half-column margin both sides
    col_h_m = diam_m * COL_HEIGHT_DIAMETERS
    entab_h_m = col_h_m * ENTABLATURE_RATIO
    krepidoma_h_m = KREPIDOMA_STEPS * KREPIDOMA_STEP_HEIGHT_M
    pediment_h_m = stylobate_w_m * PEDIMENT_SLOPE

    total_h_m = krepidoma_h_m + col_h_m + entab_h_m + pediment_h_m
    s = _scale_for(stylobate_w_m, total_h_m)

    img, draw = _new_canvas()
    cx = CANVAS // 2
    base_y = int(CANVAS * 0.92)
    krepidoma_top_y = base_y - int(krepidoma_h_m * s)
    stylobate_top_y = krepidoma_top_y  # stylobate is the top step

    diam_px = max(8, int(diam_m * s))
    col_spacing_px = (diam_m + intercol_m) * s
    first_col_cx = cx - (cols_width_m * s) / 2 + (diam_m * s) / 2

    # Krepidoma
    stylobate_half_w = int(stylobate_w_m * s / 2)
    _draw_krepidoma(draw, cx, krepidoma_top_y, base_y, stylobate_half_w)

    # Stylobate top line (extra emphasis — columns sit on this)
    draw.line([(cx - stylobate_half_w, stylobate_top_y),
               (cx + stylobate_half_w, stylobate_top_y)],
              fill=255, width=LINE_W)

    # Columns (base → shaft → capital)
    base_h_px = int(diam_px * BASE_HEIGHT_DIAMETERS)
    shaft_bottom_y = stylobate_top_y - base_h_px
    shaft_top_y = shaft_bottom_y - int((col_h_m - diam_m * BASE_HEIGHT_DIAMETERS) * s)

    capital_top_y = shaft_top_y  # placeholder
    col_xs = []
    for i in range(n):
        x = int(first_col_cx + i * col_spacing_px)
        col_xs.append(x)
        _draw_attic_base(draw, x, shaft_bottom_y, diam_px)
        _draw_column(draw, x, shaft_bottom_y, shaft_top_y, diam_px)
        capital_top_y = _draw_corinthian_capital(draw, x, shaft_top_y, diam_px)

    # Entablature sits on the abacus row
    entab_left = cx - int(stylobate_w_m * s / 2)
    entab_right = cx + int(stylobate_w_m * s / 2)
    entab_h_px = int(entab_h_m * s)
    entab_top_y = _draw_entablature(draw, entab_left, entab_right, capital_top_y, entab_h_px)

    # Pediment
    _draw_pediment(draw, entab_left, entab_right, entab_top_y)

    depth = _depth_from_canny_silhouette(img)
    return img, depth


# ---------------------------------------------------------------------------
# Side elevation (peripteral flank)
# ---------------------------------------------------------------------------

def render_side_elevation(dims: dict | None = None) -> tuple[Image.Image, Image.Image]:
    d = {**DEFAULTS, **(dims or {})}
    n = int(d["columns_side"])
    diam_m = d["column_diameter_m"]
    intercol_m = d["intercolumniation_m"]

    cols_width_m = n * diam_m + (n - 1) * intercol_m
    stylobate_w_m = cols_width_m + diam_m
    col_h_m = diam_m * COL_HEIGHT_DIAMETERS
    entab_h_m = col_h_m * ENTABLATURE_RATIO
    krepidoma_h_m = KREPIDOMA_STEPS * KREPIDOMA_STEP_HEIGHT_M

    # No pediment in flank view — replaced by horizontal roof line and the
    # backside rake of the gable. Use entab_h_m + small roof.
    roof_h_m = stylobate_w_m * PEDIMENT_SLOPE * 0.5  # half of pediment slope (visible roof line)
    total_h_m = krepidoma_h_m + col_h_m + entab_h_m + roof_h_m
    s = _scale_for(stylobate_w_m, total_h_m)

    img, draw = _new_canvas()
    cx = CANVAS // 2
    base_y = int(CANVAS * 0.92)
    krepidoma_top_y = base_y - int(krepidoma_h_m * s)
    stylobate_top_y = krepidoma_top_y

    diam_px = max(6, int(diam_m * s))
    col_spacing_px = (diam_m + intercol_m) * s
    first_col_cx = cx - (cols_width_m * s) / 2 + (diam_m * s) / 2

    stylobate_half_w = int(stylobate_w_m * s / 2)
    _draw_krepidoma(draw, cx, krepidoma_top_y, base_y, stylobate_half_w)
    draw.line([(cx - stylobate_half_w, stylobate_top_y),
               (cx + stylobate_half_w, stylobate_top_y)],
              fill=255, width=LINE_W)

    base_h_px = int(diam_px * BASE_HEIGHT_DIAMETERS)
    shaft_bottom_y = stylobate_top_y - base_h_px
    shaft_top_y = shaft_bottom_y - int((col_h_m - diam_m * BASE_HEIGHT_DIAMETERS) * s)

    capital_top_y = shaft_top_y
    for i in range(n):
        x = int(first_col_cx + i * col_spacing_px)
        _draw_attic_base(draw, x, shaft_bottom_y, diam_px)
        _draw_column(draw, x, shaft_bottom_y, shaft_top_y, diam_px)
        capital_top_y = _draw_corinthian_capital(draw, x, shaft_top_y, diam_px)

    entab_left = cx - int(stylobate_w_m * s / 2)
    entab_right = cx + int(stylobate_w_m * s / 2)
    entab_h_px = int(entab_h_m * s)
    entab_top_y = _draw_entablature(draw, entab_left, entab_right, capital_top_y, entab_h_px)

    # Visible roof line (the gable seen from the side — a horizontal eave
    # line and the slanted hip going to the far end).
    roof_h_px = int(roof_h_m * s)
    roof_top_y = entab_top_y - roof_h_px
    # Eave line (top of cornice runs to the visible apex point — but in pure
    # side elevation, the roof reads as a flat horizontal at the apex height)
    draw.line([(entab_left, roof_top_y), (entab_right, roof_top_y)], fill=255, width=LINE_W)
    # Connecting slopes (front gable peak visible at one end if perspective hints wanted)
    draw.line([(entab_left, entab_top_y), (entab_left, roof_top_y)], fill=255, width=LINE_W)
    draw.line([(entab_right, entab_top_y), (entab_right, roof_top_y)], fill=255, width=LINE_W)

    depth = _depth_from_canny_silhouette(img)
    return img, depth


# ---------------------------------------------------------------------------
# Three-quarter view (oblique projection)
# ---------------------------------------------------------------------------

def render_three_quarter(dims: dict | None = None,
                          angle_deg: float = 28.0,
                          flank_compress: float = 0.55) -> tuple[Image.Image, Image.Image]:
    """Cabinet-oblique projection: front face true-size, side face foreshortened
    by `flank_compress` and tilted at `angle_deg`. Gives SDXL clean 3D
    structure without needing a true 3D renderer. The flank is compressed
    (standard for cabinet projection) so a 36m flank doesn't dwarf a 17m
    front."""
    d = {**DEFAULTS, **(dims or {})}
    nf = int(d["columns_front"])
    ns = int(d["columns_side"])
    diam_m = d["column_diameter_m"]
    intercol_m = d["intercolumniation_m"]
    spacing_m = diam_m + intercol_m

    cols_front_w_m = nf * diam_m + (nf - 1) * intercol_m
    cols_side_w_m = ns * diam_m + (ns - 1) * intercol_m
    front_w_m = cols_front_w_m + diam_m
    flank_w_m = cols_side_w_m + diam_m

    col_h_m = diam_m * COL_HEIGHT_DIAMETERS
    entab_h_m = col_h_m * ENTABLATURE_RATIO
    krep_h_m = KREPIDOMA_STEPS * KREPIDOMA_STEP_HEIGHT_M
    ped_h_m = front_w_m * PEDIMENT_SLOPE

    # Projection vector: a flank unit becomes (dx, dy) in 2D, both scaled
    # by `flank_compress`. dy is POSITIVE (receding goes up the canvas in
    # our meter-y-up convention).
    a = math.radians(angle_deg)
    proj_dx = flank_compress * math.cos(a)
    proj_dy = flank_compress * math.sin(a)
    flank_dx_m = flank_w_m * proj_dx
    flank_dy_m = flank_w_m * proj_dy

    total_w_m = front_w_m + flank_dx_m
    total_h_m = krep_h_m + col_h_m + entab_h_m + ped_h_m + flank_dy_m
    s = _scale_for(total_w_m, total_h_m, margin=0.05)

    img, draw = _new_canvas()
    # Origin: bottom-left of front face. Canvas y grows downward, meters y up.
    x0 = int(CANVAS * 0.08)
    y0 = int(CANVAS * 0.92)

    def m2p(xm: float, ym: float, flank_t: float = 0.0) -> tuple[int, int]:
        """Project meters → canvas pixels. flank_t in [0,1] = how far back
        into the flank this point sits."""
        px = x0 + (xm + flank_t * flank_dx_m) * s
        py = y0 - (ym + flank_t * flank_dy_m) * s
        return int(px), int(py)

    diam_px = max(5, int(diam_m * s))

    # ---------------- Krepidoma (3D prism) ----------------
    # Front face: rectangle
    fl = m2p(0, 0)
    fr = m2p(front_w_m, 0)
    fl_top = m2p(0, krep_h_m)
    fr_top = m2p(front_w_m, krep_h_m)
    for a_, b_ in [(fl, fr), (fr, fr_top), (fr_top, fl_top), (fl_top, fl)]:
        draw.line([a_, b_], fill=255, width=LINE_W)

    # Receding (right) side: skew right edge of front rect back
    br_bot = m2p(front_w_m, 0, flank_t=1.0)
    br_top = m2p(front_w_m, krep_h_m, flank_t=1.0)
    bl_top = m2p(0, krep_h_m, flank_t=1.0)
    draw.line([fr, br_bot], fill=255, width=LINE_W)
    draw.line([fr_top, br_top], fill=255, width=LINE_W)
    draw.line([br_bot, br_top], fill=255, width=LINE_W)
    # Top of krepidoma — back edge (visible)
    draw.line([fl_top, bl_top], fill=255, width=LINE_W)
    draw.line([bl_top, br_top], fill=255, width=LINE_W)

    # Krepidoma step lines on front face (3 horizontal subdivisions)
    for i in range(1, KREPIDOMA_STEPS):
        ym = krep_h_m * i / KREPIDOMA_STEPS
        draw.line([m2p(0, ym), m2p(front_w_m, ym)], fill=255, width=1)

    # ---------------- Helper: draw a column at (xm, base flank_t) ----------------
    def _col_at(xm: float, flank_t: float = 0.0):
        bot_p = m2p(xm, krep_h_m, flank_t)
        top_p = m2p(xm, krep_h_m + col_h_m, flank_t)
        cap_bot_p = m2p(xm, krep_h_m + col_h_m - diam_m * CAPITAL_HEIGHT_DIAMETERS, flank_t)
        # shaft (two verticals — note that we ignore projection within column width
        # for clarity; shaft sides are pure vertical lines)
        draw.line([(bot_p[0] - diam_px // 2, bot_p[1]),
                   (cap_bot_p[0] - diam_px // 2, cap_bot_p[1])], fill=255, width=LINE_W)
        draw.line([(bot_p[0] + diam_px // 2, bot_p[1]),
                   (cap_bot_p[0] + diam_px // 2, cap_bot_p[1])], fill=255, width=LINE_W)
        # capital (flare to abacus)
        abacus_half = int(diam_px * CAPITAL_WIDTH_RATIO / 2)
        draw.line([(cap_bot_p[0] - diam_px // 2, cap_bot_p[1]),
                   (top_p[0] - abacus_half, top_p[1])], fill=255, width=LINE_W)
        draw.line([(cap_bot_p[0] + diam_px // 2, cap_bot_p[1]),
                   (top_p[0] + abacus_half, top_p[1])], fill=255, width=LINE_W)
        draw.line([(top_p[0] - abacus_half, top_p[1]),
                   (top_p[0] + abacus_half, top_p[1])], fill=255, width=LINE_W)
        return top_p

    # ---------------- Front-row columns (8) ----------------
    # Columns sit at xm positions; the corner column will be reused as the
    # first side-row column.
    for i in range(nf):
        xm = diam_m / 2 + i * spacing_m
        _col_at(xm, flank_t=0.0)

    # ---------------- Side-row columns (15, skip corner to avoid double) ----------------
    # The side row begins at the corner column (which we already drew at
    # xm=front_w_m - diam_m/2, flank_t=0). Skip i=0; draw i=1..ns-1.
    for i in range(1, ns):
        # i-th column from the front along the flank
        t = (i * spacing_m) / flank_w_m
        # column sits on the receding stylobate, at the back-right edge xm = front_w_m - diam_m/2
        # plus a translation along the flank by t.
        _col_at(front_w_m - diam_m / 2, flank_t=t)

    # ---------------- Entablature ----------------
    capital_top_ym = krep_h_m + col_h_m
    entab_top_ym = capital_top_ym + entab_h_m
    # Front face: 4 horizontal lines (architrave bottom, two dividers, cornice top)
    for frac in (0.0, 1 / 3, 2 / 3, 1.0):
        ym = capital_top_ym + entab_h_m * frac
        draw.line([m2p(0, ym), m2p(front_w_m, ym)], fill=255, width=LINE_W)
    # Receding face: top + bottom edges visible
    draw.line([m2p(front_w_m, capital_top_ym),
               m2p(front_w_m, capital_top_ym, flank_t=1.0)], fill=255, width=LINE_W)
    draw.line([m2p(front_w_m, entab_top_ym),
               m2p(front_w_m, entab_top_ym, flank_t=1.0)], fill=255, width=LINE_W)
    # Far end vertical of entablature
    draw.line([m2p(front_w_m, capital_top_ym, flank_t=1.0),
               m2p(front_w_m, entab_top_ym, flank_t=1.0)], fill=255, width=LINE_W)

    # ---------------- Pediment + roof ridge ----------------
    ped_top_ym = entab_top_ym + ped_h_m
    ped_cx_xm = front_w_m / 2
    # Front face triangle
    draw.line([m2p(0, entab_top_ym), m2p(ped_cx_xm, ped_top_ym)], fill=255, width=LINE_W)
    draw.line([m2p(ped_cx_xm, ped_top_ym), m2p(front_w_m, entab_top_ym)], fill=255, width=LINE_W)
    # Roof ridge receding back
    draw.line([m2p(ped_cx_xm, ped_top_ym),
               m2p(ped_cx_xm, ped_top_ym, flank_t=1.0)], fill=255, width=LINE_W)
    # Back eaves: from the back corners of the entablature top edge to the back of the ridge
    draw.line([m2p(0, entab_top_ym, flank_t=1.0),
               m2p(ped_cx_xm, ped_top_ym, flank_t=1.0)], fill=255, width=1)
    draw.line([m2p(front_w_m, entab_top_ym, flank_t=1.0),
               m2p(ped_cx_xm, ped_top_ym, flank_t=1.0)], fill=255, width=1)

    # Depth: real near→far gradient across the bounding extent of the canny
    depth = _depth_from_canny_3q(img, x0, y0, front_w_m, flank_dx_m, s)
    return img, depth


def _depth_from_canny_3q(canny: Image.Image, x0: int, y0: int,
                          front_w_m: float, flank_dx_m: float, s: float) -> Image.Image:
    """Real near→far gradient for three-quarter view. The front face (at
    canvas-x = x0..x0+front_w_m*s) is foreground (bright); receding face
    fades to background (dark)."""
    arr = np.array(canny)
    h, w = arr.shape
    depth = np.zeros_like(arr)

    # Determine the canny's vertical extent
    row_has_edge = (arr > 32).any(axis=1)
    if not row_has_edge.any():
        return Image.fromarray(depth)
    y_top = int(np.argmax(row_has_edge))
    y_bot = h - int(np.argmax(row_has_edge[::-1]))

    front_x_left = int(x0)
    front_x_right = int(x0 + front_w_m * s)
    back_x = int(x0 + (front_w_m + flank_dx_m) * s)
    front_x_right = max(front_x_right, front_x_left + 1)
    back_x = max(back_x, front_x_right + 1)

    for x in range(w):
        if x < front_x_left:
            v = 0
        elif x < front_x_right:
            # Front face: brightest
            v = 220
        elif x < back_x:
            # Receding face: linear from 220 → 70
            t = (x - front_x_right) / (back_x - front_x_right)
            v = int(220 - t * 150)
        else:
            v = 0
        depth[y_top:y_bot, x] = v

    return Image.fromarray(depth)


# ---------------------------------------------------------------------------
# Depth maps
# ---------------------------------------------------------------------------

def _depth_from_canny_silhouette(canny: Image.Image) -> Image.Image:
    """For pure 2D elevations, depth is flat: the temple sits at a uniform
    middle depth, background is far. Fill the convex hull-ish region with
    a constant mid-grey, leave the rest dark. Gives the depth ControlNet
    a 'flatten everything in this plane' hint that suppresses bogus 3D."""
    arr = np.array(canny)
    h, w = arr.shape
    # Region of interest: any column that has at least one bright pixel
    col_has_edge = (arr > 32).any(axis=0)
    row_has_edge = (arr > 32).any(axis=1)
    if not col_has_edge.any() or not row_has_edge.any():
        return Image.fromarray(np.zeros_like(arr))
    x0 = int(np.argmax(col_has_edge))
    x1 = w - int(np.argmax(col_has_edge[::-1]))
    y0 = int(np.argmax(row_has_edge))
    y1 = h - int(np.argmax(row_has_edge[::-1]))
    depth = np.zeros_like(arr)
    depth[y0:y1, x0:x1] = 160  # mid-grey "this plane"
    # Soft gradient: brighter near base, darker near top (gives ground plane)
    grad = np.linspace(180, 130, y1 - y0).reshape(-1, 1)
    depth[y0:y1, x0:x1] = grad.astype(np.uint8)
    return Image.fromarray(depth)


def _depth_from_oblique(stylobate_w_m, flank_proj_x_m, flank_proj_y_m,
                         krepidoma_h_m, col_h_m, entab_h_m, pediment_h_m,
                         s, x0, y0) -> Image.Image:
    """For three-quarter view: real near→far gradient along the receding face.
    Front face = closest (bright), back of flank = farthest (dim)."""
    arr = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    total_h_m = krepidoma_h_m + col_h_m + entab_h_m + pediment_h_m

    # Sweep horizontal x positions; for each x, depth is linear interpolation
    # between front-face value (200) at left and back-face value (90) at right.
    front_face_x_px = int(x0)
    back_far_x_px = int(x0 + (stylobate_w_m + flank_proj_x_m * 2) * s)
    if back_far_x_px <= front_face_x_px:
        return Image.fromarray(arr)
    top_y_px = int(y0 - total_h_m * s)
    bot_y_px = int(y0)

    width_px = back_far_x_px - front_face_x_px
    grad_h = np.linspace(200, 90, width_px).astype(np.uint8)
    for x_idx, val in enumerate(grad_h):
        x_canvas = front_face_x_px + x_idx
        if 0 <= x_canvas < CANVAS:
            arr[max(0, top_y_px):min(CANVAS, bot_y_px), x_canvas] = val

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

RENDERERS = {
    "front_elevation": render_front_elevation,
    "side_elevation": render_side_elevation,
    "three_quarter": render_three_quarter,
}


def render_view(view_name: str, dims: dict | None = None) -> tuple[Image.Image, Image.Image]:
    if view_name not in RENDERERS:
        raise ValueError(f"Unknown view '{view_name}'. Known: {list(RENDERERS)}")
    return RENDERERS[view_name](dims)


def save_view(view_name: str, dims: dict | None, out_dir: str) -> dict:
    """Render + save canny and depth for a view. Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    canny, depth = render_view(view_name, dims)
    canny_p = out / f"{view_name}_canny.png"
    depth_p = out / f"{view_name}_depth.png"
    canny.save(canny_p)
    depth.save(depth_p)
    return {"canny_path": str(canny_p), "depth_path": str(depth_p)}
