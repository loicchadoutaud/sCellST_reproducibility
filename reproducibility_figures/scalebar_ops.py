from __future__ import annotations

import math

from matplotlib import font_manager
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

_LOC_MAP = {
    "upper right": 1, "upper left": 2, "lower left": 3, "lower right": 4,
    "right": 5, "center left": 6, "center right": 7,
    "lower center": 8, "upper center": 9, "center": 10,
}


def _nice_value(x: float) -> float:
    if x <= 0:
        return 1.0
    exp = int(math.floor(math.log10(x)))
    frac = x / (10 ** exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exp)


def _format_um(mm: float) -> str:
    um = mm * 1000.0
    if um >= 1000.0:
        return f"{um / 1000.0:.1f} mm"
    else:
        return f"{int(round(um))} µm"


def _loc_to_code(loc: str | int) -> int:
    if isinstance(loc, int):
        return loc
    return _LOC_MAP.get(str(loc).lower(), 4)  # default: lower right


def add_scale_bar(
        ax,
        um_per_px: float,
        image_shape: tuple[int, int] | None = None,  # optional; falls back to current x-limits
        frac: float = 0.2,
        bar_um: float | None = None,
        *,
        loc: str | int = "lower right",
        pad_pts: float = 2.0,  # padding between bar and box (points)
        borderpad_pts: float = 2.0,  # padding between box and axes edge (points)
        height_pts: float = 6.0,  # bar thickness (points)
        facecolor: str = "white",  # bar color
        textcolor: str = "white",
        box_alpha: float = 0.6,
        box_face: str = "black",
        box_edge: str = "none",
        label_above: bool = False,
        fontsize: int = 12,
):
    """
    Add an anchored scale bar that won't get cropped or disappear with inverted axes.

    Notes:
    - Uses data units for the bar length (so it truly matches pixels), but anchors the
      whole widget inside the axes with point-based padding so layout/tight save is safe.
    - Returns the created artist in case you want to keep a handle.
    """
    # Determine the visible width in *data units* the bar should represent.
    if image_shape is not None:
        W = int(image_shape[1])
    else:
        x0, x1 = ax.get_xlim()
        W = abs(x1 - x0)

    total_um = W * um_per_px
    if bar_um is None:
        # your helper; pick a "nice" value close to target
        target = total_um * frac
        bar_um = _nice_value(target / 1000.0) * 1000.0  # back to µm

    # Convert desired length to data units (pixels on x-axis)
    bar_px = max(1, int(round(bar_um / um_per_px)))

    # Make sure the bar fits in the current view (avoid accidental overflow)
    x0, x1 = ax.get_xlim()
    view_w = abs(x1 - x0)
    if bar_px > 0.95 * view_w:
        bar_px = int(0.95 * view_w)

    # Label (your helper formats in mm/µm as you like)
    label = _format_um(bar_um / 1000.0)

    # Font props for the label
    fp = font_manager.FontProperties(size=fontsize, weight="medium")

    # Build the anchored size bar; thickness is in points, length is in *data* units
    asb = AnchoredSizeBar(
        ax.transData,
        bar_px,  # size of the bar in data units
        label,  # label text
        _loc_to_code(loc),
        pad=pad_pts / 72.0,  # matplotlib expects 'pad' in fraction of font size; using points works well here
        borderpad=borderpad_pts / 72.0,
        sep=4,  # gap between bar and label (points)
        frameon=True,
        size_vertical=height_pts,  # bar thickness (points)
        color=facecolor,  # bar color; we'll set text color separately
        label_top=label_above,
        fontproperties=fp,
    )

    # Style the background box to ensure visibility
    asb.patch.set_facecolor(box_face)
    asb.patch.set_alpha(box_alpha)
    asb.patch.set_edgecolor(box_edge)

    # Ensure the label uses the requested color
    try:
        asb.txt_label._text.set_color(textcolor)
    except Exception:
        pass

    # Keep it above the image and immune to clipping
    asb.set_zorder(10)
    asb.set_clip_on(False)
    asb.set_in_layout(True)

    ax.add_artist(asb)
    return asb
