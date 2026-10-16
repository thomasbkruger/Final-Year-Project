"""figstyle.py - type sizes and export settings for figures that go in the report.

The report's text block is 160 mm wide (A4, 25 mm margins), so figures are
drawn at about that width, with type a little larger than the body text. They
then read at true size when included at \\linewidth, and stay legible when
LaTeX scales them down to sit beside text or in a subfigure.

Saved tight to the drawn content, so no page width is spent on empty margin.
"""
from __future__ import annotations

TEXT_WIDTH = 6.3        # in; A4 less two 25 mm margins

FS_LABEL = 13           # axis labels
FS_TITLE = 12.5         # panel titles, where a figure has several panels
FS_TICK = 11.5
FS_LEGEND = 11
FS_NOTE = 10.5          # in-plot annotations, and legends in tight spaces


def use() -> None:
    """Apply the report sizes to every figure drawn after this call."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
