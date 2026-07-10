from __future__ import annotations

from matplotlib import font_manager, rcParams


def configure_matplotlib(font_size: int = 12) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    chosen = next((name for name in preferred if name in available), "DejaVu Serif")
    rcParams.update(
        {
            "font.family": chosen,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": max(8, font_size - 1),
            "ytick.labelsize": max(8, font_size - 1),
            "legend.fontsize": max(8, font_size - 1),
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return chosen
