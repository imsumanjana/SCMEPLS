from __future__ import annotations

from dataclasses import dataclass, asdict
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure


@dataclass
class PlotStyle:
    """User-adjustable publication style for Matplotlib figures."""

    font_family: str = "Times New Roman"
    axis_label_size: int = 12
    tick_label_size: int = 11
    legend_size: int = 10
    annotation_size: int = 11
    title_size: int = 13
    bold_axis_labels: bool = False
    bold_tick_labels: bool = False
    bold_legends: bool = False
    bold_annotations: bool = False
    bold_titles: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlotStyle":
        if not data:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def available_font_families() -> list[str]:
    """Return a compact, sorted font list with common publication fonts first."""
    fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
    preferred = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "Arial", "Calibri", "Cambria"]
    ordered: list[str] = []
    for name in preferred:
        if name in fonts and name not in ordered:
            ordered.append(name)
    for name in fonts:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _resolve_font(font_family: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    if font_family in available:
        return font_family
    for fallback in ["Times New Roman", "Liberation Serif", "DejaVu Serif"]:
        if fallback in available:
            return fallback
    return "DejaVu Serif"


def configure_matplotlib(font_size: int = 12, style: PlotStyle | None = None) -> str:
    """Configure default Matplotlib rcParams for newly created figures."""
    style = style or PlotStyle(axis_label_size=font_size, tick_label_size=max(8, font_size - 1), legend_size=max(8, font_size - 1), title_size=font_size + 1, annotation_size=max(8, font_size - 1))
    chosen = _resolve_font(style.font_family)
    rcParams.update(
        {
            "font.family": chosen,
            "font.size": style.axis_label_size,
            "axes.labelsize": style.axis_label_size,
            "axes.titlesize": style.title_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_size,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return chosen


def apply_plot_style(figure: Figure, style: PlotStyle) -> None:
    """Apply a user-selected style to an existing figure.

    This function updates current axes labels, tick labels, legends, titles,
    annotations, polar labels, and figure-level text. It is used by the GUI
    toolbar so exported PNG files keep the same publication style.
    """
    chosen = _resolve_font(style.font_family)
    axis_weight = "bold" if style.bold_axis_labels else "normal"
    tick_weight = "bold" if style.bold_tick_labels else "normal"
    legend_weight = "bold" if style.bold_legends else "normal"
    annotation_weight = "bold" if style.bold_annotations else "normal"
    title_weight = "bold" if style.bold_titles else "normal"

    for ax in figure.axes:
        title = ax.title
        title.set_fontfamily(chosen)
        title.set_fontsize(style.title_size)
        title.set_fontweight(title_weight)

        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontfamily(chosen)
            label.set_fontsize(style.axis_label_size)
            label.set_fontweight(axis_weight)

        # z-axis label for any future 3-D axes; harmless for normal axes.
        if hasattr(ax, "zaxis"):
            zlabel = ax.zaxis.label
            zlabel.set_fontfamily(chosen)
            zlabel.set_fontsize(style.axis_label_size)
            zlabel.set_fontweight(axis_weight)

        ax.tick_params(axis="both", which="major", labelsize=style.tick_label_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(chosen)
            tick_label.set_fontsize(style.tick_label_size)
            tick_label.set_fontweight(tick_weight)
        if hasattr(ax, "get_zticklabels"):
            for tick_label in ax.get_zticklabels():
                tick_label.set_fontfamily(chosen)
                tick_label.set_fontsize(style.tick_label_size)
                tick_label.set_fontweight(tick_weight)

        # Text annotations created with ax.text(), bar labels, radar labels, etc.
        for txt in ax.texts:
            txt.set_fontfamily(chosen)
            txt.set_fontsize(style.annotation_size)
            txt.set_fontweight(annotation_weight)

        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontfamily(chosen)
                txt.set_fontsize(style.legend_size)
                txt.set_fontweight(legend_weight)
            if legend.get_title() is not None:
                legend.get_title().set_fontfamily(chosen)
                legend.get_title().set_fontsize(style.legend_size)
                legend.get_title().set_fontweight(legend_weight)

    for txt in figure.texts:
        txt.set_fontfamily(chosen)
        txt.set_fontsize(style.annotation_size)
        txt.set_fontweight(annotation_weight)

    figure.stale = True
