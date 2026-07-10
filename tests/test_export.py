from matplotlib.figure import Figure
from scmepls_studio.io.export import export_figure


def test_export_600_dpi(tmp_path):
    fig = Figure(); ax = fig.add_subplot(111); ax.plot([0,1],[0,1])
    png, meta = export_figure(fig, tmp_path / "test.png", 600, {"provenance": "Simulation"})
    assert png.exists()
    assert meta.exists()
