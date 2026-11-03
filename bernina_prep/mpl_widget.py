"""A matplotlib figure as an ipywidget, independent of the notebook's backend."""
from __future__ import annotations


class FigureWidget:
    """``.fig`` to draw into, ``.widget`` to place in a layout, ``.redraw()``
    after drawing. Uses an ipympl canvas of its own (zoom, hover readout)
    without switching the notebook's matplotlib backend; where ipympl is
    missing, a static image in an Output widget."""

    def __init__(self, figsize, **fig_kw):
        import ipywidgets as w
        from matplotlib.figure import Figure

        fig_kw.setdefault("layout", "constrained")
        self.fig = Figure(figsize=figsize, **fig_kw)
        self.canvas = None
        try:
            from ipympl.backend_nbagg import Canvas, FigureManager
            self.canvas = Canvas(self.fig)
            FigureManager(self.canvas, 0)
            self.canvas.header_visible = False
            self.canvas.toolbar_position = "top"
            self.widget = self.canvas
        except Exception:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(self.fig)
            self.widget = w.Output()

    def redraw(self):
        if self.canvas is not None:
            self.canvas.draw_idle()
            return
        from IPython.display import display
        self.widget.clear_output(wait=True)
        with self.widget:
            display(self.fig)
