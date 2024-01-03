from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Figure


class StringPresenter:
    def __init__(self, fig: Figure, update_plot: callable, max_line: int) -> None:
        self.fig = fig
        self.update_plot = update_plot
        self.timer = None
        self.line_idx = 0
        self.max_line = max_line
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def on_key(self, event: KeyEvent):
        if event.key == 'right':
            self.line_idx = (self.line_idx + 1) % (self.max_line+1)
        elif event.key == 'left':
            self.line_idx = (self.line_idx - 1) % (self.max_line+1)

        if self.timer:
            return

        self.timer = self.fig.canvas.new_timer(interval=1)
        self.timer.add_callback(lambda: self.update_plot(self.line_idx))
        self.timer.start()

    def on_key_release(self, event: KeyEvent):
        if not self.timer:
            return
        self.timer.stop()
        self.timer = None
