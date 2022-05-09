"""
Virtual windows wrapper to visualize gym. Work with 'with' statement.
"""
from pyvirtualdisplay import Display


class Window():
    def __init__(self, visible=True):
        self.virtual_display = Display(visible=visible, size=(1400, 900))

    def __enter__(self):
        self.virtual_display.start()
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.virtual_display.stop()
