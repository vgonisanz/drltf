"""
GUI sample to create interactive interfaces

INCOMPLETE CODE
"""
import sys
import pyglet
from pyglet import gl

import imgui
from imgui.integrations.pyglet import create_renderer


def main():
    window = pyglet.window.Window(width=1280, height=720, resizable=True)
    gl.glClearColor(1, 1, 1, 1)
    imgui.create_context()
    impl = create_renderer(window)

    def update(dt):
        # pylint: disable=unused-argument

        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, _ = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    sys.exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Custom window", True)
        imgui.text("Bar")
        imgui.text_colored("Eggs", 0.2, 1., 0.)

        imgui.text_ansi("B\033[31marA\033[mnsi ")
        imgui.text_ansi_colored("Eg\033[31mgAn\033[msi ", 0.2, 1., 0.)

        imgui.end()

    def draw(dt):
        update(dt)
        window.clear()
        imgui.render()
        impl.render(imgui.get_draw_data())

    pyglet.clock.schedule_interval(draw, 1/120.)
    pyglet.app.run()
    impl.shutdown()


if __name__ == "__main__":
    main()
