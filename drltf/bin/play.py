"""
Try to play in the gym yourself
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core


def print_reward(*args):
    typer.echo(f"Current reward {args[3]}")


def main(fps: int = 20,
         zoom: int = 1):

    typer.echo(f"Running {__file__}")

    core = Core(n_envs=1)
    core.play(
        fps=fps,
        zoom=zoom,
        key_map={(ord('a'),): 1, (ord('w'),): 2, (ord('d'),): 3},
        print_reward=print_reward
    )


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
