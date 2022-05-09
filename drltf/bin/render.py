"""
Render an IA agent to solve the gym problem
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core


def main(n_envs: int = 1,
         model_name: str = "vgoni-model",
         models_path: str = "models"):

    typer.echo(f"Running {__file__}")

    core = Core(n_envs=n_envs, models_path=models_path)
    core.load_model(model_name)
    core.render()


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
