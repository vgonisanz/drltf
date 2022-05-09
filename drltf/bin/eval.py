"""
Evaluate an existing model and generate a report
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core
from drltf.entities import EvaluationConfig


def main(n_envs: int = 1,
         model_name: str = "vgoni-model",
         models_path: str = "models",
         render: bool = True):

    typer.echo(f"Running {__file__}")

    evaluation_config = EvaluationConfig(
        n_eval_episodes=10,
        deterministic=True
    )
    core = Core(n_envs=n_envs, models_path=models_path)
    core.load_model(model_name)
    core.evaluate(evaluation_config)

    if render:
        core.render()


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
