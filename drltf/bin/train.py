"""
Train a new model with your awesome parameters
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core
from drltf.entities import TrainingConfig
from drltf.utils import notify_telegram


def main(steps: int = 5,
         n_envs: int = 1,
         save: bool = True,
         notify: bool = True,
         model_name: str = "vgoni-model",
         models_path: str = "models"):

    typer.echo(f"Running {__file__}")

    train_config = TrainingConfig(
        policy='MlpPolicy',
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.97,
        ent_coef=0.01
    )
    core = Core(n_envs=n_envs, models_path=models_path)
    core.train(train_config, steps)

    if save:
        core.save_model(model_name)
    
    if notify:
        notify_telegram(f"Train {model_name} completed!")


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
