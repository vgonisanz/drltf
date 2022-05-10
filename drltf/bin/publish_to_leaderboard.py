"""
Script to backup a model in HuggingFaces user repository.

Require a repository created with {whoami['name']}/{model_architecture}-{env_id}.
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core


def main(commit_message: str,
         env_id: str = "LunarLander-v2",
         model_architecture: str = "PPO",
         model_name: str = "vgoni-model",
         models_path: str = "models",
         n_envs:int = 1):

    typer.echo(f"Running {__file__}")

    core = Core(n_envs=n_envs, models_path=models_path)
    whoami = core.whoami()

    iam = typer.confirm(
        f"Are you '{whoami['name']}'?")
    if not iam:
        typer.echo("Publish aborted by user missmatch")
        typer.Exit(0)

    repo_id = f"{whoami['name']}/{model_architecture}-{env_id}"

    send = typer.confirm(
        f"Are you sure you want to Backup model '{model_name}' at HuggingFace in {repo_id} ?")
    if send:
        typer.echo("Publishing...")
        core.publish(env_id, model_name, model_architecture, repo_id, commit_message)
        #typer.echo(f"Check the backup of the model at https://huggingface.co/{repo_id}")
    else:
        typer.echo("Publish aborted")


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
