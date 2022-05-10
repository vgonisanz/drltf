"""
Script to backup a model in HuggingFaces user repository.

Require a repository created with {whoami['name']}/{repo_name}.
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core


def main(
         repo_name: str = "lunar-lander-models",
         model_name: str = "vgoni-model",
         models_path: str = "models"):

    typer.echo(f"Running {__file__}")

    core = Core()
    whoami = core.whoami()

    iam = typer.confirm(
        f"Are you '{whoami['name']}'?")
    if not iam:
        typer.echo("Backup aborted by user missmatch")
        typer.Exit(0)

    repo_id = f"{whoami['name']}/{repo_name}"

    send = typer.confirm(
        f"Are you sure you want to Backup model '{model_name}' at HuggingFace in {repo_id} ?")
    if send:
        typer.echo("Creating backup...")
        core.backup(models_path, model_name, repo_id)
        typer.echo(f"Check the backup of the model at https://huggingface.co/{repo_id}")
    else:
        typer.echo("Backup aborted")


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
