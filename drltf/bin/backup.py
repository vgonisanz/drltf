"""
Script to backup a model in HuggingFaces user repository.

Require a repository created with {whoami['name']}/{repo_name}.
"""
import os
import typer

from huggingface_hub import HfApi

from drltf.utils import setup_logger


def main(
         repo_name: str = "lunar-lander-models",
         model_name: str = "vgoni-model",
         models_path: str = "models"):

    typer.echo(f"Running {__file__}")

    api = HfApi()
    whoami = api.whoami()

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
        files = [
            os.path.join(models_path, model_name, "model.zip"),
            os.path.join(models_path, model_name, "model_eval_report.json"),
            os.path.join(models_path, model_name, "model_train_report.json")
        ]
        for file in files:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id
            )
        typer.echo(f"Check the backup of the model at https://huggingface.co/{repo_id}")
    else:
        typer.echo("Backup aborted")


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
