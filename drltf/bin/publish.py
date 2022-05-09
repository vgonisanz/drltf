"""
Script to publish a model in HuggingFaces
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
        typer.echo("Publishing aborted by user missmatch")
        typer.Exit(0)

    model_to_upload = os.path.join(models_path, model_name, "model.zip")

    send = typer.confirm(
        f"Are you sure you want to publish model '{model_name}' at HuggingFace from ?")
    if send:
        typer.echo("Publishing...")
        files = [
            os.path.join(models_path, model_name, "model.zip"),
            os.path.join(models_path, model_name, "model_eval_report.json"),
            os.path.join(models_path, model_name, "model_train_report.json")
        ]
        for file in files:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=f"{whoami['name']}/{repo_name}"
            )
        typer.echo(f"Check the model at https://huggingface.co/{whoami['name']}/{repo_name}")
    else:
        typer.echo("Publishing aborted")


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
