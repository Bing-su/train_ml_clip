from transformers import AutoModel, AutoProcessor
from typer import Argument, Option, Typer

cli = Typer()


@cli.command(no_args_is_help=True)
def upload(
    model_id: str = Argument(
        ..., help="huggingface repository name", show_default=False
    ),
    model_path: str = Option(
        ..., help="path of model's directory to upload", show_default=False
    ),
    processor: bool = Option(True, help="upload processor too"),
    private: bool = Option(False, help="make the repository private"),
):
    model = AutoModel.from_pretrained(model_path)
    model.push_to_hub(model_id, private=private, use_auth_token=private)
    if processor:
        model_processor = AutoProcessor.from_pretrained(model_path)
        model_processor.push_to_hub(model_id, private=private, use_auth_token=private)


if __name__ == "__main__":
    cli()
