from loguru import logger
from typer import Argument, Option, Typer

from app.module import KoCLIPModule

cli = Typer()


@cli.command(no_args_is_help=True)
def convert(
    ckpt_path: str = Argument(
        ..., help="path of checkpoint to convert", show_default=False
    ),
    save_path: str = Option(
        "save/converted_model", help="path to save converted checkpoint"
    ),
):
    logger.info(f"Loading checkpoint from {ckpt_path}")
    module = KoCLIPModule.load_from_checkpoint(ckpt_path)
    module.save(save_path)
    logger.info(f"Converted checkpoint saved to {save_path}")


if __name__ == "__main__":
    cli()
