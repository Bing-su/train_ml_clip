import pytorch_lightning as pl
from typer import Typer, Option
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint

from app.dataset import KoCLIPDataModule
from app.module import KoCLIPModule


cmd = Typer()


@cmd.command()
def train(
    clip_model_name: str = Option(
        "openai/clip-vit-base-patch32", "-c", "--clip", help="name of clip model"
    ),
    text_model_name: str = Option(
        "lassl/bert-ko-small", "-t", "--text", help="name of text model"
    ),
    learning_rate: float = Option(1e-3, "--lr", help="learning rate"),
    batch_size: int = Option(32, "-b", "--batch_size", min=1, help="batch_size"),
    patience: int = Option(
        3, min=0, help="earlystopping patience, if 0, deactivate early stopping"
    ),
    auto_scale_batch_size: bool = Option(
        False, help="auto find batch size, ignore batch_size option"
    ),
    auto_lr_find: bool = Option(False, help="auto find learning_rate"),
    test: bool = Option(False, help="do test run"),
):
    datamodule = KoCLIPDataModule(clip_model_name, text_model_name, batch_size)
    module = KoCLIPModule(clip_model_name, text_model_name, learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", every_n_train_steps=1000
    )

    callbacks = [checkpoint_callback, RichProgressBar(leave=True)]

    if patience:
        early_stop = EarlyStopping("val_loss", patience=patience)
        callbacks.append(early_stop)

    trainer = pl.Trainer(
        logger=WandbLogger(),
        fast_dev_run=test,
        enable_progress_bar=True,
        accelerator="auto",
        precision=16,
        max_steps=1_000_000,
        callbacks=callbacks,
        auto_scale_batch_size=auto_scale_batch_size,
        auto_lr_find=auto_lr_find,
    )

    trainer.fit(module, datamodule=datamodule)

    module.save()


if __name__ == "__main__":
    cmd()
