import pytorch_lightning as pl
import torch
from pytorch_optimizer import MADGRAD
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    CLIPModel,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
)


class KoCLIPModule(pl.LightningModule):
    def __init__(
        self,
        clip_model_name: str,
        text_model_name: str,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.clip_model_name = clip_model_name
        self.text_model_name = text_model_name
        self.clip, self.model = self.init_model(clip_model_name, text_model_name)
        self.mse = torch.nn.MSELoss()
        self.learning_rate = learning_rate

    def init_model(
        self, clip_model_name: str, text_model_name: str
    ) -> tuple[CLIPModel, VisionTextDualEncoderModel]:
        clip = CLIPModel.from_pretrained(clip_model_name)
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            clip_model_name, text_model_name
        )

        vp_state = clip.visual_projection.state_dict()
        model.visual_projection.load_state_dict(vp_state)

        model.logit_scale = clip.logit_scale
        return clip, model

    def configure_optimizers(self):
        optimizer = MADGRAD(
            [
                {"params": self.model.text_model.parameters()},
                {"params": self.model.text_projection.parameters()},
            ],
            lr=self.learning_rate,
            weight_decay=1e-4,
            decouple_decay=True,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 0.01, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        ko_batch, en_ko_batch, en_en_batch = batch

        ko_emb = self.model.get_text_features(**ko_batch)
        en_ko_emb = self.model.get_text_features(**en_ko_batch)
        en_en_emb = self.clip.get_text_features(**en_en_batch)

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        self.log_dict(
            {"train_loss": loss, "ko_loss": ko_en_loss, "en_loss": en_en_loss},
            prog_bar=True,
            on_step=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ko_batch, en_ko_batch, en_en_batch = batch

        ko_emb = self.model.get_text_features(**ko_batch)
        en_ko_emb = self.model.get_text_features(**en_ko_batch)
        en_en_emb = self.clip.get_text_features(**en_en_batch)

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        self.log_dict(
            {"val_loss": loss, "ko_loss": ko_en_loss, "en_loss": en_en_loss},
            on_step=True,
            on_epoch=True,
        )
        return loss

    def save(self, save_dir: str = "save/my_model"):
        self.model.save_pretrained(save_dir)

        feature_extractor = AutoFeatureExtractor.from_pretrained(self.clip_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        processor.save_pretrained(save_dir)
