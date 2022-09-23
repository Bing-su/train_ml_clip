from itertools import chain
from typing import Literal

import pytorch_lightning as pl
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
)

from .util import create_optimizer


class KoCLIPModule(pl.LightningModule):
    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        model_type: Literal["clip", "dual_encoder"] = "dual_encoder",
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        max_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # init model
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.model_type = model_type
        self.teacher, self.student = self.init_model(
            teacher_model_name, student_model_name
        )

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.weight_decay = weight_decay

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = CLIPModel.from_pretrained(teacher_model_name)

        if self.model_type == "clip":
            student = CLIPModel.from_pretrained(student_model_name)
        else:
            student = VisionTextDualEncoderModel.from_vision_text_pretrained(
                teacher_model_name, student_model_name
            )

            vp_state = teacher.visual_projection.state_dict()
            student.visual_projection.load_state_dict(vp_state)
            student.logit_scale = teacher.logit_scale

        return teacher, student

    def configure_optimizers(self):
        params = list(
            chain(
                self.student.text_model.named_parameters(),
                self.student.text_projection.named_parameters(),
            )
        )

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt_class = create_optimizer(self.optimizer)
        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.max_lr, total_steps=self.trainer.estimated_stepping_batches
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        ko_batch, en_ko_batch, en_en_batch = batch

        ko_emb = self.student.get_text_features(**ko_batch)
        en_ko_emb = self.student.get_text_features(**en_ko_batch)
        en_en_emb = self.teacher.get_text_features(**en_en_batch)

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_ko": ko_en_loss,
                "train/loss_en": en_en_loss,
            },
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ko_batch, en_ko_batch, en_en_batch = batch

        ko_emb = self.student.get_text_features(**ko_batch)
        en_ko_emb = self.student.get_text_features(**en_ko_batch)
        en_en_emb = self.teacher.get_text_features(**en_en_batch)

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        self.log_dict(
            {"val/loss": loss, "val/loss_ko": ko_en_loss, "val/loss_en": en_en_loss},
            on_epoch=True,
        )
        return loss

    def save(self, save_dir: str = "save/my_model"):
        self.student.save_pretrained(save_dir)

        tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)

        if self.model_type == "clip":
            processor = CLIPProcessor.from_pretrained(self.student_model_name)
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.teacher_model_name
            )
            processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        processor.save_pretrained(save_dir)
