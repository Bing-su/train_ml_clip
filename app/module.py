from itertools import chain
from typing import Literal

import pytorch_lightning as pl
import torch
from loguru import logger
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
        weight_decay: float = 1e-4,
        use_auth_token: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # init model
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.model_type = model_type
        self.use_auth_token = use_auth_token
        self.teacher, self.student = self.init_model(
            teacher_model_name, student_model_name
        )

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = CLIPModel.from_pretrained(
            teacher_model_name, use_auth_token=self.use_auth_token
        )

        if self.model_type == "clip":
            student = CLIPModel.from_pretrained(
                student_model_name, use_auth_token=self.use_auth_token
            )
        else:
            student = VisionTextDualEncoderModel.from_vision_text_pretrained(
                teacher_model_name, student_model_name
            )

            vp_state = teacher.visual_projection.state_dict()
            student.visual_projection.load_state_dict(vp_state)
            student.logit_scale = teacher.logit_scale

        return teacher, student

    def configure_optimizers(self):
        if self.model_type == "clip":
            params = list(self.student.text_model.named_parameters())
        else:
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
            lr=5e-5,
        )

        if "bnb" in self.optimizer:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            modules = chain(
                self.student.text_model.modules(),
                self.student.text_projection.modules(),
            )
            for module in modules:
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def step(self, batch):
        ko_batch, en_ko_batch, en_en_batch = batch

        if self.model_type == "clip":
            ko_emb = self.student.text_model(**ko_batch)[1]
            en_ko_emb = self.student.text_model(**en_ko_batch)[1]
            en_en_emb = self.teacher.text_model(**en_en_batch)[1]
        else:
            ko_emb = self.student.get_text_features(**ko_batch)
            en_ko_emb = self.student.get_text_features(**en_ko_batch)
            en_en_emb = self.teacher.get_text_features(**en_en_batch)

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        loss_dict = {
            "loss": loss,
            "loss_ko": ko_en_loss,
            "loss_en": en_en_loss,
        }

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log_dict(
            {
                "train/loss": loss["loss"],
                "train/loss_ko": loss["loss_ko"],
                "train/loss_en": loss["loss_en"],
            },
            on_step=True,
            on_epoch=True,
        )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log_dict(
            {
                "val/loss": loss["loss"],
                "val/loss_ko": loss["loss_ko"],
                "val/loss_en": loss["loss_en"],
            },
            on_epoch=True,
        )
        return loss["loss"]

    def save(self, save_dir: str = "save/my_model"):
        self.student.save_pretrained(save_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_name, use_auth_token=self.use_auth_token
        )

        if self.model_type == "clip":
            processor = CLIPProcessor.from_pretrained(
                self.student_model_name, use_auth_token=self.use_auth_token
            )
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.teacher_model_name, use_auth_token=self.use_auth_token
            )
            processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        processor.save_pretrained(save_dir)
