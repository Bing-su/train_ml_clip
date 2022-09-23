from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import BatchEncoding


class KoCLIPDataset(Dataset):
    def __init__(
        self,
        ds: HFDataset,
        en_tokenizer: PreTrainedTokenizerFast,
        ko_tokenizer: PreTrainedTokenizerFast,
    ):
        self.ds = ds
        self.en_tokenizer = en_tokenizer
        self.ko_tokenizer = ko_tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx: int
    ) -> tuple["BatchEncoding", "BatchEncoding", "BatchEncoding"]:
        ko: str = self.ds[idx]["ko"]
        en: str = self.ds[idx]["en"]

        ko_token = self.ko_tokenizer(ko, truncation=True)
        en_ko_token = self.ko_tokenizer(en, truncation=True)
        en_en_token = self.en_tokenizer(en, truncation=True)

        return ko_token, en_ko_token, en_en_token


class KoCLIPDataCollator:
    def __init__(
        self,
        en_tokenizer: PreTrainedTokenizerFast,
        ko_tokenizer: PreTrainedTokenizerFast,
    ):
        self.en_tokenizer = en_tokenizer
        self.ko_tokenizer = ko_tokenizer

    def __call__(
        self,
        features: Sequence[tuple["BatchEncoding", "BatchEncoding", "BatchEncoding"]],
    ):
        ko_token, en_ko_token, en_en_token = zip(*features)
        ko_batch = self.ko_tokenizer.pad(ko_token, padding=True, return_tensors="pt")
        en_ko_batch = self.ko_tokenizer.pad(
            en_ko_token, padding=True, return_tensors="pt"
        )
        en_en_batch = self.en_tokenizer.pad(
            en_en_token, padding=True, return_tensors="pt"
        )

        return ko_batch, en_ko_batch, en_en_batch


class KoCLIPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        en_tokenizer_name: str,
        ko_tokenizer_name: str,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.en_tokenizer_name = en_tokenizer_name
        self.ko_tokenizer_name = ko_tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        load_dataset(
            "Bingsu/aihub_ko-en_parallel_corpus_collection",
            split="train+validation",
            use_auth_token=True,
        )

    def setup(self, stage=None):
        ds: HFDataset = load_dataset(
            "Bingsu/aihub_ko-en_parallel_corpus_collection",
            split="train+validation",
            use_auth_token=True,
        )
        en_tokenizer = AutoTokenizer.from_pretrained(self.en_tokenizer_name)
        ko_tokenizer = AutoTokenizer.from_pretrained(self.ko_tokenizer_name)
        self.data_collator = KoCLIPDataCollator(en_tokenizer, ko_tokenizer)

        self.train_dataset = KoCLIPDataset(ds, en_tokenizer, ko_tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )


if __name__ == "__main__":
    ds = load_dataset(
        "Bingsu/aihub_ko-en_parallel_corpus_collection",
        split="train+validation",
        use_auth_token=True,
    )

    en_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    ko_tokenizer = AutoTokenizer.from_pretrained("lassl/roberta-ko-small")

    dataset = KoCLIPDataset(ds, en_tokenizer, ko_tokenizer)

    collate_fn = KoCLIPDataCollator(en_tokenizer, ko_tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    for batch in loader:
        print(batch)
        break
