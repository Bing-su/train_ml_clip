import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class KoCLIPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        en_tokenizer: PreTrainedTokenizerBase,
        ko_tokenizer: PreTrainedTokenizerBase,
    ):
        self.df = df.reset_index(drop=True)
        self.en_tokenizer = en_tokenizer
        self.ko_tokenizer = ko_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        ko = self.df.loc[idx, "원문"]
        en = self.df.loc[idx, "번역문"]

        ko_token = self.ko_tokenizer(ko, truncation=True)
        en_ko_token = self.ko_tokenizer(en, truncation=True)
        en_en_token = self.en_tokenizer(en, truncation=True)

        return ko_token, en_ko_token, en_en_token


class KoCLIPDataCollator:
    def __init__(
        self,
        en_tokenizer: PreTrainedTokenizerBase,
        ko_tokenizer: PreTrainedTokenizerBase,
    ):
        self.en_tokenizer = en_tokenizer
        self.ko_tokenizer = ko_tokenizer

    def __call__(self, features):
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
    ):
        super().__init__()
        self.en_tokenizer_name = en_tokenizer_name
        self.ko_tokenizer_name = ko_tokenizer_name
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_csv("data/data.tsv", sep="\t")
        en_tokenizer = AutoTokenizer.from_pretrained(self.en_tokenizer_name)
        ko_tokenizer = AutoTokenizer.from_pretrained(self.ko_tokenizer_name)
        self.data_collator = KoCLIPDataCollator(en_tokenizer, ko_tokenizer)

        train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
        self.train_dataset = KoCLIPDataset(train_df, en_tokenizer, ko_tokenizer)
        self.val_dataset = KoCLIPDataset(val_df, en_tokenizer, ko_tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=8,
            pin_memory=True,
        )


if __name__ == "__main__":
    df = pd.read_csv("data/data.tsv", sep="\t")

    en_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    ko_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")

    dataset = KoCLIPDataset(df, en_tokenizer, ko_tokenizer)

    collate_fn = KoCLIPDataCollator(en_tokenizer, ko_tokenizer)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    for batch in loader:
        print(batch)
        break
