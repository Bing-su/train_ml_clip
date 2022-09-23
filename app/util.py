import os
from platform import uname

import torch
from pytorch_optimizer import load_optimizer
from torch.optim import SGD, Adam, AdamW


def is_in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def create_optimizer(name: str):
    name = name.lower()

    if name == "adam":
        return Adam
    elif name == "adamw":
        return AdamW
    elif name == "sgd":
        return SGD
    elif name in ("adam_bnb", "adamw_bnb"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BNB optimizers")

        if is_in_wsl():
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib"

        try:
            from bitsandbytes.optim import Adam8bit, AdamW8bit

            if name == "adam_bnb":
                return Adam8bit
            else:
                return AdamW8bit

        except ImportError as e:
            raise ImportError("install bitsandbytes first") from e
    else:
        return load_optimizer(name)
