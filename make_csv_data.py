import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def read_excel(path: Path):
    df = pd.read_excel(path, usecols=["원문", "번역문"])
    return df


if __name__ == "__main__":
    root = Path("data")

    zip_file = root / "3_문어체_뉴스(1).zip"
    if zip_file.exists():
        shutil.unpack_archive(zip_file, root)

    dfs = []
    excel_files = list(root.glob("*.xlsx"))

    with Pool() as pool, tqdm(excel_files) as pbar:
        for df in pool.map(read_excel, excel_files):
            dfs.append(df)
            pbar.update()

    df = pd.concat(dfs, axis=0)
    df.to_csv(root / "data.tsv", sep="\t", index=False)
