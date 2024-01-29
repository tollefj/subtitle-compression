import json
import os
import random
import re
import subprocess
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import swifter
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)


def clean(text):
    # initial dialogue hyphens
    text = re.sub(r"^-+", "", text)
    # whitespaces
    text = re.sub(r"\s+", " ", text)
    # remove any repeated hyphens
    text = re.sub(r"(-\s*){2,}", "", text)
    # text = re.sub(r'-+', '', text)
    return text.strip()


def create_csv(lang1, lang2):
    datapath = "datasets"
    csv_file = f"{datapath}/{lang1}-{lang2}.csv"
    # if it exists alrdy:
    if os.path.exists(csv_file):
        print(f"File {csv_file} already exists. Compressing:")
        return csv_file

    parallel_data = defaultdict(list)
    for lang in [lang1, lang2]:
        fp = f"{datapath}/{lang1}-{lang2}.{lang}"
        with open(fp, "r") as f:
            parallel_data[lang] = f.readlines()
        os.remove(fp)
    assert len(parallel_data[lang1]) == len(parallel_data[lang2])
    df = pd.DataFrame.from_dict(parallel_data).dropna()
    with open(f"{datapath}/{lang1}-{lang2}.info", "w") as f:
        f.write(f"Number of rows: {len(df)}\n")
        f.write(
            f"Number of unique {lang1} sentences: {len(df[lang1].unique())}\n"
        )
        f.write(
            f"Number of unique {lang2} sentences: {len(df[lang2].unique())}\n"
        )

    print("Cleaning data...")
    df = df.swifter.applymap(clean)
    print(f"Saving to {csv_file}")
    df.to_csv(csv_file, index=False)
    return csv_file


def compress(df, lang1, lang2, MAX_SIZE=400_000):
    os.makedirs("compressed", exist_ok=True)
    compressions = {}

    # find the mean length ratio between the two languages
    src_len = df[lang1].str.len()
    tgt_len = df[lang2].str.len()
    mean_len = (tgt_len / src_len).mean()

    compressions["length_ratio"] = mean_len
    print(f"Length ratio between {lang1} and {lang2}: {mean_len}")

    for comp_rate in np.arange(0.5, 1.05, 0.1):
        comp_rate = round(comp_rate, 1)
        compressed_file = f"compressed/{lang1}-{lang2}-{comp_rate}.csv"
        if os.path.exists(compressed_file):
            print(f"File {compressed_file} already exists. Skipping...")
            continue
        print(f"Compression rate: {comp_rate}")
        # OLD: compressed = df[df[lang2].str.len() < comp_rate * df[lang1].str.len()]
        # NEW: compressed based on mean length ratio weighted by comp_rate
        compressed = df[
            df[lang2].str.len() < mean_len * comp_rate * df[lang1].str.len()
        ]
        print(compressed.describe())
        # reduce to max_size
        reduced = compressed.sample(
            n=min(MAX_SIZE, compressed.shape[0]), random_state=42
        )
        # reduced.to_csv(compressed_file, index=False)
        train, test = train_test_split(reduced, test_size=0.2, random_state=42)
        test, val = train_test_split(test, test_size=0.5, random_state=42)
        filename = f"compressed/{lang1}-{lang2}-{comp_rate}"
        train.to_csv(f"{filename}.train.csv", index=False)
        test.to_csv(f"{filename}.test.csv", index=False)
        val.to_csv(f"{filename}.val.csv", index=False)

        compressions[comp_rate] = {
            "compression_rate": comp_rate,
            "count": len(compressed),
            f"unique_{lang1}": compressed[lang1].unique().shape[0],
            f"unique_{lang2}": compressed[lang2].unique().shape[0],
            "count_reduced": len(reduced),
            f"unique_{lang1}_reduced": reduced[lang1].unique().shape[0],
            f"unique_{lang2}_reduced": reduced[lang2].unique().shape[0],
        }
    with open(f"compressed/{lang1}-{lang2}.info", "w") as f:
        json.dump(compressions, f, indent=4)
    # also create a randomly sampled dataset of MAX_SIZE
    n = min(MAX_SIZE, df.shape[0])
    random_sample = df.sample(n=n, random_state=42)
    train, test = train_test_split(
        random_sample, test_size=0.2, random_state=42
    )
    test, val = train_test_split(test, test_size=0.5, random_state=42)
    # random_sample.to_csv(f"compressed/{lang1}-{lang2}-random.csv", index=False)
    train.to_csv(f"compressed/{lang1}-{lang2}-random.train.csv", index=False)
    test.to_csv(f"compressed/{lang1}-{lang2}-random.test.csv", index=False)
    val.to_csv(f"compressed/{lang1}-{lang2}-random.val.csv", index=False)


def download(lang1, lang2, version="v2016"):
    # if the file exists in datasets:
    if os.path.exists(f"./datasets/{lang1}-{lang2}.csv"):
        print(
            f"Language pair ({lang1}, {lang2}) already exists. Checking if compressed files need to be created:"
        )
        return create_csv(lang1, lang2)

    url = f"https://opus.nlpl.eu/download.php?f=OpenSubtitles/{version}/moses/{lang1}-{lang2}.txt.zip"
    print(f"Downloading from {url}")
    cmd = f"wget {url} -O {lang1}-{lang2}.txt.zip"
    subprocess.run(cmd, shell=True)

    os.makedirs("./datasets", exist_ok=True)
    with zipfile.ZipFile(f"{lang1}-{lang2}.txt.zip", "r") as zip_ref:
        zip_ref.extractall("./datasets")

    os.remove(f"{lang1}-{lang2}.txt.zip")
    os.remove(f"./datasets/OpenSubtitles.{lang1}-{lang2}.ids")
    os.rename(
        f"./datasets/OpenSubtitles.{lang1}-{lang2}.{lang1}",
        f"./datasets/{lang1}-{lang2}.{lang1}",
    )
    os.rename(
        f"./datasets/OpenSubtitles.{lang1}-{lang2}.{lang2}",
        f"./datasets/{lang1}-{lang2}.{lang2}",
    )

    return create_csv(lang1, lang2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_langs.py <lang1> <lang2>")
        sys.exit(1)
    lang1, lang2 = sys.argv[1], sys.argv[2]
    path = download(lang1, lang2)
    compress(pd.read_csv(path).dropna(), lang1, lang2)
