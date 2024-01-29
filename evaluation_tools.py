import numpy as np
import pandas as pd
from IPython.display import HTML, display

from config import COMPRESSION_RATIOS, get_prefix
from utils.model_utils import load_model, translate_all


def filter_metrics(metrics):
    ignored = ["r2", "rl", "chrf++", "bleu"]
    return {
        k: v
        for k, v in metrics.items()
        if not any(k.startswith(i) for i in ignored)
    }


def post_cleanup(df):
    cleanup = [c for c in df.columns if "len" in c and c != "len_ratio"]
    return df.drop(columns=cleanup)


def print_metrics(_lang, _metrics):
    tmpdf = pd.DataFrame(_metrics).T
    tmpdf = tmpdf[
        [c for c in tmpdf.columns if c != "len_ratio"] + ["len_ratio"]
    ]
    tmpdf = post_cleanup(tmpdf)
    normed_df = tmpdf / tmpdf.max(axis=0)
    normed_sum = normed_df.sum(axis=1)
    normed_sum = (normed_sum - normed_sum.min()) / (
        normed_sum.max() - normed_sum.min()
    )
    tmpdf["normalized_score"] = normed_sum.round(2)
    len_ratio = tmpdf["len_ratio"].copy()

    display(HTML(f"<h1>{_lang} - metrics</h1>"))
    display(tmpdf)

    for row in tmpdf.index:
        tmpdf.loc[row] = tmpdf.loc[row] / tmpdf.loc[row]["len_ratio"]
        tmpdf.loc[row] = np.round(tmpdf.loc[row], 2)
    tmpdf["len_ratio"] = len_ratio
    normed_df = tmpdf / tmpdf.max(axis=0)
    normed_sum = normed_df.sum(axis=1)
    normed_sum = (normed_sum - normed_sum.min()) / (
        normed_sum.max() - normed_sum.min()
    )
    tmpdf["normalized_score"] = normed_sum.round(2)

    display(HTML(f"<h1>{_lang} - weighted metrics (length)</h1>"))
    display(tmpdf)


def get_language_metrics(
    source_lang,
    target_lang,
    datagetter,
    baseline_model,
    baseline_tokenizer,
    evaluator,
    n_samples=1000,
):
    lang1 = source_lang
    lang = target_lang
    print(f"Getting metrics for {lang}")

    test_df = datagetter(source_lang, lang)
    test_df = test_df.sample(n=n_samples)
    ###### Normalize
    src_len = test_df[lang1].str.len()
    tgt_len = test_df[lang].str.len()
    mean_len = (tgt_len / src_len).mean()
    print(f"Mean length ratio: {mean_len}")
    ###### Compress < original length times the mean
    print(f"Size of test set: {len(test_df)}")
    test_df = test_df[
        test_df[lang].str.len() < mean_len * test_df[lang1].str.len()
    ]
    print(f"Size of test set after compression: {len(test_df)}")
    tar_texts = test_df[lang].tolist()
    src_texts = test_df[lang1].tolist()
    prefix = get_prefix(lang)
    src_texts = [f"{prefix} {s}".strip() for s in src_texts]

    baseline_translations = translate_all(
        src_texts,
        baseline_model,
        baseline_tokenizer,
        device="cuda",
        batch_size=64,
    )

    baseline_metrics = evaluator.get_metrics(
        baseline_translations, tar_texts, lang=lang
    )
    metrics = {"baseline": baseline_metrics}
    translations = {
        "source": src_texts,
        "target": tar_texts,
        "baseline": baseline_translations,
    }

    for compression_ratio in COMPRESSION_RATIOS:
        print(
            f"Loading model trained on compression rate: {compression_ratio}"
        )
        path = f"OUTPUT/{lang1}-{lang}/{compression_ratio}"
        model, tokenizer = load_model(path)
        translated = translate_all(src_texts, model, tokenizer)
        translations[compression_ratio] = translated

        comp_metrics = evaluator.get_metrics(translated, tar_texts, lang)
        metrics[compression_ratio] = comp_metrics

    return metrics, translations
