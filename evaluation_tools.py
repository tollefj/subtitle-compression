from typing import Callable, List

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from config import COMPRESSION_RATIOS, OPUS_MT_MODELS, prefixes
from evaluation.evaluator import Evaluator
from utils.model_utils import load_baseline, load_model, translate_all


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


def get_metrics(
    languages: List[str],
    datagetter: Callable,
    n_samples: int = 1000,
    device: str = "cuda",
):
    metrics_by_lang = {}
    all_translations = {}

    for lang in languages:
        _metrics, _translations = get_language_metrics(
            source_lang="en",
            target_lang=lang,
            datagetter=datagetter,
            n_samples=n_samples,
            device=device,
        )
        metrics_by_lang[lang] = _metrics
        all_translations[lang] = _translations

    return metrics_by_lang, all_translations


def get_language_metrics(
    source_lang,
    target_lang,
    datagetter,
    n_samples=1000,
    device="cuda",
):
    lang1 = source_lang
    lang = target_lang
    print(f"Getting metrics for {lang}")

    evaluator = Evaluator()

    model_id = OPUS_MT_MODELS[lang]
    baseline_model, baseline_tokenizer = load_baseline(model_id, device=device)

    test_df = datagetter(source_lang, lang)
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
    sample_size = min(n_samples, len(test_df))
    test_df = test_df.sample(n=sample_size)

    tar_texts = test_df[lang].tolist()
    src_texts = test_df[lang1].tolist()
    prefix = prefixes.get(lang, "")
    src_texts = [f"{prefix} {s}".strip() for s in src_texts]

    baseline_translations = translate_all(
        src_texts,
        baseline_model,
        baseline_tokenizer,
        device=device,
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


# if run with bootstrap sampling
# combined = {}  #
# for run in all_metrics:
#     for key in all_metrics[run]:
#         if key not in combined:
#             combined[key] = {}
#         for metric in all_metrics[run][key]:
#             if metric not in combined[key]:
#                 combined[key][metric] = []
#             combined[key][metric].append(all_metrics[run][key][metric])

# avg = {}
# std = {}

# for key in combined:
#     avg[key] = {}
#     std[key] = {}
#     for metric in combined[key]:
#         avg[key][metric] = np.round(np.mean(combined[key][metric]), 2)
#         std[key][metric] = np.round(np.std(combined[key][metric]), 2)

# # combine the average and std into a single dataframe
# avg_df = pd.DataFrame(avg).T
# # do a reverse-weighing based on len_ratio
# # all metrics should be divided by len_ratio
# # for row in avg_df.index:
# #     avg_df.loc[row] = avg_df.loc[row] / avg_df.loc[row]["len_ratio"]
# #     avg_df.loc[row] = np.round(avg_df.loc[row], 2)
# avg_df = avg_df.sort_values(by="meteor", ascending=False)
# avg_df
