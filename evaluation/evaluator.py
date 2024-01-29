import evaluate
import numpy as np
import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.chrf = evaluate.load("chrf")
        self.meteor = evaluate.load("meteor")
        self.bert = evaluate.load("bertscore")
        self.bert_model = "bert-base-multilingual-cased"

    def get_metrics(
        self,
        predictions,
        references,
        lang=None,
        decimals=2,
        lead_n=None,
        ground=False,
        verbose=False,
    ):
        # ground the predictions to the length of the references
        metric_suffix = ""
        if ground:
            metric_suffix += "_gnd"
            # print("Grounding predictions to the length of the references")
            updated_preds = []
            for pred, ref in zip(predictions, references):
                pred = pred[: len(ref)]
                updated_preds.append(pred)
            predictions = updated_preds

        # If lead_n: only consider the first n tokens of each prediction
        if lead_n is not None:
            metric_suffix += f"_L{lead_n}"
            # print(
            #     f"Only considering the first {lead_n} tokens of each prediction"
            # )
            predictions = [pred[:lead_n] for pred in predictions]

        rouge_score = self.rouge.compute(
            predictions=predictions, references=references
        )
        bleu_score = self.bleu.compute(
            predictions=predictions, references=references
        )
        _chrf = {
            "regular": self.chrf.compute(
                predictions=predictions, references=references
            ),
            "++": self.chrf.compute(
                predictions=predictions, references=references, word_order=2
            ),
        }

        meteor = self.meteor.compute(
            predictions=predictions, references=references
        )["meteor"]

        bert_score_f1 = -1

        if lang:
            bert_score = self.bert.compute(
                predictions=predictions,
                references=references,
                lang=lang,
                model_type=self.bert_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
            bert_score_f1 = np.asarray(bert_score["f1"]).mean()

        if verbose:
            print("Rouge:", rouge_score)
            print("Bleu:", bleu_score)
            print("ChrF:", _chrf["regular"])
            print("ChrF++:", _chrf["++"])
            print("BertScore:")
            print()
        result = {
            "bleu": bleu_score["score"],
            "r1": rouge_score["rouge1"],
            "r2": rouge_score["rouge2"],
            "rl": rouge_score["rougeL"],
            "chrF": _chrf["regular"]["score"],
            "chrf++": _chrf["++"]["score"],
            "meteor": meteor,
            "bert_f1": bert_score_f1,
            "length": np.mean([len(pred) for pred in predictions]),
        }
        result = {k + metric_suffix: v for k, v in result.items()}

        # create a length penalty that penalizes longer translations
        # it should be *shorter or the same* than the length of the reference

        length_ratios = [
            len(pred) / len(ref) for pred, ref in zip(predictions, references)
        ]
        result["len_ratio" + metric_suffix] = np.mean(length_ratios)
        return {k: round(v, decimals) for k, v in result.items()}

    def visualize(
        self,
        predictions,
        references,
        source,
        baselines,
        description="COMPRESSED",
    ):
        for src_text, tar_text, trans, baseline in zip(
            source, references, predictions, baselines
        ):
            print(src_text)
            print("--> TARGET (original):", tar_text)
            print(f"--> {description}:", trans)
            print("--> BASELINE:", baseline)
            metrics = self.get_metrics(
                predictions=[trans], references=[tar_text], decimals=2
            )
            print("Metrics:")
            print(metrics)
            baseline_metrics = self.get_metrics(
                predictions=[baseline], references=[tar_text], decimals=2
            )
            print("Baseline Metrics:")
            print(baseline_metrics)
            print("__" * 30)

            # convert to a string instead of printing, so we can save it to a log file:
            return

    def get_csv_data(self, predictions, references, source, baselines, lang):
        csv_rows = [
            [
                "Source Text",
                "Original Target",
                "Translation",
                "Baseline",
                "Metrics",
                "Baseline Metrics",
            ]
        ]

        iterator = tqdm(zip(source, references, predictions, baselines))

        for src_text, tar_text, trans, baseline in iterator:
            metrics = self.get_metrics(
                predictions=[trans],
                references=[tar_text],
                lang=lang,
                decimals=2,
            )
            baseline_metrics = self.get_metrics(
                predictions=[baseline],
                references=[tar_text],
                lang=lang,
                decimals=2,
            )

            csv_rows.append(
                [
                    src_text,
                    tar_text,
                    trans,
                    baseline,
                    metrics,
                    baseline_metrics,
                ]
            )

        return csv_rows
