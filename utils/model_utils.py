import os

from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline


def load_model(
    path,
    device="cuda",
):
    last_checkpoint = sorted(os.listdir(path))[-1]
    path = os.path.join(path, last_checkpoint)
    model = MarianMTModel.from_pretrained(path).to(device)
    tokenizer = MarianTokenizer.from_pretrained(path)
    return model, tokenizer


def load_baseline(model_id, device="cuda"):
    baseline_model = MarianMTModel.from_pretrained(model_id).to(device)
    baseline_tokenizer = MarianTokenizer.from_pretrained(model_id)
    return baseline_model, baseline_tokenizer


def generate(model, tokenizer, source_texts, prefix=">>nob<<", device="cuda"):
    txts = [f"{prefix} {source_text}".strip() for source_text in source_texts]
    inputs = tokenizer.batch_encode_plus(
        txts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    translated = model.generate(**inputs)
    translated_texts = [
        tokenizer.decode(t, skip_special_tokens=True) for t in translated
    ]
    return translated_texts


def translate_all(source, model, tokenizer, device="cuda", batch_size=32):
    translated = []
    for i in tqdm(range(0, len(source), batch_size)):
        batch = source[i : i + batch_size]
        translated.extend(generate(model, tokenizer, batch, device=device))
    return translated
