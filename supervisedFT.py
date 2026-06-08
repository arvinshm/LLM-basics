import random
import torch
from datasets import load_dataset

import transformer as mlt


B = 16
T = 256
lr = 1e-4
steps = 2000

ALLOWED_CATEGORIES = {
    "classification",
    "closed_qa",
    "information_extraction",
    "summarization",
}

TRAIN_FRAC = 0.9
SEED = 1337

bpe = mlt.bpe
device = mlt.device
pad_id = bpe.stoi["<pad>"]
bos_id = bpe.stoi["<bos>"]
eos_id = bpe.stoi["<eos>"]


def format_example(row):
    instruction = row["instruction"].strip()
    context = row["context"].strip()
    response = row["response"].strip()

    if context:
        prompt = (
            "Instruction:\n"
            f"{instruction}\n\n"
            "Input:\n"
            f"{context}\n\n"
            "Answer:\n"
        )
    else:
        prompt = (
            "Instruction:\n"
            f"{instruction}\n\n"
            "Answer:\n"
        )
    return prompt, response


def make_sft_example(row):
    prompt, response = format_example(row)
    prompt_ids = [bos_id] + bpe.encode(prompt)
    answer_ids = bpe.encode(response) + [eos_id]
    tokens = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    x = tokens[:-1]
    y = labels[1:]
    if len(x) == 0 or len(x) > T:
        return None
    if all(label == -100 for label in y):
        return None
    return x, y


def load_examples():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds:
        if row["category"] not in ALLOWED_CATEGORIES:
            continue
        example = make_sft_example(row)
        if example is not None:
            examples.append(example)
    random.Random(SEED).shuffle(examples)
    split_idx = int(TRAIN_FRAC * len(examples))
    return examples[:split_idx], examples[split_idx:]


train_examples, val_examples = load_examples()


def get_batch(split):
    examples = train_examples if split == "train" else val_examples
    if not examples:
        raise ValueError(f"No {split} SFT examples available")

    batch = [examples[i] for i in torch.randint(len(examples), (B,)).tolist()]
    batch_T = max(len(x) for x, _ in batch)
    X = torch.full((B, batch_T), pad_id, dtype=torch.long, device=device)
    Y = torch.full((B, batch_T), -100, dtype=torch.long, device=device)

    for i, (x, y) in enumerate(batch):
        n = len(x)
        X[i, :n] = torch.tensor(x, dtype=torch.long, device=device)
        Y[i, :n] = torch.tensor(y, dtype=torch.long, device=device)

    return X, Y
