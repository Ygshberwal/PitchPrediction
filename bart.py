# Required imports
import os
import random
import math
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# seq2seq library
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

# transformers tokenizer
from transformers import AutoTokenizer

# Metrics
import sacrebleu
import Levenshtein

# ------------------------
# User settings / constants
# ------------------------
INPUT_FILE = "filtered_sanskritdoc.txt"   # your file (one pitched sentence per line)
RANDOM_SEED = 42
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"  # multilingual seq2seq
OUTPUT_DIR = "outputs_pitch_restore"
VEDIC_ACCENTS = ['\u0951', '\u0952', '\u0953', '\u0954', '\u1CDA']  # the diacritics you'll predict
# ------------------------

# Fix seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1) Load pitched sentences
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pitched_sentences = [line.strip() for line in f if line.strip()]

# quick sanity
print(f"Loaded {len(pitched_sentences)} pitched sentences (sample):")
for s in pitched_sentences[:4]:
    print(" -", s)

# 2) Helper: remove pitch marks to create plain inputs
def remove_accents(sentence: str) -> str:
    for accent in VEDIC_ACCENTS:
        sentence = sentence.replace(accent, "")
    # Also normalize multiple spaces, trim
    sentence = " ".join(sentence.split())
    return sentence

# Build parallel dataset
data = {
    "input_text": [remove_accents(s) for s in pitched_sentences],
    "target_text": pitched_sentences,
}
df = pd.DataFrame(data)

# Optional quick check: number of identical pairs (already pitched = plain)
n_identical = (df["input_text"] == df["target_text"]).sum()
print(f"Identical lines (plain==pitched): {n_identical}")

# 3) Train/Val/Test splits
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=RANDOM_SEED)

print("Sizes -> train:", len(df_train), "val:", len(df_val), "test:", len(df_test))

# 4) Tokenizer - use AutoTokenizer for MBART
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add accent tokens as special tokens to tokenizer vocabulary
accent_tokens = [f"{a}" for a in VEDIC_ACCENTS]
# Add them as additional tokens
tokenizer.add_tokens(accent_tokens, special_tokens=False)  # not strictly "special", but added

print("Added accent tokens to tokenizer. New vocab size:", len(tokenizer))

# 5) Model arguments
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10            # start with 3 for a quick run; increase later
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.max_sequence_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.use_multiprocessing = False
model_args.overwrite_output_dir = True
model_args.output_dir = OUTPUT_DIR
model_args.best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
model_args.fp16 = True                     # use mixed precision if GPU supports it
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.evaluate_during_training_steps = 500
model_args.logging_steps = 200
model_args.save_steps = -1
model_args.learning_rate = 3e-5
model_args.gradient_accumulation_steps = 4  # if GPU memory limited
model_args.use_multiprocessing_for_evaluation = False

# 6) Initialize model
# We pass the tokenizer object to Seq2SeqModel and let it use the pretrained weights
# Path to your fine-tuned model
trained_model_path = "outputs_pitch_restore"  # change this to your actual save folder

model = Seq2SeqModel(
    encoder_decoder_type="mbart",
    encoder_decoder_name=trained_model_path,  # load from trained weights
    tokenizer=trained_model_path,             # load the tokenizer used during training
    args=model_args,
    use_cuda=True
)

# Resize token embeddings to account for added accent tokens
model.model.resize_token_embeddings(len(tokenizer))

# 7) Train
print("Starting training...")
model.train_model(
    train_data=df_train,
    eval_data=df_val,  # validation data must be passed like this
    output_dir=OUTPUT_DIR
)


# 8) Evaluate on test set with built-in eval (returns a dictionary)
raw_results = model.eval_model(df_test, verbose=True)
print("Simple eval results (from simpletransformers):", raw_results)

# 9) More detailed evaluation functions (char-level acc, word-level acc, BLEU, Levenshtein)
def char_accuracy(pred: str, ref: str) -> float:
    # pad to same length
    L = max(len(pred), len(ref))
    pred_p = pred.ljust(L)
    ref_p = ref.ljust(L)
    matches = sum(1 for a, b in zip(pred_p, ref_p) if a == b)
    return matches / L

def word_accuracy(pred: str, ref: str) -> float:
    p_tokens = pred.split()
    r_tokens = ref.split()
    # align by index up to min len, count matches; the rest considered mismatches
    minlen = min(len(p_tokens), len(r_tokens))
    matches = sum(1 for i in range(minlen) if p_tokens[i] == r_tokens[i])
    return matches / max(len(r_tokens), 1)

def exact_match(pred: str, ref: str) -> int:
    return int(pred.strip() == ref.strip())

# Run predictions in batches for test set
batch = list(df_test["input_text"].tolist())
preds = model.predict(batch)  # returns list of predicted strings

# Compute metrics
char_accs = []
word_accs = []
exacts = []
lev_dists = []
bleu_refs = []
bleu_sys = []

for pred, ref in zip(preds, df_test["target_text"].tolist()):
    char_accs.append(char_accuracy(pred, ref))
    word_accs.append(word_accuracy(pred, ref))
    exacts.append(exact_match(pred, ref))
    lev_dists.append(Levenshtein.distance(pred, ref))
    bleu_refs.append([ref])  # sacrebleu expects list of references per sentence
    bleu_sys.append(pred)

# corpus BLEU (sacrebleu)
bleu_score = sacrebleu.corpus_bleu(bleu_sys, list(zip(*bleu_refs))).score if len(bleu_sys)>0 else 0.0

metrics = {
    "char_accuracy_mean": float(np.mean(char_accs)),
    "word_accuracy_mean": float(np.mean(word_accs)),
    "exact_match_rate": float(np.mean(exacts)),
    "avg_levenshtein": float(np.mean(lev_dists)),
    "corpus_BLEU": float(bleu_score),
}

print("Detailed metrics:", metrics)

# 10) Show some qualitative examples: input / target / predicted
print("\nQualitative examples:")
N = 10
for i in range(min(N, len(batch))):
    print("---")
    print("INPUT :", batch[i])
    print("TARGET:", df_test['target_text'].iloc[i])
    print("PRED  :", preds[i])
    print()

# 11) Inference helper
def restore_pitch(sentence: str) -> str:
    """Return restored pitched sentence for a plain input."""
    sentence = sentence.strip()
    pred = model.predict([sentence])
    print(pred)
    return pred[0]

# Example
example = df_test["input_text"].iloc[0]
print("Example plain:", example)
print("Restored:", restore_pitch(example))

