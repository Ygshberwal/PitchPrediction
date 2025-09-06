# Predict the pitch for a given sentence. Given a plain sentence, generate the pitched version of that sentence.

import os
import re
import random
import unicodedata
import numpy as np
import pandas as pd
import sacrebleu
import Levenshtein
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer
from datetime import datetime
import torch

INPUT_FILE = "Datasets/filtered_sanskritdoc.txt"
RANDOM_SEED = 42
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "outputs_pitch_restore"

# Vedic/Devanagari marks
VEDIC_ACCENTS_EXPLICIT = ['\u0951', '\u0952', '\u0953', '\u0954', '\u1CDA']  # your original list
# Broader ranges (combining marks used in practice)
# 0951–0954: Devanagari stress/udatta marks
# 1CD0–1CFF: Vedic Extensions (many combining tone marks)
# A8E0–A8FF: Devanagari Extended (optional; uncomment if your corpus contains them)
COMBINING_MARK_RANGES = [
    (0x0951, 0x0954),
    (0x1CD0, 0x1CFF),
    # (0xA8E0, 0xA8FF),  # enable if needed
]

MBART_LANG = "hi_IN"


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ------------------------
# 1) Load pitched sentences
# ------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pitched_sentences = [line.strip() for line in f if line.strip()]

print("\n\n" + "="*30)
print(datetime.now())
print(f"Loaded {len(pitched_sentences)} pitched sentences (sample):")
for s in pitched_sentences[:4]:
    print(" -", s)

# ------------------------
# 2) Remove accents (broader) to create plain inputs
# ------------------------
def strip_combining_ranges(text: str, ranges) -> str:
    # Normalize to NFC to keep base+combining in a consistent form first
    text = unicodedata.normalize("NFC", text)
    res = []
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in ranges):
            continue
        res.append(ch)
    out = "".join(res)
    # Also remove any explicitly listed marks that might slip through
    for mark in VEDIC_ACCENTS_EXPLICIT:
        out = out.replace(mark, "")
    # Collapse excessive whitespace
    return " ".join(out.split())

data = {
    "input_text": [strip_combining_ranges(s, COMBINING_MARK_RANGES) for s in pitched_sentences],
    "target_text": [unicodedata.normalize("NFC", s) for s in pitched_sentences],
}
df = pd.DataFrame(data)

n_identical = (df["input_text"] == df["target_text"]).sum()
print(f"\nIdentical lines (plain==pitched): {n_identical}")

# ------------------------
# Optional: Drop identical pairs for training/validation to avoid trivial copying
# Keep a small fraction in test to see how model behaves on already-clean lines
# ------------------------
df_nontriv = df[df["input_text"] != df["target_text"]].reset_index(drop=True)
if len(df_nontriv) >= 100:  # only if you have decent data volume
    df_for_split = df_nontriv
else:
    df_for_split = df  # fallback: keep everything

# ------------------------
# 3) Train/Val/Test splits (70:15:15)
# ------------------------
df_train, df_temp = train_test_split(df_for_split, test_size=0.30, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=RANDOM_SEED)
print("train:", len(df_train), "\nval:", len(df_val), "\ntest:", len(df_test))

# ------------------------
# 4) Tokenizer (mBART needs lang codes)
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add explicit accent tokens (usually these are combining marks and won't tokenize alone, but harmless)
tokenizer.add_tokens(VEDIC_ACCENTS_EXPLICIT, special_tokens=False)
print("Added accent tokens to tokenizer. New vocab size:", len(tokenizer))

# Set language codes for mBART-50
# SimpleTransformers will use this tokenizer during train/eval/predict.
# Important for correct decoder start/forced BOS behavior.
if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    tokenizer.src_lang = MBART_LANG
    tokenizer.tgt_lang = MBART_LANG
else:
    # Fallback: still proceed, but warn once.
    print(f"[WARN] MBART language code '{MBART_LANG}' not found in tokenizer.lang_code_to_id. "
          f"Proceeding without forced language BOS; decoding quality may degrade.")

# ------------------------
# 5) Model arguments
# ------------------------
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.max_sequence_length = 256
model_args.max_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.use_multiprocessing = False
model_args.overwrite_output_dir = True
model_args.output_dir = OUTPUT_DIR
model_args.best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
model_args.fp16 = torch.cuda.is_available()
model_args.save_eval_checkpoints = True
model_args.save_model_every_epoch = False
model_args.evaluate_during_training_steps = 1000
model_args.logging_steps = 200
model_args.save_steps = 1000
model_args.learning_rate = 5e-5
model_args.gradient_accumulation_steps = 4
model_args.use_multiprocessing_for_evaluation = False
model_args.num_beams = 5
model_args.length_penalty = 1.0
model_args.early_stopping = True                 
model_args.early_stopping_metric = "eval_loss"  
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 3
model_args.max_grad_norm = 1.0                   
model_args.reprocess_input_data = True          
model_args.save_best_model = True

# ------------------------
# 6) Initialize model
# ------------------------
use_cuda = torch.cuda.is_available()
model = Seq2SeqModel(
    encoder_decoder_type="mbart",
    encoder_decoder_name=MODEL_NAME,
    tokenizer=tokenizer,
    args=model_args,
    use_cuda=use_cuda,
)

# Resize embeddings to include any newly added tokens
model.model.resize_token_embeddings(len(tokenizer))

# If tokenizer has language codes, set forced_bos_token_id for decoding
if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    forced_id = tokenizer.lang_code_to_id[MBART_LANG]
    model.model.config.forced_bos_token_id = forced_id

# ------------------------
# 7) Train
# ------------------------
print(f"\nStarting training with {model_args.num_train_epochs} epochs")
model.train_model(
    train_data=df_train,
    eval_data=df_val
)

# ------------------------
# 8) Built-in eval
# ------------------------
raw_results = model.eval_model(df_test, verbose=True)
print("Simple eval results:", raw_results)

# ------------------------
# 9) Custom metrics
# ------------------------
def char_accuracy(pred, ref):
    L = max(len(pred), len(ref))
    pred_p, ref_p = pred.ljust(L), ref.ljust(L)
    return sum(1 for a, b in zip(pred_p, ref_p) if a == b) / L

def word_accuracy(pred, ref):
    p_tokens, r_tokens = pred.split(), ref.split()
    minlen = min(len(p_tokens), len(r_tokens))
    matches = sum(1 for i in range(minlen) if p_tokens[i] == r_tokens[i])
    return matches / max(len(r_tokens), 1)

def exact_match(pred, ref):
    return int(pred.strip() == ref.strip())

def pitch_positions(s, pitch_tokens=None):
    if pitch_tokens is None:
        # Define as any char in the configured ranges or explicit list
        pts = set(VEDIC_ACCENTS_EXPLICIT)
        def is_pitch_char(ch):
            c = ord(ch)
            return (ch in pts) or any(lo <= c <= hi for lo, hi in COMBINING_MARK_RANGES)
    else:
        pts = set(pitch_tokens)
        def is_pitch_char(ch):
            return ch in pts
    return {(i, c) for i, c in enumerate(s) if is_pitch_char(c)}

def pitch_accuracy(pred, ref, pitch_tokens=None):
    pred_pos = pitch_positions(pred, pitch_tokens)
    ref_pos  = pitch_positions(ref, pitch_tokens)
    matches = len(pred_pos & ref_pos)
    return matches / max(len(ref_pos), 1)

def pitch_f1(pred, ref, pitch_tokens=None):
    pred_set = pitch_positions(pred, pitch_tokens)
    ref_set  = pitch_positions(ref, pitch_tokens)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

# ------------------------
# 10) Predictions & Metrics
# ------------------------
batch_inputs = df_test["input_text"].tolist()

# Keep decoding params explicit
model.args.max_length = 256
model.args.num_beams = 5

preds = model.predict(batch_inputs)

char_accs, word_accs, exacts, lev_dists = [], [], [], []
pitch_accs, pitch_f1s = [], []

# For BLEU
refs = df_test["target_text"].tolist()
hyps = preds

for pred, ref in zip(hyps, refs):
    char_accs.append(char_accuracy(pred, ref))
    word_accs.append(word_accuracy(pred, ref))
    exacts.append(exact_match(pred, ref))
    lev_dists.append(Levenshtein.distance(pred, ref))
    pitch_accs.append(pitch_accuracy(pred, ref))
    pitch_f1s.append(pitch_f1(pred, ref))

bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0

metrics = {
    "char_accuracy_mean": float(np.mean(char_accs)),
    "word_accuracy_mean": float(np.mean(word_accs)),
    "exact_match_rate": float(np.mean(exacts)),
    "avg_levenshtein": float(np.mean(lev_dists)),
    "pitch_accuracy_mean": float(np.mean(pitch_accs)),
    "pitch_f1_mean": float(np.mean(pitch_f1s)),
    "corpus_BLEU": float(bleu_score),
}
print("Detailed metrics:", metrics)

# ------------------------
# 11) Examples
# ------------------------
print("\nQualitative examples:")
for i in range(min(10, len(batch_inputs))):
    print("---")
    print("INPUT :", batch_inputs[i])
    print("TARGET:", refs[i])
    print("PRED  :", hyps[i])
    print()

# ------------------------
# 12) Inference helper (FIXED)
# ------------------------
def restore_pitch(sentence: str) -> str:
    # Normalize + strip combining from input to mimic training pipeline
    plain = strip_combining_ranges(sentence, COMBINING_MARK_RANGES)
    model.args.max_length = 256
    model.args.num_beams = 5
    out = model.predict([plain])[0]
    return out

example = df_test["input_text"].iloc[0]
print("Example plain:", example)
print("Restored:", restore_pitch(example))
