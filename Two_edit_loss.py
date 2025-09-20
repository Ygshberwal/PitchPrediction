import os
import unicodedata
import numpy as np
import pandas as pd
import torch
import random
from tqdm.auto import tqdm
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer
import Levenshtein
import sacrebleu
from datetime import datetime

start_time = datetime.now()
print("\n\n"+"="*40)
print(start_time)

# ---------- User config ----------
data_dir = "Datasets"
train_file = os.path.join(data_dir, "train.txt")
val_file   = os.path.join(data_dir, "val.txt")
test_file  = os.path.join(data_dir, "test.txt")

RANDOM_SEED = 42
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "outputs_pitch_restore_pairs"
MBART_LANG = "hi_IN"

# Pitch marks
VEDIC_ACCENTS_EXPLICIT = ['\u0951', '\u0952', '\u0953', '\u0954', '\u1CDA']
COMBINING_MARK_RANGES = [
    (0x0951, 0x0954),
    (0x1CD0, 0x1CFF),
]
SEP_TOKEN = " <SEP> "   # Keep a readable separator in training text (we store the literal with spaces)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def strip_combining_ranges(text, ranges):
    """Return text with combining marks from ranges removed and explicit vedic marks removed.
       Normalizes to NFC and collapses whitespace."""
    text = unicodedata.normalize("NFC", text)
    res = []
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in ranges):
            continue
        res.append(ch)
    out = "".join(res)
    for mark in VEDIC_ACCENTS_EXPLICIT:
        out = out.replace(mark, "")
    return " ".join(out.split())

def build_pair_dataframe(lines, sep_token=SEP_TOKEN):
    """
    Create pairs from consecutive lines.
    For each i: input = line_i_pitched + SEP + (line_{i+1} unpitched)
               target = line_i_pitched + SEP + (line_{i+1} pitched)
    """
    inputs, targets = [], []
    for i in range(len(lines) - 1):
        line1_pitched = unicodedata.normalize("NFC", lines[i])
        line2_pitched = unicodedata.normalize("NFC", lines[i+1])
        line2_plain = strip_combining_ranges(line2_pitched, COMBINING_MARK_RANGES)
        inp = line1_pitched + sep_token + line2_plain
        tgt = line1_pitched + sep_token + line2_pitched

        if not line1_pitched or not line2_pitched:
            continue
        inputs.append(inp)
        targets.append(tgt)
    df = pd.DataFrame({"input_text": inputs, "target_text": targets})
    return df

def split_second_sentence_from_pair(text, sep_token=SEP_TOKEN):
    """
    Given 'S1 <SEP> S2' return S1,S2. If separator not present, try few fallbacks; default: treat whole text as S2.
    Returns (s1, s2) - both stripped.
    """
    if sep_token in text:
        left, right = text.split(sep_token, 1)
        return left.strip(), right.strip()
    # fallback: try without the spaces
    if "<SEP>" in text:
        left, right = text.split("<SEP>", 1)
        return left.strip(), right.strip()

    return "", text.strip()

def exact_match(pred, ref):
    return int(pred.strip() == ref.strip())

def pitch_positions(s, pitch_tokens=None):
    if pitch_tokens is None:
        pts = set(VEDIC_ACCENTS_EXPLICIT)
        def is_pitch_char(ch):
            c = ord(ch)
            return (ch in pts) or any(lo <= c <= hi for lo, hi in COMBINING_MARK_RANGES)
    else:
        pts = set(pitch_tokens)
        def is_pitch_char(ch):
            return ch in pts
    return {(i, c) for i, c in enumerate(s) if is_pitch_char(c)}

def pitch_f1(pred, ref, pitch_tokens=None):
    pred_set = pitch_positions(pred, pitch_tokens)
    ref_set  = pitch_positions(ref, pitch_tokens)
    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1

def extract_pitch_string(s):
    """Return a string containing only the pitch marks from s."""
    return "".join(
        ch for ch in s
        if (ch in VEDIC_ACCENTS_EXPLICIT) or any(lo <= ord(ch) <= hi for lo, hi in COMBINING_MARK_RANGES)
    )

def pitch_edit_distance(pred, ref):
    """Compute raw and normalized edit distance for pitch accents only."""
    pred_pitch = extract_pitch_string(pred)
    ref_pitch  = extract_pitch_string(ref)
    dist = Levenshtein.distance(pred_pitch, ref_pitch)
    norm = dist / max(len(ref_pitch), 1)
    return dist, norm

train_sentences = load_sentences(train_file)
val_sentences   = load_sentences(val_file)
test_sentences  = load_sentences(test_file)

df_train = build_pair_dataframe(train_sentences)
df_val   = build_pair_dataframe(val_sentences)
df_test  = build_pair_dataframe(test_sentences)

print("Train pairs:", len(df_train))
print("Val pairs:  ", len(df_val))
print("Test pairs: ", len(df_test))

print("\nSample data rows (first 2):")
for i in range(min(2, len(df_train))):
    print("INPUT :", df_train.iloc[i]['input_text'])
    print("TARGET:", df_train.iloc[i]['target_text'])
    print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add vedic accent tokens and separator token if not present
tokens_to_add = []
for t in VEDIC_ACCENTS_EXPLICIT:
    if t not in tokenizer.get_vocab():
        tokens_to_add.append(t)
if "<SEP>" not in tokenizer.get_vocab():
    tokens_to_add.append("<SEP>")

if tokens_to_add:
    added = tokenizer.add_tokens(tokens_to_add)
    print(f"Added {added} tokens to tokenizer (accents + SEP). New vocab size:", len(tokenizer))
else:
    print("Accent/SEP tokens already present in tokenizer vocabulary.")

# mBART language codes handling
if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    tokenizer.src_lang = MBART_LANG
    tokenizer.tgt_lang = MBART_LANG
else:
    print(f"[WARN] MBART language code '{MBART_LANG}' not found in tokenizer.lang_code_to_id. Proceeding without forced language BOS; decoding quality may degrade.")

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 20
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
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

use_cuda = torch.cuda.is_available()

model = Seq2SeqModel(
    encoder_decoder_type = "mbart",
    encoder_decoder_name = MODEL_NAME,
    tokenizer = tokenizer,
    args = model_args,
    use_cuda = use_cuda
)

model.model.resize_token_embeddings(len(tokenizer))

if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    forced_id = tokenizer.lang_code_to_id[MBART_LANG]
    model.model.config.forced_bos_token_id = forced_id

print(f"\nStarting training with {model_args.num_train_epochs} epochs on {len(df_train)} pairs")
model.train_model(
    train_data=df_train,
    eval_data=df_val
)

# We'll use df_test inputs and evaluate second-sentence metrics only.
batch_inputs = df_test["input_text"].tolist()
refs_full = df_test["target_text"].tolist()

# decoding params explicit
model.args.max_length = 256
model.args.num_beams = 5

print("\nRunning prediction on test pairs...")
preds_full = model.predict(batch_inputs)

# ---------- Extract second sentences and compute metrics only for second sentence ----------
refs_second = []
preds_second = []

for ref_full, pred_full in zip(refs_full, preds_full):
    _, r2 = split_second_sentence_from_pair(ref_full)
    _, p2 = split_second_sentence_from_pair(pred_full)
    refs_second.append(r2)
    preds_second.append(p2)

# Compute metrics
exacts = [exact_match(p, r) for p, r in zip(preds_second, refs_second)]
pitch_f1s = [pitch_f1(p, r) for p, r in zip(preds_second, refs_second)]
pitch_edit_raw = []
pitch_edit_norm = []
for p, r in zip(preds_second, refs_second):
    d_raw, d_norm = pitch_edit_distance(p, r)
    pitch_edit_raw.append(d_raw)
    pitch_edit_norm.append(d_norm)

bleu_score = sacrebleu.corpus_bleu(preds_second, [refs_second]).score if preds_second else 0.0

print("\nEvaluation Results (SECOND SENTENCE ONLY)")
print("=" * 60)
print(f"Number of test pairs evaluated: {len(preds_second)}")
print(f"Exact Match Rate (second sent):   {float(np.mean(exacts)) * 100:.2f}%")
print(f"Pitch F1 (mean, second sent):     {float(np.mean(pitch_f1s)) * 100:.2f}%")
print(f"Pitch Edit Dist (norm, second):   {float(np.mean(pitch_edit_norm)) * 100:.2f}%")
print(f"Corpus BLEU (second sent):        {float(bleu_score):.2f}")

print("\nQualitative examples (showing second sentence predictions).")
n_examples = min(20, len(batch_inputs))
for i in range(n_examples):
    print("-" * 50)
    print("INPUT (pair)  :", batch_inputs[i])
    _, ref2 = split_second_sentence_from_pair(refs_full[i])
    _, pred2 = split_second_sentence_from_pair(preds_full[i])
    # Also show the full target for debugging if needed:
    print("TARGET (2nd)  :", ref2)
    print("PREDICTED (2nd):", pred2)
    print()

end_time = datetime.now()

print(f"Time taken for {model_args.num_train_epochs} epochs to run is {end_time-start_time}" )