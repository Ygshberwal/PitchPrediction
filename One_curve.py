import os
import unicodedata
import numpy as np
import pandas as pd
import torch
import random
import time
import json
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer
import Levenshtein
import sacrebleu

# -------------------------
# Config / timing
# -------------------------
start_time = datetime.now()
print("\n\n" + "=" * 40)
print("Start:", start_time.isoformat())

data_dir = "Datasets"
train_file = os.path.join(data_dir, "train.txt")
val_file = os.path.join(data_dir, "val.txt")
test_file = os.path.join(data_dir, "test.txt")

RANDOM_SEED = 42
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "outputs_pitch_restore"
MBART_LANG = "hi_IN"

VEDIC_ACCENTS_EXPLICIT = ['\u0951', '\u0952', '\u0953', '\u0954', '\u1CDA']
COMBINING_MARK_RANGES = [
    (0x0951, 0x0954),
    (0x1CD0, 0x1CFF),
    # (0xA8E0, 0xA8FF),  # enable if needed
]

# seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------
# Data loading
# -------------------------
def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train_sentences = load_sentences(train_file)
val_sentences = load_sentences(val_file)
test_sentences = load_sentences(test_file)


def strip_combining_ranges(text, ranges):
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
    # normalize spacing
    return " ".join(out.split())

def make_dataframe(sentences):
    return pd.DataFrame({
        "input_text": [strip_combining_ranges(s, COMBINING_MARK_RANGES) for s in sentences],
        "target_text": [unicodedata.normalize("NFC", s) for s in sentences]
    })

df_train = make_dataframe(train_sentences)
df_val = make_dataframe(val_sentences)
df_test = make_dataframe(test_sentences)

# -------------------------
# Tokenizer (and compatibility patch)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(VEDIC_ACCENTS_EXPLICIT, special_tokens=False)

if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    tokenizer.src_lang = MBART_LANG
    tokenizer.tgt_lang = MBART_LANG
else:
    print(f"[WARN] MBART language code '{MBART_LANG}' not found in tokenizer.lang_code_to_id. "
          f"Proceeding without forced language BOS; decoding quality may degrade.")

# ---------- BEGIN PATCH: compatibility for pad_to_max_length ----------
# Some SimpleTransformers versions call prepare_seq2seq_batch(..., pad_to_max_length=...)
# Newer HuggingFace tokenizers expect padding='max_length' instead. Add a thin wrapper
# so calls using pad_to_max_length still work.
def _patch_prepare_seq2seq_batch(tok):
    if not hasattr(tok, "prepare_seq2seq_batch"):
        return
    orig = tok.prepare_seq2seq_batch

    def wrapped(*args, **kwargs):
        if "pad_to_max_length" in kwargs:
            pad_flag = kwargs.pop("pad_to_max_length")
            kwargs.setdefault("padding", "max_length" if pad_flag else False)
        return orig(*args, **kwargs)

    # bind wrapper as attribute
    try:
        setattr(tok, "prepare_seq2seq_batch", wrapped)
    except Exception:
        import types
        setattr(tok, "prepare_seq2seq_batch", types.MethodType(wrapped, tok))

# Patch the top-level tokenizer
_patch_prepare_seq2seq_batch(tokenizer)
# ---------- END PATCH ----------

# -------------------------
# Model args and model creation
# -------------------------
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 30          # total intended epochs; we'll loop up to this
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

use_cuda = torch.cuda.is_available()

model = Seq2SeqModel(
    encoder_decoder_type="mbart",
    encoder_decoder_name=MODEL_NAME,
    tokenizer=tokenizer,
    args=model_args,
    use_cuda=use_cuda
)

# resize embeddings to account for added tokens
model.model.resize_token_embeddings(len(tokenizer))

if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    forced_id = tokenizer.lang_code_to_id[MBART_LANG]
    model.model.config.forced_bos_token_id = forced_id

# Also patch tokenizers inside the SimpleTransformers model object
try:
    _patch_prepare_seq2seq_batch(model.tokenizer)
except Exception:
    pass
for attr in ("encoder_tokenizer", "decoder_tokenizer"):
    tk = getattr(model, attr, None)
    if tk is not None:
        _patch_prepare_seq2seq_batch(tk)

print(f"\nModel ready. Intended total epochs: {model_args.num_train_epochs}")

# -------------------------
# Metric functions (your originals)
# -------------------------
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
    return "".join(
        ch for i, ch in enumerate(s)
        if (ch in VEDIC_ACCENTS_EXPLICIT) or any(lo <= ord(ch) <= hi for lo, hi in COMBINING_MARK_RANGES)
    )

def pitch_edit_distance(pred, ref):
    pred_pitch = extract_pitch_string(pred)
    ref_pitch  = extract_pitch_string(ref)
    dist = Levenshtein.distance(pred_pitch, ref_pitch)
    norm = dist / max(len(ref_pitch), 1)
    return dist, norm

# -------------------------
# Prediction helper params
# -------------------------
model.args.max_length = model_args.max_length
model.args.num_beams = model_args.num_beams

# -------------------------
# Epoch-by-epoch training + evaluation
# -------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

TOTAL_EPOCHS = model_args.num_train_epochs if hasattr(model_args, "num_train_epochs") else 20

# We'll set training args to 1 epoch per model.train_model call and loop
model.args.num_train_epochs = 1

# Keep a copy of whether to reprocess inputs on first epoch only
reprocess_first_epoch = True

epoch_records = []

def evaluate_and_collect(df_eval, split_name="val"):
    """
    Run model.eval_model on df_eval (to get eval_loss if provided by trainer),
    then run model.predict to get decoded predictions, and compute custom metrics.
    Returns (metrics_dict, predictions_list).
    """
    try:
        result_tuple = model.eval_model(df_eval, verbose=False)
        # result_tuple can be (result_dict, outputs) or a dict depending on versions
        if isinstance(result_tuple, tuple) and len(result_tuple) >= 1:
            result = result_tuple[0]
        elif isinstance(result_tuple, dict):
            result = result_tuple
        else:
            result = {}
    except Exception as e:
        print(f"[WARN] eval_model raised exception for {split_name}: {e}")
        result = {}

    # extract loss (keys may vary across versions)
    eval_loss = None
    for k in ("eval_loss", "loss", "eval_loss_loss"):
        if isinstance(result, dict) and k in result:
            eval_loss = result[k]
            break

    # Predict decoded outputs (this uses model.generate under the hood)
    batch_inputs = df_eval["input_text"].tolist()
    model.args.max_length = model_args.max_length
    model.args.num_beams = model_args.num_beams
    preds = model.predict(batch_inputs)

    refs = df_eval["target_text"].tolist()
    hyps = preds

    exacts = [exact_match(p, r) for p, r in zip(hyps, refs)]
    pitch_f1s = [pitch_f1(p, r) for p, r in zip(hyps, refs)]

    ped_raw, ped_norm = [], []
    for p, r in zip(hyps, refs):
        d_raw, d_norm = pitch_edit_distance(p, r)
        ped_raw.append(d_raw)
        ped_norm.append(d_norm)

    bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0

    metrics = {
        "loss": float(eval_loss) if eval_loss is not None else None,
        "exact_match_mean": float(np.mean(exacts)) if len(exacts) else 0.0,
        "pitch_f1_mean": float(np.mean(pitch_f1s)) if len(pitch_f1s) else 0.0,
        "pitch_edit_norm_mean": float(np.mean(ped_norm)) if len(ped_norm) else 0.0,
        "pitch_edit_raw_mean": float(np.mean(ped_raw)) if len(ped_raw) else 0.0,
        "bleu": float(bleu_score),
        "n_examples": len(batch_inputs),
    }

    return metrics, hyps

print(f"\nStarting epoch-by-epoch training for {TOTAL_EPOCHS} epochs")
global_start = datetime.now()

for epoch in range(1, TOTAL_EPOCHS + 1):
    print("\n" + "=" * 60)
    print(f"Starting Epoch {epoch}/{TOTAL_EPOCHS} -- {datetime.now().isoformat()}")
    epoch_start = time.time()

    # Manage reprocessing of input data (only on first epoch if requested)
    if reprocess_first_epoch:
        model.args.reprocess_input_data = getattr(model_args, "reprocess_input_data", True)
    else:
        model.args.reprocess_input_data = False

    # -----------------------------
    # IMPORTANT: Do NOT pass args=model.args here.
    # Passing a Seq2SeqArgs instance causes SimpleTransformers internals to expect a dict.
    # Let train_model use model.args already attached to the model.
    # -----------------------------
    model.train_model(
        train_data=df_train,
        eval_data=df_val
    )

    # Evaluate on train/val/test splits and compute custom metrics
    print("Evaluating on train split...")
    train_metrics, _ = evaluate_and_collect(df_train, "train")
    print("Evaluating on val split...")
    val_metrics, _ = evaluate_and_collect(df_val, "val")
    print("Evaluating on test split...")
    test_metrics, test_preds = evaluate_and_collect(df_test, "test")

    epoch_time = time.time() - epoch_start

    # Save test predictions for this epoch
    preds_file = os.path.join(OUTPUT_DIR, f"preds_epoch_{epoch}.txt")
    with open(preds_file, "w", encoding="utf-8") as fout:
        fout.write(f"# Epoch {epoch} predictions\n")
        for inp, ref, hyp in zip(df_test["input_text"].tolist(), df_test["target_text"].tolist(), test_preds):
            fout.write("INPUT\t" + inp + "\n")
            fout.write("REF  \t" + ref + "\n")
            fout.write("PRED \t" + hyp + "\n")
            fout.write("\n")

    record = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "epoch_time_seconds": epoch_time,

        "train_loss": train_metrics["loss"],
        "train_exact": train_metrics["exact_match_mean"],
        "train_pitch_f1": train_metrics["pitch_f1_mean"],
        "train_pitch_edit_norm": train_metrics["pitch_edit_norm_mean"],
        "train_bleu": train_metrics["bleu"],

        "val_loss": val_metrics["loss"],
        "val_exact": val_metrics["exact_match_mean"],
        "val_pitch_f1": val_metrics["pitch_f1_mean"],
        "val_pitch_edit_norm": val_metrics["pitch_edit_norm_mean"],
        "val_bleu": val_metrics["bleu"],

        "test_loss": test_metrics["loss"],
        "test_exact": test_metrics["exact_match_mean"],
        "test_pitch_f1": test_metrics["pitch_f1_mean"],
        "test_pitch_edit_norm": test_metrics["pitch_edit_norm_mean"],
        "test_bleu": test_metrics["bleu"],

        "preds_file": preds_file
    }

    epoch_records.append(record)

    # Save CSV + JSON after each epoch (resilient to crashes)
    metrics_csv = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
    metrics_json = os.path.join(OUTPUT_DIR, "epoch_metrics.json")
    df_epoch = pd.DataFrame(epoch_records)
    df_epoch.to_csv(metrics_csv, index=False, encoding="utf-8")
    with open(metrics_json, "w", encoding="utf-8") as jf:
        json.dump(epoch_records, jf, ensure_ascii=False, indent=2)

    print(f"Epoch {epoch} finished. Time: {epoch_time:.1f}s.")
    print(f"Val loss: {val_metrics['loss']}, Test loss: {test_metrics['loss']}")
    print(f"Saved predictions -> {preds_file}")

    # After first epoch, don't reprocess input data again (speeds up subsequent epochs)
    reprocess_first_epoch = False

global_end = datetime.now()
total_duration = global_end - global_start
print(f"\nCompleted {TOTAL_EPOCHS} epochs. Total time: {total_duration}")

# Final summary
print("\nPer-epoch summary:")
print(pd.DataFrame(epoch_records).to_string(index=False))

# Show a few sample preds from last epoch
if epoch_records:
    print(f"\nSample predictions from last epoch (epoch {epoch_records[-1]['epoch']}):\n")
    hyps = model.predict(df_test['input_text'].tolist())
    for i in range(min(5, len(df_test))):
        print("---")
        print("INPUT :", df_test['input_text'].iloc[i])
        print("TARGET:", df_test['target_text'].iloc[i])
        print("PRED  :", hyps[i])
        print()

end_time = datetime.now()
print(f"Time taken (wall) for run: {end_time - start_time}")
