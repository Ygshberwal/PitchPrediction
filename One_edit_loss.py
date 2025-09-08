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

data_file = "Datasets/filtered_sanskritdoc.txt"
train_file = "Datasets/train.txt"
val_file = "Datasets/val.txt"
test_file = "Datasets/test.txt"

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

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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
    return " ".join(out.split())

def make_dataframe(sentences):
    return pd.DataFrame({
        "input_text": [strip_combining_ranges(s, COMBINING_MARK_RANGES) for s in sentences],
        "target_text": [unicodedata.normalize("NFC", s) for s in sentences]
    })


df_train = make_dataframe(train_sentences)
df_val = make_dataframe(val_sentences)
df_test = make_dataframe(test_sentences)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(VEDIC_ACCENTS_EXPLICIT, special_tokens=False)              # one token added
print("Added accent tokens to tokenizer. New vocab size:", len(tokenizer))

if hasattr(tokenizer, "lang_code_to_id") and MBART_LANG in tokenizer.lang_code_to_id:
    tokenizer.src_lang = MBART_LANG
    tokenizer.tgt_lang = MBART_LANG
else:
    # Fallback: still proceed, but warn once.
    print(f"[WARN] MBART language code '{MBART_LANG}' not found in tokenizer.lang_code_to_id. "
          f"Proceeding without forced language BOS; decoding quality may degrade.")


model_args = Seq2SeqArgs()
model_args.num_train_epochs = 1
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

print(f"\nStarting training with {model_args.num_train_epochs} epochs")
model.train_model(
    train_data=df_train,
    eval_data=df_val
)

raw_results = model.eval_model(df_test, verbose=True)
print("Simple eval results", raw_results)



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

def extract_pitch_string(s):
    """Return a string containing only the pitch marks from s."""
    return "".join(
        ch for i, ch in enumerate(s)
        if (ch in VEDIC_ACCENTS_EXPLICIT) or any(lo <= ord(ch) <= hi for lo, hi in COMBINING_MARK_RANGES)
    )

def pitch_edit_distance(pred, ref):
    """Compute raw and normalized edit distance for pitch accents only."""
    pred_pitch = extract_pitch_string(pred)
    ref_pitch  = extract_pitch_string(ref)
    dist = Levenshtein.distance(pred_pitch, ref_pitch)
    norm = dist / max(len(ref_pitch), 1)
    return dist, norm


batch_inputs = df_test["input_text"].tolist()

# Keep decoding params explicit
model.args.max_length = 256
model.args.num_beams = 5

preds = model.predict(batch_inputs)

exacts = []
pitch_f1s = []
pitch_edit_raw, pitch_edit_norm = [], []

refs = df_test["target_text"].tolist()
hyps = preds

for pred, ref in zip(hyps, refs):
    exacts.append(exact_match(pred, ref))
    pitch_f1s.append(pitch_f1(pred, ref))    
    d_raw, d_norm = pitch_edit_distance(pred, ref)
    pitch_edit_raw.append(d_raw)
    pitch_edit_norm.append(d_norm)


bleu_score = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0

print("\nEvaluation Results")
print("=" * 50)
print(f"Exact Match Rate:             {float(np.mean(exacts)) * 100:.2f}%")
print(f"Pitch F1 Score (mean):        {float(np.mean(pitch_f1s)) * 100:.2f}%")
print(f"Pitch Edit Distance (norm):   {float(np.mean(pitch_edit_norm)) * 100:.2f}%")
print(f"Corpus BLEU:                  {float(bleu_score):.2f}")

print("\nQualitative examples:")
for i in range(min(10, len(batch_inputs))):
    print("---")
    print("INPUT :", batch_inputs[i])
    print("TARGET:", refs[i])
    print("PRED  :", hyps[i])
    print()
