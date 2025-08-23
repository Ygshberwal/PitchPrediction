# ------------------------
# Imports
# ------------------------
import os
import random
import numpy as np
import pandas as pd
import sacrebleu
import Levenshtein
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from transformers import AutoTokenizer
from datetime import datetime

# ------------------------
# User settings / constants
# ------------------------
INPUT_FILE = "filtered_sanskritdoc.txt"
RANDOM_SEED = 42
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "outputs_pitch_restore"
VEDIC_ACCENTS = ['\u0951', '\u0952', '\u0953', '\u0954', '\u1CDA']

# ------------------------
# Fix seeds
# ------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------
# 1) Load pitched sentences
# ------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pitched_sentences = [line.strip() for line in f if line.strip()]
print()
print("="*30)
print(f"Loaded {len(pitched_sentences)} pitched sentences (sample):")
for s in pitched_sentences[:4]:
    print(" -", s)

# ------------------------
# 2) Remove accents to create plain inputs
# ------------------------
def remove_accents(sentence: str) -> str:
    for accent in VEDIC_ACCENTS:
        sentence = sentence.replace(accent, "")
    return " ".join(sentence.split())

data = {
    "input_text": [remove_accents(s) for s in pitched_sentences],
    "target_text": pitched_sentences,
}
df = pd.DataFrame(data)

n_identical = (df["input_text"] == df["target_text"]).sum()
print(f"\nIdentical lines (plain==pitched): {n_identical}")

# ------------------------
# 3) Train/Val/Test splits
# ------------------------
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=RANDOM_SEED)
print("train:", len(df_train), "\nval:", len(df_val), "\ntest:", len(df_test))

# ------------------------
# 4) Tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_tokens(VEDIC_ACCENTS, special_tokens=False)
print("Added accent tokens to tokenizer. New vocab size:", len(tokenizer))

# ------------------------
# 5) Model arguments
# ------------------------
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 1
model_args.train_batch_size = 8
model_args.eval_batch_size = 8
model_args.max_sequence_length = 256
model_args.max_length = 256  # ensure decoding covers whole sentence
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.use_multiprocessing = False
model_args.overwrite_output_dir = True
model_args.output_dir = OUTPUT_DIR
model_args.best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
model_args.fp16 = True
model_args.save_eval_checkpoints = True
model_args.save_model_every_epoch = False
model_args.evaluate_during_training_steps = 1000  # adjusted for ~32k dataset
model_args.logging_steps = 200
model_args.save_steps = 1000
model_args.learning_rate = 5e-5
model_args.gradient_accumulation_steps = 4
model_args.use_multiprocessing_for_evaluation = False
model_args.num_beams = 5
model_args.length_penalty = 1.0
model_args.early_stopping = False

# ------------------------
# 6) Initialize model
# ------------------------
model = Seq2SeqModel(
    encoder_decoder_type="mbart",
    encoder_decoder_name=MODEL_NAME,
    tokenizer=tokenizer,
    args=model_args,
    use_cuda=True
)
model.model.resize_token_embeddings(len(tokenizer))

# ------------------------
# 7) Train
# ------------------------
print(f"\nStarting training... at time {datetime.now()} with {model_args.num_train_epochs} epochs")

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

def pitch_accuracy(pred, ref, pitch_tokens=VEDIC_ACCENTS):
    pred_pos = [(i, c) for i, c in enumerate(pred) if c in pitch_tokens]
    ref_pos  = [(i, c) for i, c in enumerate(ref) if c in pitch_tokens]
    matches = sum(1 for pos in pred_pos if pos in ref_pos)
    return matches / max(len(ref_pos), 1)

def pitch_f1(pred, ref, pitch_tokens=VEDIC_ACCENTS):
    pred_set = {(i, c) for i, c in enumerate(pred) if c in pitch_tokens}
    ref_set  = {(i, c) for i, c in enumerate(ref) if c in pitch_tokens}
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
batch = df_test["input_text"].tolist()
model.args.max_length = 256
model.args.num_beams = 5
preds = model.predict(batch)

char_accs, word_accs, exacts, lev_dists = [], [], [], []
pitch_accs, pitch_f1s = [], []
bleu_refs, bleu_sys = [], []

for pred, ref in zip(preds, df_test["target_text"].tolist()):
    char_accs.append(char_accuracy(pred, ref))
    word_accs.append(word_accuracy(pred, ref))
    exacts.append(exact_match(pred, ref))
    lev_dists.append(Levenshtein.distance(pred, ref))
    pitch_accs.append(pitch_accuracy(pred, ref))
    pitch_f1s.append(pitch_f1(pred, ref))
    bleu_refs.append([ref])
    bleu_sys.append(pred)

bleu_score = sacrebleu.corpus_bleu(bleu_sys, list(zip(*bleu_refs))).score if bleu_sys else 0.0

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
for i in range(min(10, len(batch))):
    print("---")
    print("INPUT :", batch[i])
    print("TARGET:", df_test['target_text'].iloc[i])
    print("PRED  :", preds[i])
    print()

# ------------------------
# 12) Inference helper
# ------------------------
def restore_pitch(sentence: str) -> str:
    model.args.max_length = 256
    model.args.num_beams = 5
    preds = model.predict(batch)
    return pred[0]

example = df_test["input_text"].iloc[0]
print("Example plain:", example)
print("Restored:", restore_pitch(example))
