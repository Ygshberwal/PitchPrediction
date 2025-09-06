# Predict the pitch for a given sentence using a helper sentence. Given a pitched and a plain sentence, generate the pitched version of both the sentences.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
import Levenshtein
import sacrebleu
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from datetime import datetime

data_file = "Datasets/filtered_sanskritdoc.txt"

with open(data_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
print("\n\n"+"="*40)
print(datetime.now())
print(f"Loaded {len(lines)} pitched sentences (sample):")
for s in lines[:4]:
    print(" -", s)

# Helper: remove pitch marks (anudatta/svarita signs)
def remove_pitch(text):
    # Sanskrit pitch diacritics: U+0951 (॑), U+0952 (॒)
    return text.replace("॑", "").replace("॒", "")

# Create dataset for seq2seq
inputs, targets = [], []
for i in range(len(lines) - 1):
    line1_pitched = lines[i]         # keep pitched
    line2_pitched = lines[i+1]       # pitched form
    line2_plain   = remove_pitch(line2_pitched)  # plain form

    inp = line1_pitched + " <SEP> " + line2_plain
    tgt = line1_pitched + " <SEP> " + line2_pitched

    inputs.append(inp)
    targets.append(tgt)

df = pd.DataFrame({"input_text": inputs, "target_text": targets})
print("\nSample data rows:")
for i in range(2):
    print("INPUT :", df.iloc[i]['input_text'])
    print("TARGET:", df.iloc[i]['target_text'], "\n")

# ------------------------
# Train/Validation/Test split (70:15:15)
# ------------------------
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)   # 70% train, 30% temp
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42) # 15% val, 15% test

print(f"\nDataset split sizes:")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

# ------------------------
# Model Arguments
# ------------------------

model_args = Seq2SeqArgs()

model_args.output_dir = "./outputs"
model_args.cache_dir = "./cache_dir"
model_args.overwrite_output_dir = True
model_args.save_total_limit = 2
model_args.num_train_epochs = 10          # train longer (1 epoch is usually too low)
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
model_args.gradient_accumulation_steps = 4
model_args.fp16 = False                  # set True if you have GPU with mixed precision
model_args.max_seq_length = 256          # allow longer input (was 64)
model_args.max_length = 256              # max length of generated sequence
model_args.gen_max_length = 256          # same for generation
model_args.min_length = 30               # prevent very short outputs
model_args.length_penalty = 1.0          # >1 favors longer generations
model_args.early_stopping = True
model_args.num_beams = 5                 # beam search for better full-sentence outputs
model_args.do_sample = False             # if True → random sampling
# model_args.top_k = 50                  # only used if do_sample=True
# model_args.top_p = 0.95                # nucleus sampling (if sampling enabled)
model_args.gradient_checkpointing = True
model_args.use_multiprocessing = False
model_args.save_steps = 2000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 2000
model_args.logging_steps = 100


# Initialize Model
model = Seq2SeqModel(
    encoder_decoder_type="mbart",   
    encoder_decoder_name="facebook/mbart-large-50-many-to-many-mmt",
    args=model_args,
    use_cuda=torch.cuda.is_available(),
)
print("Using GPU: ", torch.cuda.is_available())


# Training
print(f"\nStarting training with {model_args.num_train_epochs} epochs")
model.train_model(train_df, eval_data=val_df)

# Evaluation Function
def evaluate_predictions(preds, labels):
    char_accs, word_accs, exacts, lev_dists = [], [], [], []
    pitch_accs, pitch_f1s = [], []

    for pred, gold in zip(preds, labels):
        # Ensure strings
        pred = "" if pred is None else str(pred)
        gold = "" if gold is None else str(gold)

        # Character-level accuracy
        if len(gold) == 0:
            char_accs.append(0.0)
        else:
            min_len = min(len(pred), len(gold))
            correct_chars = sum(p == g for p, g in zip(pred[:min_len], gold[:min_len]))
            char_accs.append(correct_chars / max(len(gold), 1))

        # Word-level accuracy
        pred_words = pred.split()
        gold_words = gold.split()
        if len(gold_words) == 0:
            word_accs.append(0.0)
        else:
            min_len_w = min(len(pred_words), len(gold_words))
            correct_words = sum(p == g for p, g in zip(pred_words[:min_len_w], gold_words[:min_len_w]))
            word_accs.append(correct_words / max(len(gold_words), 1))

        # Exact match
        exacts.append(int(pred == gold))

        # Levenshtein distance
        try:
            lev = Levenshtein.distance(pred, gold)
        except Exception:
            lev = max(len(pred), len(gold))
        lev_dists.append(lev)

        # Pitch accuracy & F1 (symbols only)
        pitch_symbols = set(["॑", "॒"])
        y_true = [1 if ch in pitch_symbols else 0 for ch in gold]
        y_pred = []
        for i in range(len(gold)):
            if i < len(pred) and pred[i] in pitch_symbols:
                y_pred.append(1)
            else:
                y_pred.append(0)

        if len(y_true) > 0:
            pitch_acc = sum(1 for a, b in zip(y_pred, y_true) if a == b) / len(y_true)
            pitch_accs.append(pitch_acc)
            try:
                pitch_f1 = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                pitch_f1 = 0.0
            pitch_f1s.append(pitch_f1)

    # Corpus BLEU
    try:
        bleu_score = sacrebleu.corpus_bleu(preds, [labels]).score
    except Exception as e:
        print("Warning: sacrebleu failed:", e)
        bleu_score = 0.0

    metrics = {
        "char_accuracy_mean": float(np.mean(char_accs)) if char_accs else 0.0,
        "word_accuracy_mean": float(np.mean(word_accs)) if word_accs else 0.0,
        "exact_match_rate": float(np.mean(exacts)) if exacts else 0.0,
        "avg_levenshtein": float(np.mean(lev_dists)) if lev_dists else 0.0,
        "pitch_accuracy_mean": float(np.mean(pitch_accs)) if pitch_accs else 0.0,
        "pitch_f1_mean": float(np.mean(pitch_f1s)) if pitch_f1s else 0.0,
        "corpus_BLEU": float(bleu_score),
    }
    return metrics


# Final Evaluation on Test Set
print("\nRunning final evaluation on test set...")
to_predict = list(test_df["input_text"])
true_labels = list(test_df["target_text"])

# 🔑 set decoding configs inside model.args
model.args.max_length = 256
model.args.num_beams = 4
model.args.repetition_penalty = 1.2
model.args.early_stopping = False

raw_preds = model.predict(to_predict)

# handle both possible return formats
if isinstance(raw_preds, tuple) or (isinstance(raw_preds, list) and len(raw_preds) == 2 and not all(isinstance(x, str) for x in raw_preds)):
    preds = raw_preds[0]
else:
    preds = raw_preds

# Ensure preds is a list of strings
preds = [p if isinstance(p, str) else "" for p in preds]

# Compute metrics
try:
    metrics = evaluate_predictions(preds, true_labels)
    print("Detailed metrics:", metrics)
except Exception as e:
    print("Error during evaluation:", type(e).__name__, e)
    print("Sample inputs / preds / labels (first 10):")
    for i in range(min(10, len(to_predict))):
        print("INPUT :", to_predict[i])
        print("PRED  :", preds[i])
        print("GOLD  :", true_labels[i])
        print("-"*30)

# Print sample predictions
print("\nSample predictions (up to 20):")
n_show = min(20, len(preds))
for i in range(n_show):
    print(f"[{i+1}]")
    print("INPUT :", to_predict[i])
    print("GOLD  :", true_labels[i])
    print("PRED  :", preds[i])
    print()

# Demo Prediction
line1_pitched = lines[0]
line2_pitched = lines[1]
line2_plain   = remove_pitch(line2_pitched)

test_input = line1_pitched + " <SEP> " + line2_plain
print("\nDemo input:", test_input)

pred_demo = model.predict([test_input])[0]

print("Expected target:", line1_pitched + " <SEP> " + line2_pitched)
print("Model prediction:", pred_demo)
