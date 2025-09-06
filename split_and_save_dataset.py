import random

random.seed(42)

with open('Datasets/filtered_sanskritdoc.txt', 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]

random.shuffle(sentences)

train_size = int(len(sentences) * 0.7)
val_size = int(len(sentences) * 0.15)

train_set = sentences[:train_size]
val_set = sentences[train_size:train_size + val_size]
test_set = sentences[train_size + val_size:]

with open('Datasets/train.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(train_set))

with open('Datasets/val.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(val_set))

with open('Datasets/test.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(test_set))

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")
