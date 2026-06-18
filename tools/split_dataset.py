import random

with open("labels.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split_index = int(len(lines) * 0.9)

train_lines = lines[:split_index]
val_lines = lines[split_index:]

with open("train_labels.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("val_labels.txt", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Train:", len(train_lines))
print("Val:", len(val_lines))