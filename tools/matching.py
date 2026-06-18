from collections import Counter

labels = {}
results = {}

# labels.txt
with open("labels.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        filename = parts[0]
        plate = parts[1]

        labels[filename] = plate

# result.txt
with open("result80.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if "->" not in line:
            continue

        filename = line.split(" -> ")[0]

        right = line.split(" -> ")[1]

        plate = right.split(" [")[0].strip()

        results[filename] = plate

total = 0
correct = 0

errors = []

char_total = 0
char_correct = 0

confusions = Counter()

for filename, true_plate in labels.items():

    if filename not in results:
        continue

    pred_plate = results[filename]

    total += 1

    if true_plate == pred_plate:
        correct += 1
    else:
        errors.append(
            (filename, true_plate, pred_plate)
        )

    n = min(len(true_plate), len(pred_plate))

    for i in range(n):

        char_total += 1

        if true_plate[i] == pred_plate[i]:
            char_correct += 1
        else:
            char_correct += 0
            confusions[
                f"{true_plate[i]}->{pred_plate[i]}"
            ] += 1

accuracy = 100.0 * correct / total

char_accuracy = (
    100.0 * char_correct / char_total
)

print()
print("========== RESULT ==========")
print("Total:", total)
print("Correct:", correct)
print("Errors:", total - correct)
print("Accuracy: %.4f%%" % accuracy)
print("Char accuracy: %.4f%%" % char_accuracy)

print()
print("========== TOP ERRORS ==========")

for k, v in confusions.most_common(20):
    print(k, v)

print()
print("========== WRONG PLATES ==========")

for fn, gt, pr in errors[:50]:
    print(gt, "->", pr)