import os
import json

IMG_DIR = "img"
ANN_DIR = "ann"

output = open("labels.txt", "w", encoding="utf-8")

count = 0

for file in os.listdir(ANN_DIR):

    if not file.endswith(".json"):
        continue

    json_path = os.path.join(ANN_DIR, file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

   
    label = data.get("description", "").strip()

    if len(label) == 0:
        continue

   
    image_name = file.replace(".json", ".png")

    image_path = os.path.join(IMG_DIR, image_name)

    
    if not os.path.exists(image_path):
        print("Missing image:", image_name)
        continue

    
    output.write(f"{image_name} {label}\n")

    count += 1

output.close()

print("Done:", count)