import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================
# CONFIG
# =========================

IMG_DIR = "img"

TRAIN_LABELS = "train_labels.txt"
VAL_LABELS = "val_labels.txt"

BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#ALPHABET = "0123456789ABCEHKMOPTXY"
# ALPHABET = "0123456789ABCEGHIKMOPTXY"
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# =========================
# CHAR MAPS
# =========================

char_to_int = {c: i + 1 for i, c in enumerate(ALPHABET)}
int_to_char = {i + 1: c for i, c in enumerate(ALPHABET)}

# 0 = CTC blank
int_to_char[0] = ""

# =========================
# DATASET
# =========================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class OCRDataset(Dataset):

    def __init__(self, labels_file):

        self.samples = []

        with open(labels_file, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 2:
                    continue

                image_name, text = parts

                self.samples.append((image_name, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_name, text = self.samples[idx]

        image_path = os.path.join(IMG_DIR, image_name)

        if not os.path.exists(image_path):
            print("Missing:", image_path)
            return self.__getitem__((idx + 1) % len(self.samples))

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        target = torch.tensor(
            [char_to_int[c] for c in text],
            dtype=torch.long
        )

        return image, target, text  

# =========================
# COLLATE
# =========================

def collate_fn(batch):

    images = []
    targets = []
    target_lengths = []

    texts = []

    for image, target, text in batch:

        images.append(image)

        targets.extend(target)

        target_lengths.append(len(target))

        texts.append(text)

    images = torch.stack(images)

    targets = torch.tensor(targets, dtype=torch.long)

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths, texts

# =========================
# CRNN
# =========================

class CRNN(nn.Module):

    def __init__(self, num_classes):

        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1))
        )

        self.rnn = nn.LSTM(
            512 * 2,
            256,
            bidirectional=True,
            num_layers=2,
            batch_first=True
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.cnn(x)

        batch, channels, height, width = x.size()

        x = x.reshape(batch, channels * height, width)

        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.classifier(x)

        x = x.permute(1, 0, 2)

        x = nn.functional.log_softmax(x, dim=2)

        return x

# =========================
# LOAD DATA
# =========================

train_dataset = OCRDataset(TRAIN_LABELS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# MODEL
# =========================

num_classes = len(ALPHABET) + 1

model = CRNN(num_classes)


#state_dict = torch.load(
#    "crnn_ua_finetuned.pth",
#    map_location=DEVICE
#)


#del state_dict["classifier.weight"]
#del state_dict["classifier.bias"]


#model.load_state_dict(
#    state_dict,
#    strict=False
#)

model = model.to(DEVICE)

print("loaded")

# =========================
# LOSS / OPTIMIZER
# =========================

criterion = nn.CTCLoss(blank=0)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# TRAIN
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, targets, target_lengths, texts in train_loader:

        images = images.to(DEVICE)

        targets = targets.to(DEVICE)

        preds = model(images)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=preds.size(0),
            dtype=torch.long
        )

        loss = criterion(
            preds,
            targets,
            input_lengths,
            target_lengths
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"Loss: {total_loss:.4f}")

    torch.save(
        model.state_dict(),
        f"checkpoint_epoch_{epoch+1}.pth"
    )
# =========================
# SAVE
# =========================

torch.save(
    model.state_dict(),
    "crnn_ua_finetuned_new.pth"
)

print("Training complete")
