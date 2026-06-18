import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
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


#OCR_ALPHABET = '0123456789ABCEHKMOPTXY'  
OCR_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_classes = len(OCR_ALPHABET) + 1

model = CRNN(num_classes)

state_dict = torch.load(
    "checkpoint_epoch_80.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

dummy_input = torch.randn(1, 1, 32, 128)

torch.onnx.export(
    model,
    dummy_input,
    "crnn_ua_80.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {1: "batch"}
    }
)

print("CRNN ONNX export complete")
