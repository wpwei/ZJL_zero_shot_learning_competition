import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, channel, reduction=False):
        super().__init__()
        out_channel = channel * 2 if reduction else channel
        stride = 2 if reduction else 1

        self.cnn = nn.Sequential(*[
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, out_channel, 3, stride=stride, padding=1)
        ])

    def forward(self, x):
        return self.cnn(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(*[
            nn.Conv2d(3, 16, 3, padding=1),  # 16 x 64 x 64

            ConvBlock(16),
            ConvBlock(16, True),  # 32 x 32 x 32

            ConvBlock(32),
            ConvBlock(32, True),  # 64 x 16 x 16

            ConvBlock(64),
            ConvBlock(64, True),  # 128 x 8 x 8

            ConvBlock(128),
            ConvBlock(128, True),  # 256 x 4 x 4

            nn.AvgPool2d(3, stride=1)  # 256 x 2 x 2
        ])

        self.classifier = nn.Linear(1024, 230)

    def forward(self, x):
        feature = self.cnn(x).view(x.size(0), -1)
        logits = self.classifier(feature)
        return feature, logits

    def get_loss(self, x, label):
        _, logits = self.forward(x)
        return F.cross_entropy(logits, label)

    def predict(self, x):
        with torch.no_grad():
            _, logits = self.forward(x)
            pred = logits.max(1)[1]
        return pred.detach().cpu().numpy()


class DeViSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()
        self.classifier = nn.Sequential(*[
            nn.Linear(1024, 230),
            nn.ReLU()
        ])
        self.visual_transformer = nn.Sequential(*[
            nn.Linear(1024, 1024),
            nn.ReLU()
        ])
        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(300, 1024),
            nn.ReLU()
        ])

    def forward(self, image, word_embeddings):
        n_class = word_embeddings.size(0)
        batch_size = image.size(0)

        image_feature = self.cnn(image)
        logits = self.classifier(image_feature)

        visual_emb = self.visual_transformer(image_feature)
        visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)

        semantic_emb = self.word_emb_transformer(word_embeddings)
        semantic_emb = semantic_emb.repeat(batch_size, 1)

        scores = torch.mul(visual_emb, semantic_emb).sum(dim=1).view(batch_size, -1)

        return logits, scores

    def get_loss(self, image, label, word_embeddings, LAMBDA=0.5):
        logtis, scores = self.forward(image, word_embeddings)
        loss = F.cross_entropy(logtis, label) * LAMBDA + (1 - LAMBDA) * F.multi_margin_loss(scores, label)

        return loss

    def predict(self, image, word_embeddings):
        with torch.no_grad():
            _, score = self.forward(image, word_embeddings)
            pred = score.max(1)[1]
        return pred.detach().cpu().numpy()


