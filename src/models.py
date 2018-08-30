import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM(nn.Module):
    def __init__(self, cnn, semantic_dim=300, hidden_dim=1024):
        super().__init__()
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False

        visual_dim = cnn.classifier.in_features

        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim),
            nn.ReLU(),
        ])

    def forward(self, image, label, word_embeddings):
        self.cnn.eval()
        visual_emb, _ = self.cnn(image)
        semantic_emb = self.word_emb_transformer(word_embeddings[label])

        return semantic_emb, visual_emb

    def get_loss(self, image, label, word_embeddings):
        loss = F.mse_loss(*self.forward(image, label, word_embeddings))

        return loss

    def predict(self, image, word_embeddings):
        with torch.no_grad():
            n_class = word_embeddings.size(0)
            batch_size = image.size(0)

            self.cnn.eval()
            visual_emb, _ = self.cnn(image)

            visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
            semantic_emb = self.word_emb_transformer(word_embeddings).repeat(batch_size, 1)

            score = (visual_emb - semantic_emb).norm(dim=1).view(batch_size, n_class)
            pred = score.min(1)[1]

        return pred.detach().cpu().numpy()


