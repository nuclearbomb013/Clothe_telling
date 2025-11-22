# models.py
"""
CNN(ResNet50) encoder + GRU decoder for image captioning.

Usage:
    from models import ImageCaptioningModel
    model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings


class EncoderCNN(nn.Module):
    """
    Encoder: ResNet50 backbone that produces a single global feature vector per image.
    - If use_pretrained=True, it will try to load pretrained weights (requires network).
    - The final feature is projected to `embed_size`.
    """

    def __init__(self, embed_size: int = 256, use_pretrained: bool = False, train_backbone: bool = False):
        super().__init__()
        # ResNet50 backbone
        try:
            # new torchvision versions: weights param; fall back to pretrained for older versions
            resnet = models.resnet50(pretrained=use_pretrained)
        except TypeError:
            # in case your torchvision expects weights=..., try that
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None)
            except Exception:
                # last resort
                resnet = models.resnet50(pretrained=use_pretrained)

        # remove fully connected layer and avgpool (we will keep avgpool for global feature)
        modules = list(resnet.children())[:-1]  # everything except the final fc
        self.backbone = nn.Sequential(*modules)  # outputs [B, 2048, 1, 1]

        # freeze backbone parameters if not training backbone
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = resnet.fc.in_features  # 2048 for resnet50
        self.linear = nn.Linear(feat_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input:
            images: Tensor [B, 3, H, W]
        Output:
            features: Tensor [B, embed_size] - normalized projection of global feature
        """
        x = self.backbone(images)           # [B, 2048, 1, 1]
        x = x.reshape(x.size(0), -1)        # [B, 2048]
        x = self.linear(x)                  # [B, embed_size]
        x = self.bn(x)                      # [B, embed_size]
        x = F.relu(x)
        return x


class DecoderGRU(nn.Module):
    """
    Decoder: Embedding -> GRU -> Linear
    - During training: forward(features, captions) returns logits for each time-step
      (suitable for CrossEntropyLoss with targets being captions[:, 1:] etc.)
    - For inference: sample(image_feature) returns generated token ids (greedy)
    """

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=pad_idx)
        # We'll feed the image feature as the first input vector (projected to embed_size),
        # and optionally use a linear to map feature->initial hidden state
        self.feature2hidden = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_uniform_(self.feature2hidden.weight)
        if self.feature2hidden.bias is not None:
            nn.init.constant_(self.feature2hidden.bias, 0.0)

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        """
        Training forward.

        Inputs:
            features: [B, embed_size]  (output from Encoder)
            captions: [B, T] (token ids)  -- expected to include <start> at captions[:,0]
                      usually captions are padded; T is batch-dependent (collate_fn max_len)
        Returns:
            outputs: [B, T, vocab_size]  logits for each step (aligned with captions input)
                     Note: outputs[:, 0, :] corresponds to step after feeding image feature (or to image token)
        Implementation detail:
            We create inputs to GRU by concatenating `features` as the first "token" (unsqueeze dim1)
            followed by embeddings(captions[:, :-1]) so that target for time t is captions[:, t].
            This follows teacher forcing convention.
        """
        batch_size = features.size(0)
        device = features.device

        # prepare initial hidden state from features
        h0 = torch.tanh(self.feature2hidden(features))  # [B, hidden_size]
        # expand to num_layers: [num_layers, B, hidden_size]
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # get caption embeddings (we will feed embeddings for each time-step)
        embeddings = self.embedding(captions)  # [B, T, embed_size]

        # Construct inputs: [image_feature_as_token, embeddings[:, :-1, :]]
        # Project features to embed_size so dimension matches embedding size (if needed)
        # Here features are already embed_size (since encoder projects to embed_size)
        image_tokens = features.unsqueeze(1)  # [B, 1, embed_size]
        # Optional: you might want to dropout the inputs
        rnn_inputs = torch.cat((image_tokens, embeddings[:, :-1, :]), dim=1)  # [B, T, embed_size]

        # run GRU
        outputs, _ = self.gru(rnn_inputs, h0)  # outputs: [B, T, hidden_size]
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)  # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def sample(self, features: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 30, device=None):
        """
        Greedy decoding (inference)
        Inputs:
            features: [B, embed_size]
            sos_idx: index of <start> token (not strictly needed if we feed features as first step)
            eos_idx: index of <end> token
            max_len: maximum generation length (number of tokens to produce)
        Returns:
            sampled_ids: Tensor [B, L] (L <= max_len) containing generated token ids (excluding <start> token)
        """
        if device is None:
            device = features.device

        batch_size = features.size(0)

        # init hidden state from features
        h = torch.tanh(self.feature2hidden(features))  # [B, hidden_size]
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, hidden_size]

        # first input is the feature vector (already embed_size)
        inputs = features.unsqueeze(1)  # [B, 1, embed_size]

        sampled_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for t in range(max_len):
            out, h = self.gru(inputs, h)  # out: [B, 1, hidden_size]
            out = out.squeeze(1)          # [B, hidden_size]
            logits = self.fc(out)         # [B, vocab_size]
            preds = logits.argmax(dim=-1)  # [B]

            # record and prepare next inputs
            for i in range(batch_size):
                if not finished[i]:
                    sampled_ids[i].append(int(preds[i].item()))
                    if preds[i].item() == eos_idx:
                        finished[i] = True

            # prepare next input embedding
            next_inputs = self.embedding(preds).unsqueeze(1)  # [B, 1, embed_size]
            inputs = next_inputs

            if all(finished):
                break

        # pad to same length for tensor return
        max_out_len = max(len(x) for x in sampled_ids)
        out_tensor = torch.full((batch_size, max_out_len), fill_value=self.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(sampled_ids):
            out_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        return out_tensor


class ImageCaptioningModel(nn.Module):
    """
    High-level wrapper: contains encoder and decoder.
    """

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, dropout: float = 0.3, pad_idx: int = 0,
                 use_pretrained_encoder: bool = True, train_backbone: bool = False):
        super().__init__()
        self.encoder = EncoderCNN(embed_size=embed_size, use_pretrained=use_pretrained_encoder,
                                  train_backbone=train_backbone)
        self.decoder = DecoderGRU(embed_size=embed_size, hidden_size=hidden_size,
                                  vocab_size=vocab_size, num_layers=num_layers,
                                  dropout=dropout, pad_idx=pad_idx)

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        """
        Full forward for training: returns logits for each time-step.
        - images: [B, 3, H, W]
        - captions: [B, T]
        returns logits: [B, T, vocab_size]
        """
        features = self.encoder(images)               # [B, embed_size]
        logits = self.decoder(features, captions)     # [B, T, vocab_size]
        return logits

    def generate(self, images: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 30, device=None):
        """
        Full generate for inference (greedy).
        returns token ids [B, L]
        """
        device = device or next(self.parameters()).device
        images = images.to(device)
        features = self.encoder(images)
        sampled = self.decoder.sample(features, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len, device=device)
        return sampled


# ----------------------------
# Quick smoke-test when run as script
# ----------------------------
if __name__ == "__main__":
    # quick test to assert shapes
    B = 4
    C, H, W = 3, 224, 224
    embed_size = 256
    hidden_size = 512
    vocab_size = 200  # example vocab size
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2
    max_caption_len = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel(embed_size=embed_size,
                                 hidden_size=hidden_size,
                                 vocab_size=vocab_size,
                                 num_layers=1,
                                 dropout=0.2,
                                 pad_idx=pad_idx,
                                 use_pretrained_encoder=False).to(device)

    # fake batch
    images = torch.randn(B, C, H, W).to(device)
    # build fake captions with <start> ... <end> (T)
    captions = torch.randint(low=3, high=vocab_size, size=(B, max_caption_len), dtype=torch.long).to(device)
    captions[:, 0] = sos_idx
    captions[:, -1] = eos_idx

    logits = model(images, captions)  # [B, T, V]
    print("Logits shape:", logits.shape)  # expect [B, T, vocab_size]

    # test generate
    sampled = model.generate(images, sos_idx=sos_idx, eos_idx=eos_idx, max_len=15, device=device)
    print("Sampled shape:", sampled.shape)  # [B, L]
    print("Sampled (first):", sampled[0].tolist())

