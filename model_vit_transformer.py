"""
Vision Transformer encoder + Transformer decoder for image captioning.

Usage:
    from model_vit_transformer import ImageCaptioningModelViT
    model = ImageCaptioningModelViT(embed_size=384, decoder_dim=512, vocab_size=len(vocab))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import warnings


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer decoder.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, d_model]
        Returns:
            x: Tensor [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class EncoderViT(nn.Module):
    """
    Encoder: Vision Transformer backbone that produces a sequence of patch features.
    - Uses custom ViT configuration: patch_size=16, embed_dim=384, depth=12, num_heads=8
    - The final features are projected to `embed_size`.
    """

    def __init__(self, embed_size: int = 384, use_pretrained: bool = False, train_backbone: bool = False):
        super().__init__()
        
        # Create custom ViT configuration
        # Note: We'll use torchvision's ViT but with custom parameters
        try:
            # Try to create ViT with custom parameters
            vit = models.vit_b_16(
                pretrained=use_pretrained,
                num_classes=1000  # dummy value, we'll remove the classifier
            )
            # Modify the ViT to our specifications
            # Since torchvision doesn't easily allow custom configs, we'll create our own simplified version
            warnings.warn("Using custom ViT implementation since torchvision doesn't support arbitrary configs")
            
            # Create custom ViT encoder
            self.patch_size = 16
            self.embed_dim = embed_size  # 384 as specified
            self.num_patches = (224 // self.patch_size) ** 2  # 14x14 = 196 patches for 224x224 images
            
            # Patch embedding layer
            self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            
            # Class token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
            # Position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=8,  # as specified
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)  # depth=12 as specified
            
            # Layer norm
            self.norm = nn.LayerNorm(self.embed_dim)
            
            # Projection to embed_size (though it's already embed_size)
            if self.embed_dim != embed_size:
                self.proj = nn.Linear(self.embed_dim, embed_size)
            else:
                self.proj = nn.Identity()
                
        except Exception as e:
            # Fallback to custom implementation
            print(f"Error creating ViT: {e}")
            # Create custom ViT encoder
            self.patch_size = 16
            self.embed_dim = embed_size  # 384 as specified
            self.num_patches = (224 // self.patch_size) ** 2  # 14x14 = 196 patches for 224x224 images
            
            # Patch embedding layer
            self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            
            # Class token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            
            # Position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=8,  # as specified
                dim_feedforward=self.embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)  # depth=12 as specified
            
            # Layer norm
            self.norm = nn.LayerNorm(self.embed_dim)
            
            # Projection to embed_size (though it's already embed_size)
            if self.embed_dim != embed_size:
                self.proj = nn.Linear(self.embed_dim, embed_size)
            else:
                self.proj = nn.Identity()

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'patch_embed'):
            nn.init.xavier_uniform_(self.patch_embed.weight)
            if self.patch_embed.bias is not None:
                nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input:
            images: Tensor [B, 3, H, W] (H=W=224 expected)
        Output:
            features: Tensor [B, embed_size] - global feature from cls token
        """
        B = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [B, num_patches + 1, embed_dim]
        
        # Layer norm
        x = self.norm(x)
        
        # Take cls token as global representation
        cls_features = x[:, 0]  # [B, embed_dim]
        
        # Project to embed_size
        features = self.proj(cls_features)  # [B, embed_size]
        
        return features


class DecoderTransformer(nn.Module):
    """
    Decoder: Transformer-based decoder for image captioning.
    - During training: forward(features, captions) returns logits for each time-step
    - For inference: sample(image_feature) returns generated token ids (greedy)
    """

    def __init__(self, embed_size: int, decoder_dim: int, vocab_size: int,
                 num_layers: int = 8, num_heads: int = 8, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_idx = pad_idx

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_dim, padding_idx=pad_idx)
        
        # Linear projection from encoder features to decoder dimension
        self.feature_proj = nn.Linear(embed_size, decoder_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(decoder_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final classification layer
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_uniform_(self.feature_proj.weight)
        if self.feature_proj.bias is not None:
            nn.init.constant_(self.feature_proj.bias, 0.0)

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        """
        Training forward.

        Inputs:
            features: [B, embed_size]  (output from Encoder)
            captions: [B, T] (token ids)  -- expected to include <start> at captions[:,0]
        Returns:
            outputs: [B, T, vocab_size]  logits for each step
        """
        batch_size = features.size(0)
        device = features.device
        
        # Project features to decoder dimension
        memory = self.feature_proj(features)  # [B, decoder_dim]
        memory = memory.unsqueeze(1)  # [B, 1, decoder_dim]
        
        # Get caption embeddings
        tgt = self.embedding(captions)  # [B, T, decoder_dim]
        tgt = self.pos_encoding(tgt)  # [B, T, decoder_dim]
        
        # Create target mask for causal attention (prevent looking at future tokens)
        T = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        # Create padding mask
        tgt_key_padding_mask = (captions == self.pad_idx)  # [B, T]
        
        # Run transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [B, T, decoder_dim]
        
        output = self.dropout(output)
        logits = self.fc(output)  # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def sample(self, features: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 30, device=None):
        """
        Greedy decoding (inference)
        Inputs:
            features: [B, embed_size]
            sos_idx: index of <start> token
            eos_idx: index of <end> token
            max_len: maximum generation length
        Returns:
            sampled_ids: Tensor [B, L] containing generated token ids
        """
        if device is None:
            device = features.device

        batch_size = features.size(0)
        
        # Project features to decoder dimension
        memory = self.feature_proj(features)  # [B, decoder_dim]
        memory = memory.unsqueeze(1)  # [B, 1, decoder_dim]
        
        # Initialize with SOS token
        input_tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        sampled_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for t in range(max_len):
            # Get embeddings for current input
            tgt = self.embedding(input_tokens)  # [B, current_len, decoder_dim]
            tgt = self.pos_encoding(tgt)
            
            # Create causal mask
            current_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(current_len, current_len, device=device), diagonal=1).bool()
            
            # Run transformer decoder
            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [B, current_len, decoder_dim]
            
            # Get predictions for last token
            logits = self.fc(output[:, -1, :])  # [B, vocab_size]
            preds = logits.argmax(dim=-1)  # [B]
            
            # Record predictions
            for i in range(batch_size):
                if not finished[i]:
                    sampled_ids[i].append(int(preds[i].item()))
                    if preds[i].item() == eos_idx:
                        finished[i] = True
            
            # Prepare next input
            next_input = preds.unsqueeze(1)  # [B, 1]
            input_tokens = torch.cat([input_tokens, next_input], dim=1)  # [B, current_len + 1]
            
            if all(finished):
                break

        # Pad to same length for tensor return
        max_out_len = max(len(x) for x in sampled_ids) if any(sampled_ids) else 1
        out_tensor = torch.full((batch_size, max_out_len), fill_value=self.pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(sampled_ids):
            if len(seq) > 0:
                out_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        return out_tensor


class ImageCaptioningModelViT(nn.Module):
    """
    High-level wrapper: contains ViT encoder and Transformer decoder.
    """

    def __init__(self, vocab_size: int, embed_size: int = 384, decoder_dim: int = 512,
                 num_decoder_layers: int = 8, num_decoder_heads: int = 8, 
                 dropout: float = 0.3, pad_idx: int = 0,
                 use_pretrained_encoder: bool = True, train_backbone: bool = False):
        super().__init__()
        self.encoder = EncoderViT(
            embed_size=embed_size, 
            use_pretrained=use_pretrained_encoder,
            train_backbone=train_backbone
        )
        self.decoder = DecoderTransformer(
            embed_size=embed_size,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            dropout=dropout,
            pad_idx=pad_idx
        )

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
    embed_size = 384  # as specified
    decoder_dim = 512  # as specified
    vocab_size = 200  # example vocab size
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2
    max_caption_len = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModelViT(
        embed_size=embed_size,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        num_decoder_layers=8,  # as specified
        num_decoder_heads=8,   # as specified
        dropout=0.2,
        pad_idx=pad_idx,
        use_pretrained_encoder=False
    ).to(device)

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
