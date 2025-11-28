import json
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_vit_transformer import ImageCaptioningModelViT
import os

# 清洗文本
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 分词
def tokenize(text):
    return text.split()

PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"
UNK = "<unk>"

# 构建词表
def build_vocab(captions, min_freq=1):
    counter = Counter()
    for cap in captions:
        tokens = tokenize(clean_text(cap))
        counter.update(tokens)

    vocab = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
    index = 4

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index += 1

    return vocab

def numericalize(text, vocab):
    tokens = tokenize(clean_text(text))
    ids = [vocab[SOS]]

    for t in tokens:
        ids.append(vocab.get(t, vocab[UNK]))

    ids.append(vocab[EOS])
    return ids

def pad_sequence(sequences, max_len, pad_idx=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return torch.tensor(padded)

def collate_fn(batch):
    # batch: [(image, caption_ids), ...]
    images = []
    captions = []
    lengths = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)
        lengths.append(len(cap))

    # 将图片堆叠成 batch
    images = torch.stack(images, dim=0)

    # 动态 padding
    max_len = max(lengths)

    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return images, padded_captions


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab, image_root):
        self.dataset = dataset
        self.vocab = vocab
        self.image_root = image_root

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rel_path, caption = self.dataset[idx]

        # 拼接为绝对路径（不会产生双斜杠）
        img_path = os.path.join(self.image_root, rel_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = numericalize(caption, self.vocab)
        return image, torch.tensor(tokens)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    
    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        # Create targets (shifted captions)
        targets = captions[:, 1:]  # Remove SOS token
        captions_input = captions[:, :-1]  # Remove EOS token
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images, captions_input)
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / total_batches
    print(f'Epoch [{epoch}] Average Loss: {avg_loss:.4f}')
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Create targets (shifted captions)
            targets = captions[:, 1:]  # Remove SOS token
            captions_input = captions[:, :-1]  # Remove EOS token
            
            with autocast():
                outputs = model(images, captions_input)
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / total_batches
    print(f'Validation Epoch [{epoch}] Average Loss: {avg_loss:.4f}')
    return avg_loss


def main():
    # Configuration
    EMBED_SIZE = 384
    DECODER_DIM = 512
    NUM_DECODER_LAYERS = 8
    NUM_DECODER_HEADS = 8
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Paths (as specified in original train.py)
    JSON_PATH = r"D:\clothes_telling\captions.json"
    IMAGE_ROOT = r"D:\clothes_telling\images"
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    dataset = []
    for img_name, caption in data.items():
        dataset.append((f"{img_name}", caption))
    print(f"Loaded {len(dataset)} samples")
    print(f"Sample: {dataset[0]}")
    
    # Create vocabulary
    captions = [cap for _, cap in dataset]
    vocab = build_vocab(captions, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset and dataloader
    caption_dataset = CaptionDataset(dataset, vocab, image_root=IMAGE_ROOT)
    
    train_loader = DataLoader(
        caption_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = ImageCaptioningModelViT(
        vocab_size=len(vocab),
        embed_size=EMBED_SIZE,
        decoder_dim=DECODER_DIM,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_decoder_heads=NUM_DECODER_HEADS,
        dropout=DROPOUT,
        pad_idx=vocab[PAD],
        use_pretrained_encoder=True,
        train_backbone=False
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function (ignore padding index)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate (using same data for now, in practice you'd have separate validation set)
        val_loss = validate_epoch(model, train_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'vocab': vocab
            }, 'best_model_vit_transformer.pth')
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'vocab': vocab
        }, 'latest_model_vit_transformer.pth')
    
    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
