"""
继续训练脚本 - 从checkpoint恢复并实现早停机制

功能:
1. 从现有的best_model_vit_transformer.pth加载模型继续训练
2. 自动保存验证loss最低的模型
3. 早停机制避免过拟合
4. 支持训练/验证集分割
"""

import json
import re
import os
from collections import Counter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from code.model_vit_transformer import ImageCaptioningModelViT

# ========== 数据处理函数 ==========

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"
UNK = "<unk>"

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

def collate_fn(batch):
    images = []
    captions = []
    lengths = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)
        lengths.append(len(cap))

    images = torch.stack(images, dim=0)
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
        img_path = os.path.join(self.image_root, rel_path)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = numericalize(caption, self.vocab)
        return image, torch.tensor(tokens)


# ========== 训练和验证函数 ==========

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    total_batches = len(dataloader)

    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        targets = captions[:, 1:]
        captions_input = captions[:, :-1]

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, captions_input)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / total_batches
    return avg_loss


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_batches = len(dataloader)

    for images, captions in dataloader:
        images = images.to(device)
        captions = captions.to(device)

        targets = captions[:, 1:]
        captions_input = captions[:, :-1]

        with autocast():
            outputs = model(images, captions_input)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

        total_loss += loss.item()

    avg_loss = total_loss / total_batches
    return avg_loss


# ========== 早停类 ==========

class EarlyStopping:
    """早停机制: 当验证loss不再下降时提前停止训练"""

    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        """
        Args:
            patience: 在验证loss不再改善的情况下等待的epoch数
            min_delta: 被认为是改善的最小变化量
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered!')
        else:
            self.best_loss = val_loss
            self.counter = 0


# ========== 主训练函数 ==========

def main():
    # ========== 配置参数 ==========
    EMBED_SIZE = 384
    DECODER_DIM = 512
    NUM_DECODER_LAYERS = 8
    NUM_DECODER_HEADS = 8
    DROPOUT = 0.3
    BATCH_SIZE = 32
    NUM_EPOCHS = 100  # 最大训练轮数
    LEARNING_RATE = 5e-5  # 继续训练使用更小的学习率
    WEIGHT_DECAY = 1e-4

    # 早停参数
    EARLY_STOP_PATIENCE = 15  # 15个epoch没有改善就停止
    MIN_DELTA = 0.0001  # 最小改善阈值

    # 验证集比例
    VAL_SPLIT = 0.1  # 10%作为验证集

    # 路径配置
    JSON_PATH = r"D:\clothes_telling\captions.json"
    IMAGE_ROOT = r"D:\clothes_telling\images"
    CHECKPOINT_PATH = r"D:\clothes_telling\best_model_vit_transformer.pth"
    BEST_MODEL_PATH = r"D:\clothes_telling\best_model_continued.pth"
    LATEST_MODEL_PATH = r"D:\clothes_telling\latest_model_continued.pth"

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== 加载数据 ==========
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    dataset = [(img_name, caption) for img_name, caption in data.items()]
    print(f"Total samples: {len(dataset)}")

    # ========== 加载checkpoint或创建新词表 ==========
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # 从checkpoint加载词表
        if 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
            print(f"Loaded vocab from checkpoint. Vocab size: {len(vocab)}")
        else:
            print("No vocab in checkpoint, building new vocab...")
            captions = [cap for _, cap in dataset]
            vocab = build_vocab(captions, min_freq=1)
            print(f"Built new vocab. Vocab size: {len(vocab)}")

        start_epoch = checkpoint.get('epoch', 0)
        previous_best_loss = checkpoint.get('loss', float('inf'))
        print(f"Checkpoint info:")
        print(f"  - Epoch: {start_epoch}")
        print(f"  - Best loss: {previous_best_loss:.4f}")
    else:
        print(f"\nNo checkpoint found at {CHECKPOINT_PATH}")
        print("Building vocab from scratch...")
        captions = [cap for _, cap in dataset]
        vocab = build_vocab(captions, min_freq=1)
        print(f"Vocab size: {len(vocab)}")
        start_epoch = 0
        previous_best_loss = float('inf')

    # ========== 创建数据集 ==========
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)

    full_dataset = CaptionDataset(dataset, vocab, image_root=IMAGE_ROOT)

    # 分割训练集和验证集
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可复现
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # ========== 初始化模型 ==========
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)

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

    # ========== 加载模型权重 ==========
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading model weights from checkpoint...")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully!")
    else:
        print("\nNo checkpoint found, training from scratch...")

    # ========== 设置优化器和调度器 ==========
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # 如果checkpoint包含优化器状态，可以选择加载（继续训练时可选）
    if os.path.exists(CHECKPOINT_PATH) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded from checkpoint")
        except:
            print("Could not load optimizer state, using fresh optimizer")

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    # 混合精度训练
    scaler = GradScaler()

    # 早停机制
    early_stopping = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=MIN_DELTA,
        verbose=True
    )

    # ========== 训练循环 ==========
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Starting from epoch: {start_epoch + 1}")
    print(f"  - Max epochs: {NUM_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"  - Validation split: {VAL_SPLIT * 100}%")
    print("="*60 + "\n")

    best_val_loss = previous_best_loss

    for epoch in range(start_epoch + 1, start_epoch + NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{start_epoch + NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        print(f'Epoch [{epoch}] Train Loss: {train_loss:.4f}')

        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f'Epoch [{epoch}] Validation Loss: {val_loss:.4f}')

        # 更新学习率
        scheduler.step()

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement = previous_best_loss - val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss': val_loss,  # 兼容inference.py
                'vocab': vocab,
                'config': {
                    'embed_size': EMBED_SIZE,
                    'decoder_dim': DECODER_DIM,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'num_decoder_heads': NUM_DECODER_HEADS,
                    'dropout': DROPOUT
                }
            }, BEST_MODEL_PATH)
            print(f'✓ Saved best model! Val loss: {val_loss:.4f} (improved by {improvement:.4f})')

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss': val_loss,  # 兼容inference.py
            'vocab': vocab,
            'config': {
                'embed_size': EMBED_SIZE,
                'decoder_dim': DECODER_DIM,
                'num_decoder_layers': NUM_DECODER_LAYERS,
                'num_decoder_heads': NUM_DECODER_HEADS,
                'dropout': DROPOUT
            }
        }, LATEST_MODEL_PATH)

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Latest model saved to: {LATEST_MODEL_PATH}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
