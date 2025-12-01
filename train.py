import json
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ImageCaptioningModel
#清洗文本
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
#分词
def tokenize(text):
    return text.split()
PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"
UNK = "<unk>"
#构建词表
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

    return images, padded_captions, lengths




class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab, image_root):
        self.dataset = dataset
        self.vocab = vocab
        self.image_root = image_root

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rel_path, caption = self.dataset[idx]

        # 拼接为绝对路径（不会产生双斜杠）
        img_path = self.image_root + "/" + rel_path

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = numericalize(caption, self.vocab)
        return image, torch.tensor(tokens)



if __name__ == '__main__':
    with open("D:\clothes_telling\captions.json", "r") as f:
        data = json.load(f)
    dataset = []
    for img_name, caption in data.items():
        dataset.append((f"{img_name}", caption))
    print(dataset[0])
    #创建词表数组
    captions = [cap for _, cap in dataset]
    vocab = build_vocab(captions, min_freq=1)
    print("Vocabulary size:", len(vocab))
    #创建数据集
    image_root = "D:\clothes_telling\images"
    caption_dataset = CaptionDataset(dataset, vocab, image_root=image_root)
    #在数据集里面进行填充

    
    train_loader = DataLoader(
        caption_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # ========== 训练配置 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模型参数
    embed_size = 256
    hidden_size = 512
    num_epochs = 10
    learning_rate = 0.001
    
    # 初始化模型
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        pad_idx=vocab[PAD],
        use_pretrained_encoder=True,
        train_backbone=False
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # ========== 训练循环 ==========
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # 前向传播
            outputs = model(images, captions)  # [B, T, V]
            
            # 计算损失: 预测下一个词
            # outputs[:, :-1, :] 对应输入的 captions[:, :-1]
            # targets 是 captions[:, 1:]
            targets = captions[:, 1:]
            outputs = outputs[:, :-1, :].reshape(-1, len(vocab))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')
    
    # ========== 保存模型 ==========
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Training completed! Model saved as 'trained_model.pth'")
