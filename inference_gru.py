"""
Image Captioning Inference Script for GRU Model

This script loads the trained ResNet50+GRU model and generates captions for random images.
It randomly selects 3 images from the JSON file and outputs both the ground truth and predicted captions.
"""

import json
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ImageCaptioningModel
import os


# ========== 复用 train.py 中的文本处理函数 ==========
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
    from collections import Counter
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
# ===================================================


def load_model(checkpoint_path, vocab, device):
    """
    Load the trained GRU model.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        vocab: Vocabulary dictionary (must be rebuilt from data)
        device: torch device (cpu or cuda)

    Returns:
        model: Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize model with same parameters as training
    model = ImageCaptioningModel(
        embed_size=256,
        hidden_size=512,
        vocab_size=len(vocab),
        pad_idx=vocab[PAD],
        use_pretrained_encoder=True,
        train_backbone=False
    ).to(device)

    # Load model weights (state_dict)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("GRU Model loaded successfully!")
    return model


def preprocess_image(image_path):
    """
    Load and preprocess an image for the GRU model.
    Note: No normalization is applied, as it wasn't used during training.

    Args:
        image_path: Path to the image file

    Returns:
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Note: No Normalize transform, matching train.py
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def decode_caption(token_ids, vocab):
    """
    Convert token IDs to text caption.

    Args:
        token_ids: Tensor or list of token indices
        vocab: Vocabulary dictionary

    Returns:
        caption: Decoded text caption
    """
    # Create reverse vocabulary
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    words = []
    for idx in token_ids:
        word = idx_to_word.get(idx, '<unk>')
        # Stop at end token or padding
        if word in ['<end>', '<pad>']:
            break
        # Skip start token
        if word == '<start>':
            continue
        words.append(word)

    caption = ' '.join(words)
    return caption


def predict_caption(model, image_tensor, vocab, device, max_len=30):
    """
    Generate caption for an image using the trained GRU model.

    Args:
        model: Trained ImageCaptioningModel model
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        vocab: Vocabulary dictionary
        device: torch device
        max_len: Maximum generation length

    Returns:
        caption: Predicted caption text
    """
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        # Generate token IDs using the model's generate method
        sampled_ids = model.generate(
            images=image_tensor,
            sos_idx=vocab[SOS],
            eos_idx=vocab[EOS],
            max_len=max_len,
            device=device
        )

        # Decode the first (and only) sample in the batch
        caption = decode_caption(sampled_ids[0], vocab)

    return caption


def main():
    # Configuration
    CHECKPOINT_PATH = r"D:\clothes_telling\trained_model_gru.pth"
    JSON_PATH = r"D:\clothes_telling\captions.json"
    IMAGE_ROOT = r"D:\clothes_telling\images"
    NUM_SAMPLES = 3  # Number of random images to test

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load JSON data to rebuild vocabulary
    print(f"Loading captions from: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding='utf-8') as f:
        captions_data = json.load(f)

    print(f"Total images in dataset: {len(captions_data)}")
    
    # Rebuild vocabulary exactly as in train.py
    captions = [cap for _, cap in captions_data.items()]
    vocab = build_vocab(captions, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}\n")

    # Load model
    model = load_model(CHECKPOINT_PATH, vocab, device)

    # Randomly select N images
    all_image_names = list(captions_data.keys())
    selected_images = random.sample(all_image_names, NUM_SAMPLES)

    print("="*80)
    print(f"GRU INFERENCE RESULTS - {NUM_SAMPLES} Random Samples")
    print("="*80)

    # Process each selected image
    for idx, image_name in enumerate(selected_images, 1):
        print(f"\n[Sample {idx}/{NUM_SAMPLES}]")
        print(f"Image: {image_name}")
        print("-" * 80)

        # Get ground truth caption
        ground_truth = captions_data[image_name]

        # Construct image path
        image_path = os.path.join(IMAGE_ROOT, image_name)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠ Warning: Image not found at {image_path}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Prediction: [SKIPPED - Image not found]")
            continue

        # Preprocess image
        image_tensor = preprocess_image(image_path)

        # Generate prediction
        predicted_caption = predict_caption(
            model=model,
            image_tensor=image_tensor,
            vocab=vocab,
            device=device,
            max_len=30
        )

        # Display results
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction:   {predicted_caption}")

    print("\n" + "="*80)
    print("GRU Inference completed!")
    print("="*80)


if __name__ == "__main__":
    import re
    main()
