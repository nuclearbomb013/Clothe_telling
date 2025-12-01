"""
Image Captioning Inference Script

This script loads the trained ViT-Transformer model and generates captions for random images.
It randomly selects 3 images from the JSON file and outputs both the ground truth and predicted captions.
"""

import json
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_vit_transformer import ImageCaptioningModelViT
import os


def load_model_and_vocab(checkpoint_path, device):
    """
    Load the trained model and vocabulary from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        device: torch device (cpu or cuda)

    Returns:
        model: Loaded model in eval mode
        vocab: Vocabulary dictionary
        idx_to_word: Reverse vocabulary (index to word mapping)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract vocabulary
    vocab = checkpoint['vocab']
    print(f"Vocabulary size: {len(vocab)}")

    # Create reverse vocabulary (index to word)
    idx_to_word = {idx: word for word, idx in vocab.items()}

    # Initialize model with same parameters as training
    model = ImageCaptioningModelViT(
        vocab_size=len(vocab),
        embed_size=384,
        decoder_dim=512,
        num_decoder_layers=8,
        num_decoder_heads=8,
        dropout=0.3,
        pad_idx=vocab['<pad>'],
        use_pretrained_encoder=True,
        train_backbone=False
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    print(f"Validation loss: {checkpoint['loss']:.4f}")

    return model, vocab, idx_to_word


def preprocess_image(image_path):
    """
    Load and preprocess an image for the model.

    Args:
        image_path: Path to the image file

    Returns:
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
    """
    # Same transforms as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def decode_caption(token_ids, idx_to_word):
    """
    Convert token IDs to text caption.

    Args:
        token_ids: Tensor or list of token indices
        idx_to_word: Dictionary mapping indices to words

    Returns:
        caption: Decoded text caption
    """
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


def predict_caption(model, image_tensor, vocab, idx_to_word, device, max_len=50):
    """
    Generate caption for an image using the trained model.

    Args:
        model: Trained ImageCaptioningModelViT model
        image_tensor: Preprocessed image tensor [1, 3, 224, 224]
        vocab: Vocabulary dictionary
        idx_to_word: Reverse vocabulary mapping
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
            sos_idx=vocab['<start>'],
            eos_idx=vocab['<end>'],
            max_len=max_len,
            device=device
        )

        # Decode the first (and only) sample in the batch
        caption = decode_caption(sampled_ids[0], idx_to_word)

    return caption


def main():
    # Configuration
    CHECKPOINT_PATH = r"D:\clothes_telling\continue_trained_latest_model.pth"
    JSON_PATH = r"D:\clothes_telling\captions.json"
    IMAGE_ROOT = r"D:\clothes_telling\images"
    NUM_SAMPLES = 3  # Number of random images to test

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model and vocabulary
    model, vocab, idx_to_word = load_model_and_vocab(CHECKPOINT_PATH, device)

    # Load JSON data (read first 30 lines to understand structure, but load all for sampling)
    print(f"\nLoading captions from: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding='utf-8') as f:
        captions_data = json.load(f)

    print(f"Total images in dataset: {len(captions_data)}\n")

    # Randomly select N images
    all_image_names = list(captions_data.keys())
    selected_images = random.sample(all_image_names, NUM_SAMPLES)

    print("="*80)
    print(f"INFERENCE RESULTS - {NUM_SAMPLES} Random Samples")
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
            idx_to_word=idx_to_word,
            device=device,
            max_len=50
        )

        # Display results
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction:   {predicted_caption}")

    print("\n" + "="*80)
    print("Inference completed!")
    print("="*80)


if __name__ == "__main__":
    main()
