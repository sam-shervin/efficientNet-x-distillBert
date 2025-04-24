# validate_checkpoint.py
import os
import torch
import glob
import random
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, AutoModel
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import (
    Config, PrecachedDataset,
    ContrastiveModel, compute_retrieval_metrics
)

def validate(model, loader, tokenizer, device):
    """Compute both image-to-text and text-to-image retrieval metrics"""
    model.eval()
    all_img_embs = []
    all_txt_embs = []
    
    with torch.no_grad():
        for imgs, caps in tqdm(loader, desc="Validating"):
            # Process images
            imgs = imgs.to(device)
            
            # Process text
            caps = [
                c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else
                (c if isinstance(c, str) else '')
                for c in caps
            ]
            enc = tokenizer(caps, padding=True, truncation=True, 
                           return_tensors='pt').to(device)
            
            # Get embeddings
            img_emb, txt_emb = model.encode(imgs, enc.input_ids, enc.attention_mask)
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
    
    # Concatenate all embeddings
    all_img_embs = torch.cat(all_img_embs, dim=0).to(device)
    all_txt_embs = torch.cat(all_txt_embs, dim=0).to(device)
    
    # Calculate similarity matrices with temperature
    temperature = torch.exp(model.temperature).item()
    logits_i2t = (all_img_embs @ all_txt_embs.T) 
    logits_t2i = logits_i2t.T
    
    # Compute metrics for both directions
    labels = torch.arange(len(all_img_embs), device=device)
    metrics_i2t = compute_retrieval_metrics(logits_i2t, labels)
    metrics_t2i = compute_retrieval_metrics(logits_t2i, labels)
    
    return metrics_i2t, metrics_t2i

def load_model(checkpoint_path, device):
    """Load model architecture and weights from checkpoint"""
    cfg = Config()
    
    # Create base encoders
    effnet = models.efficientnet_b0(pretrained=True)
    img_enc = torch.nn.Sequential(*list(effnet.children())[:-1])
    img_enc.output_dim = effnet.classifier[1].in_features
    
    txt_enc = AutoModel.from_pretrained('distilbert-base-uncased')
    
    # Initialize model
    model = ContrastiveModel(img_enc, txt_enc, cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model

def get_val_loader():
    """Create validation DataLoader matching training setup"""
    cfg = Config()
    precache_dir = 'precache'
    
    # Replicate original dataset split
    all_files = sorted([
        os.path.join(precache_dir, f) 
        for f in os.listdir(precache_dir) 
        if f.endswith('.pt')
    ])
    random.seed(42)  # Match training seed
    random.shuffle(all_files)
    
    # Use last 2% for validation
    total_files = len(all_files)
    val_files = all_files[int(0.9 * total_files):]
    
    val_ds = PrecachedDataset(val_files)
    return DataLoader(
        val_ds,
        batch_size=64,
        num_workers=16,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=False,
        shuffle=False,
    )

def plot_metrics(metrics_i2t, metrics_t2i, checkpoint_name):
    """Plot retrieval metrics for both directions"""
    ks = [1, 2, 3, 5]
    i2t_accs = [metrics_i2t[f'acc@{k}'] for k in ks]
    t2i_accs = [metrics_t2i[f'acc@{k}'] for k in ks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, i2t_accs, marker='o', label='Image-to-Text')
    plt.plot(ks, t2i_accs, marker='o', label='Text-to-Image')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title(f'Retrieval Metrics - {checkpoint_name}')
    plt.xticks(ks)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # List available checkpoints
    checkpoints = sorted(glob.glob("checkpoint_epoch*.pt"))
    if not checkpoints:
        print("No checkpoint files found!")
        return
    
    print("Available checkpoints:")
    for i, path in enumerate(checkpoints):
        print(f"[{i+1}] {os.path.basename(path)}")
    
    # User selection
    try:
        selection = int(input("\nEnter checkpoint number: ")) - 1
        checkpoint_path = checkpoints[selection]
    except (ValueError, IndexError):
        print("Invalid selection!")
        return
    
    # Load components
    model = load_model(checkpoint_path, device)
    val_loader = get_val_loader()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Compute metrics
    metrics_i2t, metrics_t2i = validate(model, val_loader, tokenizer, device)
    
    # Print results
    print("\nImage-to-Text Retrieval:")
    for k in [1, 2, 3, 5]:
        print(f"acc@{k}: {metrics_i2t[f'acc@{k}']:.4f}")
    
    print("\nText-to-Image Retrieval:")
    for k in [1, 2, 3, 5]:
        print(f"acc@{k}: {metrics_t2i[f'acc@{k}']:.4f}")
    
    # Plot results
    plot_metrics(metrics_i2t, metrics_t2i, os.path.basename(checkpoint_path))

if __name__ == "__main__":
    main()