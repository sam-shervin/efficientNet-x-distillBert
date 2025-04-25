# train.py
import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast, GradScaler
from transformers import DistilBertTokenizer, AutoModel
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import (
    Config, PrecachedDataset,
    ContrastiveModel, compute_retrieval_metrics
)
import matplotlib.pyplot as plt
flag = True
def validate(model, loader, tokenizer, device):
    model.eval()
    all_img_embs = []
    all_txt_embs = []
    
    with torch.no_grad():
        for imgs, caps in tqdm(loader, desc="Validating", leave=False):
            
                

            # Process images
            imgs = imgs.to(device)
            
            # Process text
            caps = [
                c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else
                (c if isinstance(c, str) else '')
                for c in caps
            ]
            enc = tokenizer(caps, padding=True, truncation=True, return_tensors='pt').to(device)
            
            # Get embeddings
            img_emb, txt_emb = model.encode(imgs, enc.input_ids, enc.attention_mask)
            all_img_embs.append(img_emb.cpu())
            all_txt_embs.append(txt_emb.cpu())
    
    # Concatenate all embeddings
    all_img_embs = torch.cat(all_img_embs, dim=0).to(device)
    all_txt_embs = torch.cat(all_txt_embs, dim=0).to(device)
    
    # Calculate full similarity matrix
    logits = (all_img_embs @ all_txt_embs.T) / model.temperature
    labels = torch.arange(len(all_img_embs), device=device)
    
    # Compute metrics on full dataset
    metrics = compute_retrieval_metrics(logits, labels)
    return metrics

def plot_and_save(xs, ys_dict, x_label, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for metric, ys in ys_dict.items():
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(x_label)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_label}')
        plt.savefig(os.path.join(out_dir, f'{prefix}_{metric}.png'))
        plt.close()

def main():
    cfg = Config()
    print(f'Config: {cfg}')
    device = torch.device(cfg.device)
    print(f'Using device: {device}')
    
    # List and shuffle all .pt files in the precache directory
    precache_dir = 'precache'
    all_files = sorted([os.path.join(precache_dir, f) for f in os.listdir(precache_dir) if f.endswith('.pt')])
    # Shuffle files to ensure random train/val split
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(all_files)
    
    # Split files into training and validation
    total_files = len(all_files)
    train_files_count = int(0.99 * total_files)
    train_files = all_files[:train_files_count]
    val_files = all_files[train_files_count:]
    
    # Create datasets
    train_ds = PrecachedDataset(train_files)
    val_ds = PrecachedDataset(val_files)
    
    print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}')
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_memory,
        shuffle=False,  # IterableDataset doesn't support shuffle
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_memory,
        shuffle=False,
    )
    print("Data loaders created.")

    # model + optim
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    effnet = models.efficientnet_b0(pretrained=True)
    img_enc = torch.nn.Sequential(*list(effnet.children())[:-1])
    img_enc.output_dim = effnet.classifier[1].in_features
    txt_enc = AutoModel.from_pretrained('distilbert-base-uncased')


    model = ContrastiveModel(img_enc, txt_enc, cfg).to(device)
    print("Model created.")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler()

    # ——— Resume from latest checkpoint if present ———
    import glob
    checkpoint_files = glob.glob('checkpoint_epoch*.pt')
    if checkpoint_files:
        # sort by epoch number in filename
        ckpts = sorted(
            checkpoint_files,
            key=lambda fn: int(fn.split('checkpoint_epoch')[1].split('.pt')[0])
        )
        latest_ckpt = ckpts[-1]
        ckpt = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        epoch_indices = ckpt['epoch_indices']
        epoch_metrics = ckpt['epoch_metrics']
        epoch_losses = ckpt['epoch_losses']

        print(f"Loaded checkpoint {latest_ckpt}, resuming from epoch {start_epoch}")
    else:
        start_epoch = 1
        epoch_metrics = []
        epoch_indices = []
        epoch_losses = []
        print("No checkpoint found, starting from scratch")

    print("Optimizer created.")
    print("Training...")



    total_batches = len(train_loader)
    print(f'Total batches: {total_batches}')

    for epoch in range(start_epoch, cfg.epochs+1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for idx, (images, caps) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"),
                start=1
            ):
            
            images = images.to(device)
            caps = [
                c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else
                (c if isinstance(c, str) else '')
                for c in caps
            ]
            enc = tokenizer(caps, padding=True, truncation=True, return_tensors='pt').to(device)

            opt.zero_grad()
            with autocast(device_type=device.type):
                loss, _, _ = model(images, enc.input_ids, enc.attention_mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if idx % 50 == 0:
                tqdm.write(f'Batch Loss: {loss.item():.4f}')
            epoch_loss += loss.item()
            num_batches += 1

        # epoch validation
        m = validate(model, val_loader, tokenizer, device)
        epoch_metrics.append(m)
        epoch_indices.append(epoch)
        # compute & store avg train loss
        avg_loss = epoch_loss / num_batches if num_batches>0 else 0.0
        epoch_losses.append(avg_loss)
        # one unified plot: loss + all retrieval metrics
        ys = {'loss': epoch_losses}
        ys.update({k: [em[k] for em in epoch_metrics] for k in m})
        plot_and_save(
            epoch_indices,
            ys,
            x_label='Epochs',
            out_dir='plots/epochs',
            prefix=f'epoch_{epoch}'
        )
        # ——— checkpoint: model + optimizer + scaler + training state ———
        ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch_indices': epoch_indices,
        'epoch_metrics': epoch_metrics,
        'epoch_losses': epoch_losses,
        }
        torch.save(ckpt, f'checkpoint_epoch{epoch}.pt')
        print(f'>> Epoch {epoch} done, checkpoint saved.')

    print("Training complete.")

if __name__ == '__main__':
    main()
