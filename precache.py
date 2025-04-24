# precache.py
import os
import math
import torch
import kagglehub
from modules import Config, make_dataset_paths, LAIONIterableDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def precache():
    os.environ['KAGGLEHUB_CACHE'] = './kaggle_cache'
    cfg = Config()

    # download parquet files
    parquet_dir = kagglehub.dataset_download(cfg.kaggle_dataset)
    files = make_dataset_paths(parquet_dir, cfg.num_parquet_files)

    # preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor()
    ])

    ds = LAIONIterableDataset(files, transform=transform, max_samples=cfg.max_samples)
    os.makedirs('precache', exist_ok=True)

    # DataLoader pipeline with multiple workers + prefetch
    total_batches = math.ceil(len(ds) / 32)
    loader = DataLoader(
        ds,
        batch_size=32,
        num_workers=cfg.num_workers,
        pin_memory=False,
        prefetch_factor=cfg.prefetch_factor
    )

    # use manual tqdm so we can call set_postfix()
    with tqdm(total=total_batches, desc="Pre-caching", unit="batch") as pbar:
        for batch_idx, (imgs, caps) in enumerate(loader):
            torch.save(
                {'images': imgs, 'captions': caps},
                f'precache/batch_{batch_idx:05d}.pt'
            )
            # update bar
            pbar.update(1)
            # show how many samples we've done so far
            pbar.set_postfix(samples=(batch_idx + 1) * 32)

    print("Pre‑caching complete.")

if __name__ == '__main__':
    precache()
