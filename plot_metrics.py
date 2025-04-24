# plot_metrics.py
import torch
import matplotlib.pyplot as plt
import os

def plot(xs, ys, x_label, metric, out_dir, prefix):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.title(f'{metric} vs {x_label}')
    plt.savefig(os.path.join(out_dir, f'{prefix}_{metric}.png'))
    plt.close()

def main():
    data = torch.load('metrics_details.pt')
    ei = data['epoch_indices']
    em = data['epoch_metrics']
    el = data['epoch_losses']

    os.makedirs('replots/epochs', exist_ok=True)
    # Plot accuracy metrics + loss
    for k in list(em[0].keys()) + ['loss']:
        if 'recall' in k: continue  # Skip any remaining recall entries
        ys = [m[k] for m in em] if k != 'loss' else el
        plot(ei, ys, 'Epochs', k, 'replots/epochs', 'epoch')

if __name__ == '__main__':
    main()
