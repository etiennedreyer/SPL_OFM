import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse

from util.dataset import CaloDataset

def get_data(data_path, splits, batch_size=256, workers=6):

    # Load dataset
    dataset = CaloDataset(data_path, downsample_factors=[4,4,4], layer=None)
    total_size = len(dataset)
    
    # Calculate split sizes
    assert sum(splits.values()) <= 1.0, "Splits must sum to 1.0 or less"

    splits = {k: int(v * total_size) for k, v in splits.items()}

    print("Dataset sizes:")
    print(splits)

    ranges = {}
    ranges['train'] = (0, splits['train'])
    ranges['val'] = (ranges['train'][1], ranges['train'][1] + splits['val'])
    ranges['test'] = (ranges['val'][1], ranges['val'][1] + splits['test'])

    # Create data loaders
    dls = {k: DataLoader(Subset(dataset, range(v[0], v[1])), 
                            batch_size=batch_size, 
                            shuffle=(k=='train'), num_workers=workers)
                         for k, v in ranges.items()
                    }

    return dataset, dls

def get_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from models.fno import FNO

    model = FNO(config['modes'], vis_channels=config['vis_channels'], 
                hidden_channels=config['hidden_channels'], 
                proj_channels=config['proj_channels'], 
                x_dim=3, t_scaling=1)
    model.to(device)

    Nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {Nparams}")

    from ofm_OT_likelihood import OFMModel

    # GP hyperparameters
    n_z = 8
    n_xy = 32
    dims = [n_z, n_xy, n_xy]
    kernel_length=0.01
    kernel_variance=1
    nu = 0.5 # default
    sigma_min=1e-4

    ofm_model = OFMModel(model, 
                         kernel_length=kernel_length, 
                         kernel_variance=kernel_variance, 
                         nu=nu, sigma_min=sigma_min, 
                         dims=dims, device=device)

    return ofm_model


def train(ofm_model, dls, args):

    optimizer = torch.optim.Adam(ofm_model.model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//10, gamma=0.8)

    from pathlib import Path
    save_path = Path(args.save_path)

    ofm_model.train(dls['train'], optimizer, scheduler=scheduler, epochs=args.epochs, 
                    test_loader=dls['val'], eval_int=10,
                    save_int=int(2), saved_model=True, save_path=save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data file')
    parser.add_argument('--config', '-c', type=str, default='./configs/config.yaml', help='Path to the config file')
    parser.add_argument('--train_split', '-ts', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--val_split', '-vs', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--test_split', '-es', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_path', '-sp', type=str, default='./model_checkpoints', help='Path to save model checkpoints')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    splits = {
        'train': args.train_split,
        'val': args.val_split,
        'test': args.test_split
    }
    ds, dls = get_data(args.data, splits, args.batch_size)

    # Get model
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    ofm_model = get_model(config)

    train(ofm_model, dls, args)