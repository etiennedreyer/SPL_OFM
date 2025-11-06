import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse

from util.dataset import CaloDataset, CaloChallengeDataset
import h5py
from tqdm import tqdm



def get_data(config, splits, batch_size=256, workers=6):

    data_path = config['file_path']
    # Load dataset
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        dataset = CaloChallengeDataset(config, entry_start=config.get('entry_start', 0), entry_stop=config.get('entry_stop', None))
    elif data_path.endswith('.root'):
        dataset = CaloDataset(data_path, downsample_factors=[4,4,4], layer=None)
    else:
        raise ValueError("Requires .root or .h5 data file")
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
                         for k, v in ranges.items() if v[1] > v[0]
                    }

    return dataset, dls


def load_checkpoint(model, checkpoint_path):

    for param in model.parameters():
        param.requires_grad = False

    from neuralop.layers.spectral_convolution import SpectralConv
    torch.serialization.add_safe_globals([torch._C._nn.gelu])
    torch.serialization.add_safe_globals([SpectralConv])

    print(f"Loading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if '_metadata' in state_dict:
        del state_dict['_metadata']

    model.load_state_dict(state_dict)


def get_model(config, checkpoint=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from models.fno import FNO, FNO_cond

    model_class = FNO
    if config['file_path'].split('.')[-1] in ['h5', 'hdf5']:
        print("Using conditional FNO model")
        model_class = FNO_cond

    model = model_class(config['modes'], vis_channels=config['vis_channels'], 
                hidden_channels=config['hidden_channels'], 
                proj_channels=config['proj_channels'], 
                x_dim=3, t_scaling=1)
    
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)

    model.to(device)

    Nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {Nparams}")

    from ofm_OT_likelihood import OFMModel

    # GP hyperparameters
    # n_z = 8
    # n_xy = 32
    dims = config['dims']
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

    optimizer = torch.optim.Adam(ofm_model.model.parameters(), lr=5e-4)
    scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//10, gamma=0.7)

    from pathlib import Path
    save_path = Path(args.save_path)

    ofm_model.train(dls['train'], optimizer, scheduler=scheduler, epochs=args.epochs, 
                    test_loader=dls['val'], eval_int=10,
                    save_int=int(2), saved_model=True, save_path=save_path)


def generate(ofm_model, dl_test, config, N):

    n_eval = 24
    method = 'euler'

    for test_batch in dl_test:
        break  # Get the first batch only

    Nvoxels = len(test_batch[0][0].flatten())
    bs = dl_test.batch_size

    with h5py.File('your_output_dataset_name.hdf5', 'w') as f:
        
        ds_energies = f.create_dataset('incident_energies',
                                        shape=(0, 1),
                                        maxshape=(N, 1),
                                        chunks=(bs, 1),
                                        dtype='f4',
                                        compression='gzip')

        ds_showers = f.create_dataset('showers',
                                        shape=(0, Nvoxels),
                                        maxshape=(N, Nvoxels),
                                        chunks=(bs, Nvoxels),
                                        dtype='f4',
                                        compression='gzip')

        for batch in tqdm(dl_test):

            conds = batch[1].to(ofm_model.device)
            current_bs = len(conds)

            samples = ofm_model.sample(config['dims'], conds=conds, n_channels=1,
                                       n_samples=current_bs, n_eval=n_eval, method=method)

            incident_energies = CaloChallengeDataset.transform(conds, 
                                                            'incident_energies', 
                                                            config['transforms'],
                                                            inverse=True).cpu().numpy()

            shower_energies = CaloChallengeDataset.transform(samples.reshape(current_bs, -1),
                                                            'showers', 
                                                            config['transforms'],
                                                            inverse=True).cpu().numpy()

            new_size = ds_showers.shape[0] + current_bs
            ds_showers.resize(new_size, axis=0)
            ds_energies.resize(new_size, axis=0)

            ds_showers[-current_bs:, :] = shower_energies
            ds_energies[-current_bs:, :] = incident_energies


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', '-d', type=str, required=True, help='Path to the data file')
    parser.add_argument('--config', '-c', type=str, default='./configs/config.yaml', help='Path to the config file')
    parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'generate'], help='Mode: train or generate')
    parser.add_argument('--num_samples', '-n', type=int, default=10000, help='Number of samples to generate in generation mode')
    parser.add_argument('--train_split', '-ts', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--val_split', '-vs', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--test_split', '-es', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None, help='Path to model checkpoint for generation')
    parser.add_argument('--save_path', '-sp', type=str, default='./model_checkpoints', help='Path to save model checkpoints')

    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    splits = {
        'train': args.train_split,
        'val': args.val_split,
        'test': args.test_split
    }

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['dims'] = [config['dims']['z'],
                      config['dims']['x'],
                      config['dims']['y']]

    ds, dls = get_data(config, splits, args.batch_size)

    ofm_model = get_model(config, checkpoint=args.checkpoint)

    if args.mode == 'train':
        train(ofm_model, dls, args)
    elif args.mode == 'generate':
        generate(ofm_model, dls['test'], config, args.num_samples)