import uproot
import torch
from torch.utils.data import Dataset

class CaloDataset(Dataset):

    def __init__(self, data_file_path,
                    Nx = 128, Ny = 128, Nz = 64,
                    downsample_factors: list | None = None):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        if downsample_factors is not None:
            assert len(downsample_factors) == 3
            for dsf in downsample_factors:
                assert Nx % dsf == 0
                assert Ny % dsf == 0
                assert Nz % dsf == 0
            self.dsfs = downsample_factors

        with uproot.open(data_file_path) as f:
            tree = f["Cells"]
            self.data = tree.arrays(["cell_e_dep", 
                                     "cell_idx_x", 
                                     "cell_idx_y", 
                                     "cell_idx_z"], 
                                    library="np")

    def __len__(self):
        return len(self.data["cell_e_dep"])

    def pad_single_example(self, e, idx_x, idx_y, idx_z):

        ### Create 3D voxel grid from sparse representation
        voxels = torch.zeros(self.Nx, self.Ny, self.Nz, dtype=torch.float32)

        voxels[idx_x, idx_y, idx_z] = torch.tensor(e, dtype=torch.float32)
        return voxels
    
    def downsample_single_example(self, voxels, dsfs):

        ### sum pooling to downsample
        dsf_x, dsf_y, dsf_z = dsfs
        voxels = voxels.unfold(0, dsf_x, dsf_x).sum(dim=-1)
        voxels = voxels.unfold(1, dsf_y, dsf_y).sum(dim=-1)
        voxels = voxels.unfold(2, dsf_z, dsf_z).sum(dim=-1)

        return voxels

    def __getitem__(self, idx, downsample=True):
    
        e = self.data["cell_e_dep"][idx]
        idx_x = self.data["cell_idx_x"][idx]
        idx_y = self.data["cell_idx_y"][idx]
        idx_z = self.data["cell_idx_z"][idx]

        voxels = self.pad_single_example(e, idx_x, idx_y, idx_z)

        if downsample:
            voxels = self.downsample_single_example(voxels, self.dsfs)

        ### put layer dimension first (channel)
        voxels = voxels.permute(2, 0, 1)

        voxels = voxels[:1, :, :] # HACK: using only first layer for 2D FNO

        return voxels
