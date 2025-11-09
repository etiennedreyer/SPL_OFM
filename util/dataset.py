import uproot
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import h5py

class CaloDataset(Dataset):

    def __init__(self, data_file_path,
                    Nx = 128, Ny = 128, Nz = 64,
                    downsample_factors: list | None = None,
                    drop_last: int = 32,
                    layer: int | None = None):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.drop_last = drop_last
        self.layer = layer
        
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

        if self.drop_last > 0:
            voxels = voxels[:, :, :-self.drop_last]

        if downsample:
            voxels = self.downsample_single_example(voxels, self.dsfs)

        if self.layer is not None:
            ### using only one layer for 2D FNO
            voxels = voxels[:, :, self.layer:self.layer+1].squeeze(-1)

        ### put z before x and y and insert channel dimension
        voxels = voxels.permute(2, 0, 1).unsqueeze(0)

        return voxels


from util.XMLHandler import XMLHandler

class CaloChallengeDataset(Dataset):
    def __init__(self, file_path, config, entry_start=0, entry_stop=None):
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.file_path = file_path
        self.config = config
        self.particle = config['particle']
        self.transform_dict = config.get('transforms', {})
        self.init_vars()
        self.read_data(entry_start, entry_stop)

    def read_data(self, entry_start, entry_stop):

        print(f"Reading {self.file_path}...")
        with h5py.File(self.file_path, "r") as f:
            self.data = {
                "incident_energies": torch.tensor(f["incident_energies"][entry_start:entry_stop]).float(),
                "showers": torch.tensor(f["showers"][entry_start:entry_stop]).float(),
            }

        self.n_events = self.data["showers"].shape[0]

        self.data["showers"] = self.reshape(self.data["showers"])

        self.data = {k: self.transform(v, k, self.transform_dict) for k, v in self.data.items()}


    def __str__(self):

        s = f"CaloChallengeDataset with {self.n_events} {self.particle}s\n"

        s +=f"N cells total: {self.n_cells}\n"
        s +=f"N layers z: {self.n_layers}\n"
        s +=f"N radial bins: {self.n_radial_bins}\n"
        s +=f"N azimuthal bins: {self.n_azimuthal_bins}\n"
        s +=f"N bins per layer: {self.n_bins_per_layer}\n"

        info = self.data
        info["r"] = self.r
        info["phi"] = self.phi
        info["z"] = self.z

        for k, v in info.items():
            s += (
                f"{k}: "
                f"min: {v.min().item():.3f}, "
                f"max: {v.max().item():.3f}, "
                f"mean: {v.mean().item():.3f}, "
                f"std: {v.std().item():.3f}, "
                f"shape: {v.shape}\n"
            )

        return s

    def init_vars(self):
        self.xml = XMLHandler(
            self.particle, self.config['binning']
        )

        # self.bin_edges = self.xml.GetBinEdges()
        # self.eta_all_layers, self.phi_all_layers = self.xml.GetEtaPhiAllLayers()
        # self.relevantLayers = self.xml.GetRelevantLayers()
        # self.layersBinnedInAlpha = self.xml.GetLayersWithBinningInAlpha()
        # self.r_edges = [redge for redge in self.xml.r_edges if len(redge) > 1]
        # self.num_alpha = [len(self.xml.alphaListPerLayer[idx][0]) for idx, redge in \
        #                   enumerate(self.xml.r_edges) if len(redge) > 1]

        self.eta = torch.tensor(np.concatenate(self.xml.eta_all_layers)).float()
        self.phi = torch.tensor(np.concatenate(self.xml.phi_all_layers)).float()
        self.n_cells = len(self.eta)
        self.layers = torch.concat(
            [
                torch.ones(len(el)) * i
                for i, el in enumerate(self.xml.eta_all_layers)
                if len(el) > 0
            ]
        )
        self.r = (self.eta**2 + self.phi**2) ** 0.5
        self.theta = torch.atan2(self.phi, self.eta)
        self.z = self.layers
        self.n_a_bins = np.array(self.xml.a_bins)[
            np.array(self.xml.relevantlayers)
        ]
        self.n_r_bins = np.array(self.xml.r_bins)[
            np.array(self.xml.relevantlayers)
        ]
        self.n_bins = self.n_a_bins * self.n_r_bins
        self.layer_cells = torch.arange(self.n_cells).split(self.n_bins.tolist())

        self.static_features = {
            "r": self.r,
            "phi": self.phi,
            "z": self.z,
        }

        # binning parameters
        self.n_layers = len(self.layer_cells)
        self.n_radial_bins = self.n_r_bins[0]
        self.n_azimuthal_bins = self.n_a_bins[0]
        self.n_bins_per_layer = self.n_bins[0]

        # transform each static feature and reshape
        self.static_features = {
            k: self.reshape(self.transform(v, k, self.transform_dict))
            for k, v in self.static_features.items()
        }

    def reshape(self, x):
        assert x.shape[-1] == self.n_cells
        shape = (self.n_layers, self.n_azimuthal_bins, self.n_radial_bins)
        if x.dim() == 2:
            return x.reshape(x.shape[0], *shape)
        elif x.dim() == 1:
            return x.reshape(*shape)

    @staticmethod
    def transform(x, var, transform_dict, inverse=False):

        if var not in transform_dict:
            raise ValueError(f"Variable {var} not in transforms")

        transforms = {
            "minmax": lambda x, d: (x - d["min"]) / (d["max"] - d["min"]),
            "minmax_inv": lambda x, d: x * (d["max"] - d["min"]) + d["min"],
            "log": lambda x, d: (torch.log(x + d.get("offset", 0)) + d.get("shift", 0)) / d.get("norm", 1),
            "log_inv": lambda x, d: torch.exp(x * d.get("norm", 1) - d.get("shift", 0)) - d.get("offset", 0),
            "standard": lambda x, d: (x - d["mean"]) / d["std"],
            "standard_inv": lambda x, d: x * d["std"] + d["mean"],
        }

        if transform_dict[var]["type"] not in transforms:
            raise NotImplementedError(
                f"Transform {transform_dict[var]['type']} not implemented"
            )

        key = transform_dict[var]["type"]
        if inverse:
            key += "_inv"

        return transforms[key](x, transform_dict[var])

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, index: int):

        shower_energy = self.data["showers"][index].unsqueeze(0)
        incident_energy = self.data["incident_energies"][index]
        # incident_energy = torch.ones_like(shower_energy)*incident_energy

        # shower_features = torch.stack([
        #         shower_energy,
        #         incident_energy,
        #         self.static_features["r"],
        #         self.static_features["phi"],
        #         self.static_features["z"],
        #     ], dim=0,
        # )

        return shower_energy, incident_energy