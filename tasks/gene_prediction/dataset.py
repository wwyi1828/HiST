import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from types import SimpleNamespace
from src.span.preprocessing import get_sorted_indices


def _set_transform_multiplier(transform, multiplier) -> None:
    for t in getattr(transform, 'transforms', ()):
        if hasattr(t, 'set_multiplier'):
            t.set_multiplier(multiplier)


def custom_collate_fn(batch):

    if len(batch) == 1:
        return batch[0]

    batch_collated = SimpleNamespace()

    keys = batch[0].__dict__.keys()
    for key in keys:
        values = [getattr(item, key) for item in batch]

        if isinstance(values[0], torch.Tensor):
            try:
                setattr(batch_collated, key, torch.stack(values))
            except:
                setattr(batch_collated, key, values)
        else:
            setattr(batch_collated, key, values)
    return batch_collated

class HiSTDataset(Dataset):

    def __init__(self, slide_items, transform=None, image_suffix='RAW'):
        self.slide_items = slide_items
        self.transform = transform
        self.image_suffix = image_suffix

    def __len__(self):
        return len(self.slide_items)

    def set_aug_multiplier(self, multiplier):
        _set_transform_multiplier(self.transform, multiplier)

    def __getitem__(self, idx):
        sample = SimpleNamespace()

        item = self.slide_items[idx]
        if len(item) == 4:
            src_folder, slide_name, gene_type, num_genes = item
            gene_dir = Path(src_folder) / "gene"
        elif len(item) == 5:
            src_folder, slide_name, gene_type, num_genes, gene_dir_value = item
            gene_dir = Path(gene_dir_value)
            if not gene_dir.is_absolute():
                gene_dir = Path(src_folder) / gene_dir_value
        else:
            raise ValueError(f"Unexpected slide item format (len={len(item)}): {item}")
        sample.slide_name = slide_name

        src_path = Path(src_folder)
        gene_file = gene_dir.joinpath(f"{slide_name}.h5")
        imge_file = src_path.joinpath(f'imge_{self.image_suffix}', f"{slide_name}.h5")

        with h5py.File(imge_file, 'r') as file:
            for key in file.keys():
                attr_name = f"{key}"
                if isinstance(file[key][()], (np.ndarray, list)) and file[key].dtype.kind == 'S':
                    data = [s.decode('utf-8') if isinstance(s, bytes) else s for s in file[key][:]]
                    setattr(sample, attr_name, data)
                else:
                    try:
                        data = torch.tensor(file[key][:])
                        setattr(sample, attr_name, data)
                    except TypeError:
                        data = file[key][:]
                        setattr(sample, attr_name, data)

        with h5py.File(gene_file, 'r') as file:
            for key in file.keys():
                attr_name = f"{key}"
                if isinstance(file[key][()], (np.ndarray, list)) and file[key].dtype.kind == 'S':
                    data = [s.decode('utf-8') if isinstance(s, bytes) else s for s in file[key][:]]
                    setattr(sample, attr_name, data)
                else:
                    try:
                        data = torch.tensor(file[key][:])
                        setattr(sample, attr_name, data)
                    except TypeError:
                        data = file[key][:]
                        setattr(sample, attr_name, data)

        if self.transform and hasattr(sample, 'float_cords'):
            pos = sample.float_cords.float()
            out = self.transform(pos)
            if isinstance(out, tuple):
                pos, mask = out
            else:
                pos, mask = out, None
            sample.final_cords = pos
            
            if mask is not None:
                attributes_to_filter = ['mol_feats', 'mor_feats', 'orig_cords', 'total_umi_counts', 'feats']
                for attr_name in attributes_to_filter:
                    if hasattr(sample, attr_name):
                        current_value = getattr(sample, attr_name)
                        if len(current_value) == len(mask):
                            setattr(sample, attr_name, current_value[mask])
        else:
            sample.final_cords = sample.cords

        if hasattr(sample, 'final_cords'):
            if hasattr(sample, 'cords'):
                delattr(sample, 'cords')
            if hasattr(sample, 'float_cords'):
                delattr(sample, 'float_cords')

        sorted_indices = get_sorted_indices(sample.final_cords)
        sample.final_cords = sample.final_cords[sorted_indices]

        attributes_to_sort = ['mol_feats', 'mor_feats', 'orig_cords', 'total_umi_counts']
        for attr_name in attributes_to_sort:
            if hasattr(sample, attr_name):
                current_value = getattr(sample, attr_name)
                setattr(sample, attr_name, current_value[sorted_indices])

        if hasattr(sample, 'mol_feats'):
            if gene_type == 'hvg' and hasattr(sample, 'hvg_indices'):
                indices = sample.hvg_indices[:num_genes] if num_genes else sample.hvg_indices
                sample.mol_feats = sample.mol_feats[:, indices]
            elif gene_type == 'heg' and hasattr(sample, 'heg_indices'):
                indices = sample.heg_indices[:num_genes] if num_genes else sample.heg_indices
                sample.mol_feats = sample.mol_feats[:, indices]

        return sample
