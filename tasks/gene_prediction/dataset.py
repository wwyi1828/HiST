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


def _read_h5(path):
    values = {}
    with h5py.File(path, 'r') as file:
        for key, dataset in file.items():
            if h5py.check_string_dtype(dataset.dtype) is not None:
                data = dataset.asstr()[()]
                values[key] = data.tolist() if isinstance(data, np.ndarray) else data
            else:
                data = dataset[()]
                try:
                    values[key] = torch.as_tensor(data)
                except TypeError:
                    values[key] = data
    return values

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

        src_folder, slide_name, gene_type, num_genes, gene_dir_value = self.slide_items[idx]
        gene_dir = Path(gene_dir_value)
        if not gene_dir.is_absolute():
            gene_dir = Path(src_folder) / gene_dir
        sample.slide_name = slide_name

        src_path = Path(src_folder)
        gene_file = gene_dir.joinpath(f"{slide_name}.h5")
        imge_file = src_path.joinpath(f'imge_{self.image_suffix}', f"{slide_name}.h5")

        vars(sample).update(_read_h5(imge_file))
        vars(sample).update(_read_h5(gene_file))

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

        indices = None
        if gene_type in ('hvg', 'heg'):
            ranking = getattr(sample, f'{gene_type}_indices')
            indices = ranking[:num_genes] if num_genes else ranking
            sample.mol_feats = sample.mol_feats[:, indices]
        if hasattr(sample, 'union_gene_names'):
            sample.gene_names = (sample.union_gene_names if indices is None else
                                 [sample.union_gene_names[i] for i in indices.tolist()])

        return sample
