import torch
import math

class SPAN_Padder:

    def __init__(self, kernel_size, stride, dilation, n_layers, pad_feats=None):

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.n_layers = n_layers
        self.pad_feats = pad_feats

        K_eff = (kernel_size - 1) * dilation + 1
        S_n = stride ** n_layers
        if stride == 1:
            RF_n = 1 + n_layers * (K_eff - 1)
        else:
            RF_n = 1 + (K_eff - 1) * (S_n - 1) // (stride - 1)
        self.S_n = S_n
        self.RF_n = RF_n

    def pad_sparse_points(self, pos, feats):

        is_list = isinstance(feats, list)
        feats_list = feats if is_list else [feats]

        curr_max = pos.max(dim=0).values
        curr_w = curr_max[0] + 1
        curr_h = curr_max[1] + 1

        pad_left, pad_right = self.compute_padding(curr_w)
        pad_top, pad_bottom = self.compute_padding(curr_h)

        new_pos_list = []
        if pad_left > 0 or pad_top > 0:
            pos1 = torch.tensor(
                [[-pad_left, -pad_top]],
                dtype=pos.dtype,
                device=pos.device
            )
            new_pos_list.append(pos1)

        new_pos_list.append(pos)

        if pad_right > 0 or pad_bottom > 0:
            pos2 = torch.tensor(
                [[curr_w - 1 + pad_right, curr_h - 1 + pad_bottom]],
                dtype=pos.dtype,
                device=pos.device
            )
            new_pos_list.append(pos2)

        new_pos = torch.cat(new_pos_list, dim=0)

        all_new_feats = []
        for feat in feats_list:
            new_feats_list = []

            if pad_left > 0 or pad_top > 0:
                pad_feat = self._get_pad_feat(feat)
                new_feats_list.append(pad_feat)

            new_feats_list.append(feat)

            if pad_right > 0 or pad_bottom > 0:
                pad_feat = self._get_pad_feat(feat)
                new_feats_list.append(pad_feat)

            new_feat = torch.cat(new_feats_list, dim=0)
            all_new_feats.append(new_feat)

        if pad_left > 0 or pad_top > 0:
            shift = torch.tensor([pad_left, pad_top], dtype=pos.dtype, device=pos.device)
            new_pos = new_pos + shift

        return new_pos, all_new_feats[0] if not is_list else all_new_feats

    def __call__(self, pos, feats):

        return self.pad_sparse_points(pos, feats)

    def compute_padding(self, I_orig):

        k = math.ceil((I_orig - self.RF_n) / self.S_n)
        if k < 0:
            k = 0
        I_adj = (self.RF_n + self.S_n * k) + self.RF_n
        total_padding = I_adj - I_orig
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        return pad_left, pad_right

    def _get_pad_feat(self, feats):

        if self.pad_feats is None:
            return torch.zeros((1, feats.size(1)), dtype=feats.dtype, device=feats.device)
        else:
            return self.pad_feats.view(1, -1).to(feats.dtype).to(feats.device)
