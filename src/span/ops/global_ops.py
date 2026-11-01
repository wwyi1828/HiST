import torch


def _pair(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def _pool(out, pool):
    if pool == 'sum':
        return out.sum(1)
    if pool == 'mean':
        return out.mean(1)
    if pool == 'max':
        return out.amax(1)
    raise ValueError('global pooling must be sum/mean/max')


def global_convolution(feat, conv, pool='mean'):
    kh, kw = _pair(conv.kernel_size)
    dh, dw = _pair(conv.dilation)
    sh, sw = _pair(conv.stride)
    ph, pw = _pair(conv.padding)
    ih, iw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    oh, ow = (ih + 2 * ph - (kh - 1) * dh - 1) // sh + 1, (iw + 2 * pw - (kw - 1) * dw - 1) // sw + 1
    y = torch.arange(oh, device=feat.device)[:, None] * sh - ph + torch.arange(kh, device=feat.device)[None, :] * dh
    x = torch.arange(ow, device=feat.device)[:, None] * sw - pw + torch.arange(kw, device=feat.device)[None, :] * dw
    mask = (((y >= 0) & (y < ih))[:, None, :, None]
            & ((x >= 0) & (x < iw))[None, :, None, :]).reshape(oh * ow, kh, kw)
    weights = torch.einsum('pkl,oikl->poi', mask.to(conv.weight.dtype), conv.weight)
    groups = conv.groups
    weights = weights.reshape(oh * ow, groups, conv.out_channels // groups, conv.in_channels // groups)
    inputs = feat.reshape(len(feat), groups, conv.in_channels // groups)
    out = torch.einsum('bgi,pgoi->bpgo', inputs, weights).reshape(len(feat), oh * ow, conv.out_channels)
    if conv.bias is not None:
        out = out + conv.bias.to(out.dtype)
    return _pool(out, pool)


def global_transpose(feat, conv, pool='mean'):
    kh, kw = _pair(conv.kernel_size)
    dh, dw = _pair(conv.dilation)
    ph, pw = _pair(conv.padding)
    oph, opw = _pair(conv.output_padding)
    oh, ow = (kh - 1) * dh - 2 * ph + oph + 1, (kw - 1) * dw - 2 * pw + opw + 1
    groups = conv.groups
    weights = conv.weight.reshape(groups, conv.in_channels // groups,
                                   conv.out_channels // groups, kh * kw)
    inputs = feat.reshape(len(feat), groups, conv.in_channels // groups)
    values = torch.einsum('bgi,giok->bkgo', inputs, weights).reshape(len(feat), kh * kw, conv.out_channels)
    y = torch.arange(oh, device=feat.device)[:, None] + ph
    x = torch.arange(ow, device=feat.device)[None, :] + pw
    valid = (y % dh == 0) & (x % dw == 0) & (y // dh < kh) & (x // dw < kw)
    ids = (y // dh * kw + x // dw).clamp(0, kh * kw - 1).reshape(-1)
    out = values[:, ids].masked_fill(~valid.reshape(1, -1, 1), 0)
    if conv.bias is not None:
        out = out + conv.bias.to(out.dtype)
    return _pool(out, pool)
