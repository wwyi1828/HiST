import os
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from .config import get_backbone_config
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class PatchEncoder(nn.Module):

    def __init__(
        self, model_name, trainable,
        patch_weights, use_lora,
        lora_rank, lora_alpha, lora_trainable=True, patch_aug=False
    ):

        super().__init__()

        alias_map = {
            'UNI': 'vit_large_patch16_224',
            'V2': 'vit_large_patch16_224',
        }
        normalized_name = alias_map.get(model_name, model_name)

        backbone_config = get_backbone_config(normalized_name)
        self.model_name = normalized_name
        self.use_lora = use_lora
        self.patch_aug = patch_aug

        if normalized_name == 'vit_large_patch16_224':
            self.model = timm.create_model(
                normalized_name,
                img_size=backbone_config['img_size'],
                patch_size=backbone_config['patch_size'],
                init_values=backbone_config['init_values'],
                num_classes=0,
                dynamic_img_size=backbone_config['dynamic_img_size']
            )
            if patch_weights is None:
                patch_weights = os.environ.get("UNI_WEIGHTS")
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        if patch_weights is not None:
            print(f"Loading pretrained weights from {patch_weights}")
            state_dict = torch.load(patch_weights, map_location="cpu")
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("Successfully loaded weights for ViT")
            except Exception as e:
                print(f"Warning: Failed to load some pretrained weights: {e}")

        for p in self.model.parameters():
            p.requires_grad = trainable

        if use_lora and normalized_name == 'vit_large_patch16_224':
            if LoRALayer is None:
                print("Warning: LoRA requested but LoRA module not available. Skipping LoRA.")
                use_lora = False
            else:
                self.lora_layers = {}

                for name, module in self.model.named_modules():
                    if 'attn.qkv' in name:
                        hidden_size = module.weight.shape[1]
                        out_features = module.weight.shape[0]
                        query_key_dim = (out_features * 2) // 3

                        lora_name = name.replace('.', '_')
                        self.lora_layers[lora_name] = LoRALayer(
                            hidden_size, query_key_dim,
                            rank=lora_rank, alpha=lora_alpha
                        )

                for name, layer in self.lora_layers.items():
                    self.add_module(f"lora_{name}", layer)

                num_lora_params = 0
                for name, param in self.named_parameters():
                    if 'lora_' in name:
                        param.requires_grad = lora_trainable
                        num_lora_params += 1

                print(f"Added LoRA (rank={lora_rank}, alpha={lora_alpha}) to {len(self.lora_layers)} attention layers")
                print(f"LoRA parameters: {num_lora_params} ({'trainable' if lora_trainable else 'frozen'})")

                self._store_orig_forward_methods()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.output_dim = backbone_config['embedding_dim']

    def _store_orig_forward_methods(self):

        if not self.use_lora or self.model_name != 'vit_large_patch16_224':
            return

        for name, module in self.model.named_modules():
            if 'attn.qkv' in name:
                module.__dict__.setdefault('_original_forward', module.forward)

                lora_name = name.replace('.', '_')
                lora_layer = getattr(self, f"lora_{lora_name}")

                def make_forward(orig_module, lora):
                    def forward(x):
                        orig_output = orig_module._original_forward(x)
                        batch_size, seq_len, hidden_dim = orig_output.shape
                        qkv_dim = hidden_dim // 3

                        lora_output = lora(x)

                        combined = orig_output.clone()
                        combined[:, :, :2*qkv_dim] += lora_output

                        return combined

                    return forward

                module.forward = make_forward(module, lora_layer)

    def forward(self, x):

        if len(x.shape) == 5:
            B, N, H, W, C = x.shape
            x = x.reshape(B*N, H, W, C)

        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = x / 255.0

        if self.training and self.patch_aug:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-2])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
            B, C, H, W = x.shape
            if torch.rand(()) < 0.3:
                ks = int(2 * torch.randint(1, 3, (1,)).item() + 1)
                sigma = 0.1 + (torch.rand(()).item() * 0.9)
                x = TF.gaussian_blur(x, kernel_size=[ks, ks], sigma=[sigma, sigma])
            if torch.rand(()) < 0.15:
                ksz = int(2 * torch.randint(1, 3, (1,)).item() + 1)
                ang = int(torch.randint(0, 4, (1,)).item())
                ker = torch.zeros((ksz, ksz), device=x.device, dtype=x.dtype)
                if ang == 0:
                    ker[ksz // 2, :] = 1
                elif ang == 1:
                    ker[:, ksz // 2] = 1
                elif ang == 2:
                    for i in range(ksz):
                        ker[i, i] = 1
                else:
                    for i in range(ksz):
                        ker[i, ksz - 1 - i] = 1
                ker = ker / ker.sum()
                w = ker.view(1, 1, ksz, ksz).repeat(C, 1, 1, 1)
                x = F.conv2d(x, w, padding=ksz // 2, groups=C)
            if torch.rand(()) < 0.3:
                g = 0.1
                gh, gw = 16, 16
                mask = torch.rand((1, 1, gh, gw), device=x.device, dtype=x.dtype) * (2 * g) + (1 - g)
                mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                x = x * mask
            if torch.rand(()) < 0.5:
                b = 1.0 + (torch.rand(()).item() * 0.2 - 0.1)
                c = 1.0 + (torch.rand(()).item() * 0.2 - 0.1)
                s = 1.0 + (torch.rand(()).item() * 0.1 - 0.05)
                x = TF.adjust_brightness(x, b)
                x = TF.adjust_contrast(x, c)
                x = TF.adjust_saturation(x, s)
            if torch.rand(()) < 0.5:
                gm = 0.9 + (torch.rand(()).item() * 0.2)
                x = TF.adjust_gamma(x, gm)
                bg = 1.0 + (torch.rand(()).item() * 0.2 - 0.1)
                x = TF.adjust_brightness(x, bg)
            if torch.rand(()) < 0.3:
                sn = 0.005 + (torch.rand(()).item() * 0.015)
                x = x + torch.randn_like(x) * sn
            if torch.rand(()) < 0.2:
                B = x.size(0)
                K = 2
                eh = torch.randint(8, min(25, H + 1), (B, K), device=x.device)
                ew = torch.randint(8, min(25, W + 1), (B, K), device=x.device)
                y0 = torch.randint(0, H, (B, K), device=x.device)
                x0 = torch.randint(0, W, (B, K), device=x.device)
                y0 = torch.minimum(y0, (H - eh))
                x0 = torch.minimum(x0, (W - ew))
                use_rect = (torch.rand(B, K, device=x.device) < 0.5)
                none_mask = ~use_rect.any(dim=1)
                if none_mask.any():
                    use_rect[none_mask, 0] = True
                gy = torch.arange(H, device=x.device).view(1, 1, H, 1)
                gx = torch.arange(W, device=x.device).view(1, 1, 1, W)
                y0e = y0.unsqueeze(-1).unsqueeze(-1)
                x0e = x0.unsqueeze(-1).unsqueeze(-1)
                ehe = eh.unsqueeze(-1).unsqueeze(-1)
                ewe = ew.unsqueeze(-1).unsqueeze(-1)
                mk = (gy >= y0e) & (gy < (y0e + ehe)) & (gx >= x0e) & (gx < (x0e + ewe))
                mk = mk & use_rect.unsqueeze(-1).unsqueeze(-1)
                mask = mk.any(dim=1, keepdim=True)
                x = x.masked_fill(mask, 0.0)
            x = x.clamp_(0.0, 1.0)

        x = (x - self.mean) / self.std

        features = self.model(x)

        if isinstance(features, torch.Tensor) and features.dim() > 2:
            features = features.reshape(features.size(0), -1)

        return features
