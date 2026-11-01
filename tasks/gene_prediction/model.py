import torch
import torch.nn as nn
from lib.backbones import PatchEncoder

class PredictionModel(nn.Module):
    def __init__(
        self,
        padder,
        slide_encoder,
        decoder,
        image_suffix='RAW',
        cfg=None,
    ):
        super().__init__()

        if image_suffix != 'RAW':
            self.patch_encoder = nn.Identity()
        else:
            if cfg is None:
                raise ValueError("cfg required when image_suffix='RAW'")
            trainable = cfg.model.train_mode == 'full'
            use_lora = cfg.model.train_mode == 'lora'

            self.patch_encoder = PatchEncoder(
                model_name=cfg.patch_backbone,
                trainable=trainable,
                patch_weights=None,
                use_lora=use_lora,
                lora_rank=cfg.model.lora_rank,
                lora_alpha=cfg.model.lora_alpha,
                lora_trainable=use_lora,
                patch_aug=cfg.training.patch_aug
            )

        self.padder = padder
        self.slide_encoder = slide_encoder
        self.decoder = decoder

    def forward(self, spot_positions, image_features, return_slide_token=False):
        image_features = self.patch_encoder(image_features)
        spot_positions, image_features, sentinel_mask = self.padder(spot_positions, image_features)

        ins_pos, image_feats_multi, global_feats, spatial_shape, pos_dict = self.slide_encoder(
            spot_positions, image_features, sentinel_mask=sentinel_mask
        )

        if return_slide_token:
            decoded_feats, decoder_global_feat = self.decoder(
                ins_pos[-1],
                image_feats_multi,
                global_feats,
                spatial_shape,
                pos_dict,
                return_pre_output_global_feat=True,
            )
        else:
            decoded_feats = self.decoder(
                ins_pos[-1], image_feats_multi, global_feats, spatial_shape, pos_dict
            )

        log_prediction = decoded_feats[~pos_dict.sentinel_mask]

        if return_slide_token:
            slide_token_parts = []
            if len(global_feats) > 0:
                slide_token_parts.append(global_feats[-1])
            if decoder_global_feat is not None:
                slide_token_parts.append(decoder_global_feat)
            if slide_token_parts:
                hidden_dims = sorted({token.shape[-1] for token in slide_token_parts})
                if len(hidden_dims) != 1:
                    raise RuntimeError(
                        f"Expected encoder and decoder pre-output slide tokens to share one hidden dim, got {hidden_dims}"
                    )
                slide_token = torch.cat(slide_token_parts, dim=0)
            else:
                slide_token = None
            return log_prediction, slide_token

        return log_prediction
