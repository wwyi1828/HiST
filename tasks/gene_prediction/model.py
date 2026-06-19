import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbones import PatchEncoder

class PredictionModel(nn.Module):
    def __init__(
        self,
        padder,
        slide_encoder,
        decoder,
        segmentor,
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
                lora_trainable=True if use_lora else False,
                patch_aug=cfg.training.patch_aug
            )

        self.padder = padder
        self.slide_encoder = slide_encoder
        self.decoder = decoder
        self.segmentor = segmentor

    def forward(self, batch, return_slide_token=False):
        image_features = self.patch_encoder(batch.mor_feats)
        spot_positions = batch.cords if hasattr(batch, 'cords') else None

        spot_positions, [image_features] = self.padder(spot_positions, [image_features])

        ins_pos, image_feats_multi, global_feats, spatial_shape, pos_dict = self.slide_encoder(
            spot_positions, image_features
        )

        if return_slide_token:
            output_pos, decoded_feats, decoder_global_feat = self.decoder(
                ins_pos[-1],
                image_feats_multi,
                global_feats,
                spatial_shape,
                pos_dict,
                return_pre_output_global_feat=True,
            )
        else:
            output_pos, decoded_feats = self.decoder(
                ins_pos[-1], image_feats_multi, global_feats, spatial_shape, pos_dict
            )

        final_prediction = self.segmentor(decoded_feats[-1])[1:-1]
        gene_features = batch.mol_feats
        total_loss = F.mse_loss(final_prediction, torch.log1p(gene_features))

        prediction = torch.expm1(final_prediction)
        prediction = F.relu(prediction)

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
            return prediction, total_loss, slide_token

        return prediction, total_loss
