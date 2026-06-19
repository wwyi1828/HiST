def build_input_projection_linear_config(
    *,
    in_channels,
    out_channels,
    input_projection,
    global_value,
    global_key='global_strategy',
):
    activation = input_projection.get('activation', None)
    conv_config = {
        'layer_type': 'linear',
        'in_channels': in_channels,
        'out_channels': out_channels,
        'activation': activation,
        'dropout': input_projection.get('dropout', 0.0),
        'prenorm': input_projection.get('prenorm', False),
        'bias': input_projection.get('bias', True),
        global_key: global_value,
    }
    if activation is None:
        conv_config['drop_first'] = input_projection.get('drop_first', False)
    else:
        conv_config['mlp_ratio'] = input_projection.get('mlp_ratio', 0.5)
    return conv_config


def build_model_config(cfg, num_genes, in_channels=1024, enc_act='relu'):
    slide_configs = str(cfg.model.slide_configs)
    channel_factor = cfg.model.channel_factor
    trans_type = cfg.model.trans_type
    global_strategy = cfg.model.global_strategy

    ff_ratio = cfg.model.ff_ratio
    input_proj_cfg = cfg.model.input_projection
    input_projection_config = build_input_projection_linear_config(
        in_channels=in_channels,
        out_channels=in_channels // cfg.model.embed_dim_div,
        input_projection=input_proj_cfg,
        global_value=global_strategy,
    )
    drop_path = cfg.model.drop_path
    proj_drop = cfg.model.proj_drop
    attn_drop = cfg.model.attn_drop
    window_size = cfg.model.window_size
    embed_dim_div = cfg.model.embed_dim_div
    embed_dim = in_channels // embed_dim_div
    num_heads = embed_dim // 64
    econvs_type = cfg.model.econvs_type
    pos_std = cfg.model.pos_std
    dilation = cfg.model.dilation
    kernel_size = cfg.model.kernel_size
    stride = cfg.model.stride
    groups = cfg.model.get('groups', 1)
    edge_mode = cfg.model.get('edge_mode', 'none')
    edge_eps = cfg.model.get('edge_eps', 1e-6)
    conv_bias = cfg.model.get('conv_bias', False)
    share_qkv = cfg.model.get('share_qkv', True)
    pos_emb_type = cfg.model.get('pos_emb', 'rpb')
    conv_norm_position = cfg.model.get('conv_norm_position', 'post')

    if '-' in slide_configs:
        encoder_config_string, decoder_config_string = slide_configs.split('-', 1)
    else:
        encoder_config_string = slide_configs
        decoder_config_string = ''
    ds_layers = len(decoder_config_string)

    encoder_config = _build_encoder_config(
        encoder_config_string, in_channels, embed_dim, num_heads,
        channel_factor, trans_type, enc_act,
        input_projection_config,
        attn_drop,
        proj_drop, drop_path, ff_ratio, pos_std,
        econvs_type, window_size, kernel_size, stride, groups,
        dilation, edge_mode, edge_eps, conv_bias, share_qkv, global_strategy, pos_emb_type,
        conv_norm_position
    )

    if decoder_config_string:
        skip_first = cfg.model.skip_first
        dtrans_type = cfg.model.dtrans_type
        dconvs_type = cfg.model.dconvs_type
        connection = cfg.model.connection
        decoder_config = _build_decoder_config(
            decoder_config_string, encoder_config_string, embed_dim,
            num_heads, channel_factor, skip_first, dtrans_type, enc_act,
            attn_drop, proj_drop, drop_path, ff_ratio,
            pos_std, dconvs_type, window_size, kernel_size, stride,
            groups, dilation, edge_mode, edge_eps, conv_bias, connection, num_genes, share_qkv, global_strategy, pos_emb_type,
            conv_norm_position
        )
    else:
        skip_first = cfg.model.get('skip_first', 1)
        decoder_config = []

    token_init_types = cfg.model.get('token_init_types', [1e-4])

    model_components = {
        'ds_layers': ds_layers,
        'dilation': dilation,
        'embed_dim': embed_dim,
        'in_channels': in_channels,
        'encoder_config_string': encoder_config_string,
        'decoder_config_string': decoder_config_string,
        'channel_factor': channel_factor,
        'skip_first': skip_first,
        'token_init_types': token_init_types
    }

    return encoder_config, decoder_config, model_components

def _build_encoder_config(encoder_config_string, in_channels, embed_dim,
                          num_heads, channel_factor, trans_type, enc_act,
                          input_projection_config,
                          attn_drop, proj_drop, drop_path,
                          ff_ratio, pos_std, econvs_type,
                          window_size, kernel_size, stride, groups,
                          dilation, edge_mode, edge_eps, conv_bias, share_qkv, global_strategy, pos_emb_type,
                          conv_norm_position):

    encoder_config = []
    current_embed_dim = embed_dim

    for i, repeat_str in enumerate(encoder_config_string):
        repeat_count = int(repeat_str)

        if i == 0:
            conv_config = dict(input_projection_config)
            trans_embed_dim = current_embed_dim
            trans_num_heads = num_heads
        else:
            conv_config = {
                'layer_type': econvs_type,
                'in_channels': current_embed_dim,
                'out_channels': current_embed_dim * channel_factor,
                'kernel_size': kernel_size,
                'stride': stride,
                'groups': groups,
                'dilation': dilation,
                'edge_mode': edge_mode,
                'edge_eps': edge_eps,
                'bias': conv_bias,
                'global_strategy': global_strategy,
                'pool': 'mean',
                'norm_position': conv_norm_position,
            }
            current_embed_dim = current_embed_dim * channel_factor
            trans_embed_dim = current_embed_dim
            trans_num_heads = num_heads * (channel_factor ** i)

        trans_configs = []
        for _ in range(repeat_count):
            trans_configs.append({
                'layer_type': trans_type,
                'embed_dim': trans_embed_dim,
                'num_heads': trans_num_heads,
                'attn_drop': attn_drop,
                'proj_drop': proj_drop,
                'drop_path': drop_path,
                'activation': enc_act,
                'ff_ratio': ff_ratio,
                'pos_std': pos_std,
                'share_qkv': share_qkv,
                'pos_emb_type': pos_emb_type
            })

        layer_config = [
            ('convs', [conv_config]),
            ('attn_builder', {'window_size': window_size, 'mode': trans_type}),
            ('trans', trans_configs),
        ]
        encoder_config.append(layer_config)

    return encoder_config

def _build_decoder_config(decoder_config_string, encoder_config_string,
                          embed_dim, num_heads, channel_factor, skip_first,
                          dtrans_type, enc_act, attn_drop, proj_drop,
                          drop_path, ff_ratio, pos_std,
                          dconvs_type, window_size, kernel_size, stride,
                          groups, dilation, edge_mode, edge_eps, conv_bias, connection, num_genes, share_qkv, global_strategy, pos_emb_type,
                          conv_norm_position):

    decoder_config = []
    current_in_dim = embed_dim * (channel_factor ** (len(encoder_config_string) - 1))
    current_out_dim = current_in_dim
    skip_mode = "none" if connection is None else str(connection).strip().lower()
    if skip_mode not in ("concat", "add", "none"):
        raise ValueError(f"Unsupported skip connection type: {connection!r}. Use 'concat', 'add', or 'none'.")

    use_skip = skip_mode != "none"
    concat_skip = skip_mode == "concat"

    for i, repeat_str in enumerate(decoder_config_string):
        repeat_count = int(repeat_str)
        current_out_dim = current_out_dim // channel_factor

        conv_config = {
            'layer_type': dconvs_type,
            'in_channels': current_in_dim,
            'out_channels': current_out_dim,
            'kernel_size': kernel_size,
            'stride': stride,
            'groups': groups,
            'dilation': dilation,
            'edge_mode': edge_mode,
            'edge_eps': edge_eps,
            'bias': conv_bias,
            'global_strategy': global_strategy,
            'pool': 'mean',
            'norm_position': conv_norm_position,
        }

        trans_configs = []
        trans_embed_dim = current_out_dim
        trans_num_heads = num_heads * (current_out_dim // embed_dim)

        for _ in range(repeat_count):
            trans_configs.append({
                'layer_type': dtrans_type,
                'embed_dim': trans_embed_dim,
                'num_heads': trans_num_heads,
                'attn_drop': attn_drop,
                'proj_drop': proj_drop,
                'drop_path': drop_path,
                'activation': enc_act,
                'ff_ratio': ff_ratio,
                'pos_std': pos_std,
                'share_qkv': share_qkv,
                'pos_emb_type': pos_emb_type
            })

        skip_index = -(i + 2)

        if skip_first == 1:

            layer_config = [
                ('convs', [conv_config]),
                ('attn_builder', {'window_size': window_size, 'mode': dtrans_type}),
                ('trans', trans_configs),
            ]
            if use_skip:
                layer_config.append(('skipc', {'skip_type': skip_mode, 'skip_index': skip_index}))
            decoder_config.append(layer_config)
            current_in_dim = current_out_dim * 2 if concat_skip else current_out_dim
        else:

            ds_config = {
                'layer_type': 'linear',
                'in_channels': current_out_dim * 2 if concat_skip else current_out_dim,
                'out_channels': current_out_dim,
                'activation': None,
                'dropout': 0,
                'prenorm': False,
                'bias': False,
                'global_strategy': global_strategy
            }

            layer_config = [
                ('convs', [conv_config]),
                ('convs', [ds_config]),
                ('attn_builder', {'window_size': window_size, 'mode': dtrans_type}),
                ('trans', trans_configs),
            ]
            if use_skip:
                layer_config.insert(1, ('skipc', {'skip_type': skip_mode, 'skip_index': skip_index}))
            decoder_config.append(layer_config)
            current_in_dim = current_out_dim

    final_layer = [
        ('convs', [
            {
                'layer_type': 'linear',
                'in_channels': current_in_dim,
                'out_channels': num_genes,
                'activation': None,
                'bias': True,
                'dropout': 0,
                'mlp_ratio': 2,
                'drop_first': True,
                'prenorm': False,
                'global_strategy': global_strategy
            },
        ]),
    ]
    decoder_config.append(final_layer)

    return decoder_config
