def get_backbone_config(model_name):

    configs = {
        "vit_large_patch16_224": {
            "embedding_dim": 1024,
            "img_size": 224,
            "dynamic_img_size": True,
            "patch_size": 16,
            "init_values": 1e-5
        },
    }
    return configs.get(model_name, configs["vit_large_patch16_224"])
