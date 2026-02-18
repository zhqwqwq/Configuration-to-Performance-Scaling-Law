import torch


def _infer_checkpoint_dtype(state_dict):
    dtypes = {v.dtype for v in state_dict.values() if torch.is_tensor(v)}
    if len(dtypes) == 1:
        return next(iter(dtypes))
    if len(dtypes) > 1:
        return torch.float32
    return None


def get_model(cfg, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import ScalingLawForecaster

    training_cfg = cfg.get('training', {}) if hasattr(cfg, 'get') else {}
    model_cfg = cfg.get('model', {}) if hasattr(cfg, 'get') else {}
    force_fp32 = model_cfg.get('force_fp32', None)
    state_dict = None
    inferred_dtype = None

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        inferred_dtype = _infer_checkpoint_dtype(state_dict)
        if force_fp32 is None and inferred_dtype is not None:
            force_fp32 = inferred_dtype == torch.float32
    if force_fp32 is None:
        force_fp32 = not training_cfg.get('amp', False)

    model = ScalingLawForecaster(
        base_model_name=cfg['model']['base_model'],
        init_from_pretrained=cfg['model'].get('init_from_pretrained', True),
        force_fp32=force_fp32,
    )
    if inferred_dtype is not None and not force_fp32:
        model = model.to(dtype=inferred_dtype)
    print(model.config)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    return model
