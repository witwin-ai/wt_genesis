"""MDM Tiny with its model, diffusion and SMPL data kept loaded across calls.

``witwin.genesis.mdm_tiny_runtime`` rebuilds the MDM model and CLIP (1.7 s
warm, 3.3 s cold) and re-reads the SMPL model data (0.7 s) on every call,
around about 0.5 s of actual diffusion sampling for a 9.8 s motion. This
backend runs the same bundled runtime, checkpoint and SMPLify fit (every
frame, the requested iterations) but keeps those objects for the life of the
ProjectServer. The seed is applied right before sampling, so one seed always
reproduces one motion whether or not the model was already loaded.

Sampling and fitting follow the vendored ``api._sample_text_motion`` and
``api._fit_smpl_pose``; only model construction and SMPL data loading are
cached.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading
from typing import Any, Sequence

import torch

_LOCK = threading.RLock()
_MODELS: dict[tuple, tuple[Any, Any]] = {}
_SMPL_DATA: dict[str, Any] = {}
_ASSETS_READY = False


def clear_cache() -> None:
    """Drop the loaded model and SMPL data (tests and explicit reloads)."""
    global _ASSETS_READY
    with _LOCK:
        _MODELS.clear()
        _SMPL_DATA.clear()
        _ASSETS_READY = False


def generate_smpl_motion(
    prompts: str | Sequence[str],
    *,
    motion_length: float = 6.0,
    seed: int = 10,
    device: int = 0,
    guidance_param: float = 2.5,
    smplify_iters: int = 20,
    **_unused: Any,
) -> dict[str, torch.Tensor]:
    """Genesis backend contract: same inputs and outputs as MDM Tiny."""
    from witwin.genesis import mdm_tiny_runtime as rt

    with _LOCK:
        _ensure_assets(rt)
        with rt._vendor_import_path():
            api = importlib.import_module("api")
            prompt_list = api._normalize_prompts(prompts)
            options = api.MDMOptions(
                model_path=Path(rt.DEFAULT_CHECKPOINT),
                stats_dir=Path(rt.DEFAULT_STATS_DIR),
                seed=int(seed),
                motion_length=float(motion_length),
                device=int(device),
                guidance_param=float(guidance_param),
                smplify_iters=int(smplify_iters),
            )
            with api.runtime_context():
                args = api._load_checkpoint_args(options, prompt_list)
                data = api.build_minimal_humanml_data(options.stats_dir)
                joints, lengths = _sample_text_motion(args, data)
                fit_device = torch.device(
                    "cpu" if options.device < 0 or not torch.cuda.is_available() else f"cuda:{options.device}"
                )
                smpl = _fit_smpl_pose(api, joints, options.smplify_iters, fit_device)
    return {**smpl, "joints": joints, "lengths": lengths}


def prewarm() -> None:
    """Load the model and run CUDA kernels once so the first request is warm."""
    generate_smpl_motion(["a person stands still"], motion_length=1.0, smplify_iters=1)


def _ensure_assets(rt: Any) -> None:
    global _ASSETS_READY
    if _ASSETS_READY:
        return
    from witwin.genesis.assets import ensure_model_data, validate_smpl_assets

    auto_download = os.environ.get("WITWIN_GENESIS_AUTO_DOWNLOAD", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    ensure_model_data(rt.DATA_DIR, download=auto_download)
    validate_smpl_assets(rt.DATA_DIR)
    _ASSETS_READY = True


def _sample_text_motion(args: Any, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from humanml import collate, recover_from_ric
    from runtime import (
        ClassifierFreeSampleModel,
        create_model_and_diffusion,
        dev,
        fixseed,
        load_saved_model,
        setup_dist,
    )

    fps = 20
    n_frames = min(196, int(args.motion_length * fps))
    data.fixed_length = n_frames
    setup_dist(args.device)
    key = (str(args.model_path), bool(args.guidance_param != 1), bool(args.use_ema), int(args.device))
    cached = _MODELS.get(key)
    if cached is None:
        model, diffusion = create_model_and_diffusion(args, data)
        load_saved_model(model, args.model_path, use_avg=args.use_ema)
        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)
        model.to(dev())
        model.eval()
        cached = _MODELS[key] = (model, diffusion)
    model, diffusion = cached
    fixseed(args.seed)

    collate_args = [
        {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames, "text": prompt}
        for prompt in args.text_prompt
    ]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {
        key: val.to(dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dev()) * args.guidance_param
    model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, 22)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1).contiguous()
    return sample.cpu(), model_kwargs["y"]["lengths"].cpu()


def _fit_smpl_pose(api: Any, motions: torch.Tensor, num_iters: int, device: torch.device) -> dict[str, torch.Tensor]:
    import h5py
    import smplx
    from smplify import SMPLify3D, config

    model_path = Path(config.SMPL_MODEL_DIR) / "smpl" / "SMPL_NEUTRAL.pkl"
    model_data = _SMPL_DATA.get(str(model_path))
    if model_data is None:
        model_data = _SMPL_DATA[str(model_path)] = smplx.utils.Struct(**api.load_smpl_model_data(model_path))

    smpl_pose, smpl_betas, smpl_trans = [], [], []
    confidence = torch.ones(22, device=device)
    for motion in motions:
        joints = motion.permute(2, 0, 1).to(device).float()
        n_frames = joints.shape[0]
        smpl_model = smplx.create(
            config.SMPL_MODEL_DIR,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=n_frames,
            data_struct=model_data,
        ).to(device)
        with h5py.File(config.SMPL_MEAN_FILE, "r") as handle:
            init_pose = torch.from_numpy(handle["pose"][:]).float().to(device).unsqueeze(0).repeat(n_frames, 1)
            init_betas = torch.from_numpy(handle["shape"][:]).float().to(device).unsqueeze(0).repeat(n_frames, 1)
        init_trans = torch.zeros(n_frames, 3, device=device)
        smplify = SMPLify3D(
            smplxmodel=smpl_model,
            batch_size=n_frames,
            joints_category="AMASS",
            num_iters=num_iters,
            device=device,
        )
        _, _, pose, betas, trans, _ = smplify(
            init_pose.detach(),
            init_betas.detach(),
            init_trans.detach(),
            joints,
            conf_3d=confidence,
            seq_ind=0,
        )
        smpl_pose.append(pose.detach().cpu().view(n_frames, 24, 3))
        smpl_betas.append(betas.detach().cpu())
        smpl_trans.append(trans.detach().cpu().view(n_frames, 3))
    return {
        "smpl_pose": torch.stack(smpl_pose, dim=0),
        "smpl_betas": torch.stack(smpl_betas, dim=0),
        "smpl_trans": torch.stack(smpl_trans, dim=0),
    }
