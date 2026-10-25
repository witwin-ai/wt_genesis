"""MDM Tiny with its model kept loaded across generations.

Measured on a 9.8 s motion, every MDM Tiny call rebuilt the model and CLIP
(1.7 s warm, 3.3 s cold) and re-read the SMPL model data (0.7 s) around 0.5 s
of actual diffusion sampling. The cached backend keeps both loaded and still
fits SMPLify on every frame.
"""
import pytest
import torch

from wt_genesis.adapter import kimodo_backend, mdm_cached_backend


def test_the_cached_backend_is_selected_by_name(monkeypatch):
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "mdm_tiny_cached")
    assert kimodo_backend.configured_backend() is mdm_cached_backend.generate_smpl_motion
    assert kimodo_backend.configured_backend_name() == "mdm_tiny_cached"
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "mdm_tiny")
    assert kimodo_backend.configured_backend() is None
    assert kimodo_backend.configured_backend_name() == "mdm_tiny"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MDM Tiny generation needs CUDA")
def test_the_model_is_built_once_and_a_seed_still_reproduces_the_motion():
    mdm_cached_backend.clear_cache()
    first = mdm_cached_backend.generate_smpl_motion(["a person walks forward"], motion_length=1.0, seed=7,
                                                    smplify_iters=2)
    models = dict(mdm_cached_backend._MODELS)
    second = mdm_cached_backend.generate_smpl_motion(["a person walks forward"], motion_length=1.0, seed=7,
                                                     smplify_iters=2)
    assert len(models) == 1 and mdm_cached_backend._MODELS == models
    assert len(mdm_cached_backend._SMPL_DATA) == 1
    assert first["smpl_pose"].shape == (1, 20, 24, 3)
    assert torch.equal(first["joints"], second["joints"])
    assert torch.allclose(first["smpl_pose"], second["smpl_pose"])
