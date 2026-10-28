"""Kimodo generation with a disk cache of text embeddings.

Run with the isolated Kimodo interpreter (``python kimodo_cached_generate.py
"<prompt>" --model ... --text-cache DIR``); arguments follow
``kimodo.scripts.generate`` for one sample and one NPZ output.

Kimodo's LLM2Vec text encoder (Llama-3-8B in bf16) takes ~18 GB of RAM and
tens of seconds to load for every clip, and a clip was killed loading it next
to other processes. A prompt's embedding is one deterministic vector (the
encoder runs at batch size 1), so it is cached by prompt text and the encoder
is loaded only when a prompt has not been seen. Kimodo's source is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np


class CachedTextEncoder:
    """Kimodo text encoder interface backed by ``<cache>/<sha256>.npy`` files."""

    def __init__(self, cache_dir, *, factory: Callable[[], object], dim: int = 4096):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.factory = factory
        self.dim = int(dim)
        self.encoder = None
        self.device = "cpu"

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        return self

    def eval(self):
        return self

    def _path(self, text: str) -> Path:
        return self.cache_dir / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".npy")

    def _embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros((1, self.dim), dtype=np.float32)
        path = self._path(text)
        if path.is_file():
            return np.load(path)
        if self.encoder is None:
            self.encoder = self.factory()
        feat, lengths = self.encoder([text])
        value = feat[0, : int(lengths[0])].detach().float().cpu().numpy()
        self.dim = int(value.shape[-1])
        partial = path.with_suffix(".tmp.npy")
        np.save(partial, value)
        partial.replace(path)
        return value

    def __call__(self, texts):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        values = [self._embedding(text) for text in texts]
        lengths = [int(value.shape[0]) for value in values]
        padded = np.zeros((len(values), max(lengths), values[0].shape[-1]), dtype=np.float32)
        for index, value in enumerate(values):
            padded[index, : value.shape[0]] = value
        return torch.from_numpy(padded), lengths


def _llm2vec_factory():
    from kimodo.model.load_model import _build_local_text_encoder_conf
    from kimodo.model import instantiate_from_dict

    return instantiate_from_dict(_build_local_text_encoder_conf())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--duration", required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--num_transition_frames", type=int, default=5)
    parser.add_argument("--text-cache", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.scripts.generate import get_texts_and_num_frames_from_prompt
    from kimodo.tools import seed_everything

    args = parse_args(argv)
    if args.num_samples != 1:
        raise SystemExit("the cached runner generates exactly one sample")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = CachedTextEncoder(args.text_cache, factory=_llm2vec_factory)
    model, resolved = load_model(args.model, device=device, default_family="Kimodo",
                                 return_resolved_name=True, text_encoder=encoder)
    texts, num_frames = get_texts_and_num_frames_from_prompt(args.prompt, args.duration, model.fps)
    constraints = load_constraints_lst(args.constraints, model.skeleton) if args.constraints else []
    if args.seed is not None:
        seed_everything(args.seed)
    output = model(
        texts, num_frames, constraint_lst=constraints,
        num_denoising_steps=args.diffusion_steps, num_samples=1, multi_prompt=True,
        num_transition_frames=args.num_transition_frames,
        post_processing="g1" not in resolved, return_numpy=True,
    )
    single = {key: (value[0] if hasattr(value, "shape") and len(value.shape) > 0
                    and value.shape[0] == 1 else value)
              for key, value in output.items()}
    path = args.output if args.output.endswith(".npz") else args.output + ".npz"
    save_kimodo_npz(path, single)
    print(f"Saved {path}; text encoder loaded: {encoder.encoder is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
