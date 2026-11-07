"""Data-only model artifacts: JSON metadata and non-pickled float32 arrays."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .model import Config, check_params
from .tokenizer import Tokenizer


def coverage(params, groups):
    if not isinstance(groups, list) or not groups or len(groups) > 10000:
        raise ValueError("invalid provenance groups")
    covered = {k: torch.zeros(v.shape, dtype=torch.bool) for k, v in params.items()}
    names = set()
    for group in groups:
        if set(group) != {"name", "tensor", "index", "mechanism", "why"}:
            raise ValueError("invalid group fields")
        for key in ("name", "tensor", "mechanism", "why"):
            if not isinstance(group[key], str) or not group[key].strip():
                raise ValueError("group strings must be nonempty")
        if group["name"] in names:
            raise ValueError("duplicate group name")
        names.add(group["name"])
        mask = covered[group["tensor"]]
        index = group["index"]
        if index is None:
            mask[:] = True
        else:
            if not isinstance(index, list):
                raise ValueError("group index must be a list or null")
            axes = []
            for axis in index:
                if isinstance(axis, dict):
                    if set(axis) - {"start", "stop", "step"}:
                        raise ValueError("invalid slice")
                    axes.append(
                        slice(axis.get("start"), axis.get("stop"), axis.get("step"))
                    )
                elif type(axis) is int or (
                    isinstance(axis, list) and all(type(i) is int for i in axis)
                ):
                    axes.append(axis)
                else:
                    raise ValueError("invalid axis")
            mask[tuple(axes)] = True
    nonzero = sum(int((t != 0).sum()) for t in params.values())
    explained = sum(int(((t != 0) & covered[k]).sum()) for k, t in params.items())
    return {
        "params": sum(t.numel() for t in params.values()),
        "nonzero": nonzero,
        "provenance_coverage": explained / nonzero if nonzero else 1.0,
        "groups": len(groups),
    }


def save(directory, cfg, params, groups, words):
    cfg = Config(**cfg)
    tok = Tokenizer(words)
    check_params(cfg, params, tok.vocab_size)
    report = coverage(params, groups)
    if report["provenance_coverage"] != 1.0:
        raise ValueError("unexplained nonzero weights")
    # JSON round-trip rejects non-data metadata before creating the artifact.
    meta = json.loads(
        json.dumps(
            {"format": 1, "config": cfg.to_dict(), "groups": groups, "words": words}
        )
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "weights.npz",
        **{k: t.detach().cpu().numpy() for k, t in params.items()},
    )
    (directory / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    return report


def load(directory):
    directory = Path(directory)
    meta_path, arrays_path = directory / "model.json", directory / "weights.npz"
    if meta_path.stat().st_size > 5_000_000:
        raise ValueError("metadata too large")
    meta = json.loads(meta_path.read_text())
    if set(meta) != {"format", "config", "groups", "words"} or meta["format"] != 1:
        raise ValueError("unsupported artifact schema")
    cfg, tok = Config(**meta["config"]), Tokenizer(meta["words"])
    # Bound decompressed data before numpy opens it (including malicious zip files).
    with zipfile.ZipFile(arrays_path) as archive:
        entries = archive.infolist()
        if len(entries) > 256 or sum(i.file_size for i in entries) > 82_000_000:
            raise ValueError("array archive too large")
        if len({i.filename for i in entries}) != len(entries):
            raise ValueError("duplicate arrays")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        params = {}
        for name in arrays.files:
            a = arrays[name]
            if a.dtype != np.float32:
                raise ValueError("only float32 arrays allowed")
            params[name] = torch.from_numpy(a.copy())
    check_params(cfg, params, tok.vocab_size)
    report = coverage(params, meta["groups"])
    if report["provenance_coverage"] != 1.0:
        raise ValueError("incomplete provenance")
    return cfg, params, tok, report


def hashes(directory):
    root = Path(directory)
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("model.json", "weights.npz")
    }
