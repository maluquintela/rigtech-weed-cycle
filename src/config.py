"""Carregamento do config.yaml e resolução de caminhos relativos.

Uma única função pública: ``load()``. Todo caminho do YAML é resolvido contra a
raiz do repositório, o que permite invocar ``python -m src.qa_static`` de
qualquer diretório.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _ns(obj: Any) -> Any:
    """Converte dicts em SimpleNamespace recursivamente.

    Dicts com chave não-string (ex.: o dicionário de classes ``{0: ..., 1: ...}``)
    permanecem como dict — SimpleNamespace só aceita chaves string.
    """
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            return obj
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def _resolve_paths(paths_ns: SimpleNamespace) -> SimpleNamespace:
    """Converte cada caminho em Path absoluto sob REPO_ROOT."""
    resolved: dict[str, Path] = {}
    for key, value in vars(paths_ns).items():
        p = Path(value)
        resolved[key] = p if p.is_absolute() else REPO_ROOT / p
    return SimpleNamespace(**resolved)


def load(path: str | os.PathLike | None = None) -> SimpleNamespace:
    """Lê o YAML, converte em namespace e resolve caminhos.

    Expõe também ``cfg.class_names`` (lista ordenada por class_id) e ``cfg.nc``
    (número de classes), derivados de ``project.classes``.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = _ns(raw)
    cfg.paths = _resolve_paths(cfg.paths)
    cfg.repo_root = REPO_ROOT

    classes_dict: dict[int, str] = raw["project"]["classes"]
    ordered = sorted(classes_dict.items(), key=lambda kv: int(kv[0]))
    cfg.class_names = [name for _, name in ordered]
    cfg.nc = len(ordered)

    return cfg


if __name__ == "__main__":
    cfg = load()
    print(f"repo_root:   {cfg.repo_root}")
    print(f"classes:     {cfg.class_names}")
    print(f"nc:          {cfg.nc}")
    print(f"live_dir:    {cfg.paths.live_dir}")
    print(f"golden_dir:  {cfg.paths.golden_dir}")
    print(f"model.imgsz: {cfg.model.imgsz}")
    print(f"model.seed:  {cfg.model.seed}")
