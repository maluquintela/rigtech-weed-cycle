"""Versionamento imutável do dataset vivo.

Cada versão vN é uma pasta em ``work/versions/vN/`` contendo:

- ``manifest.json``  — sha256 e tamanho de cada arquivo, mais metadados.
- ``changelog.json`` — diff contra vN-1 (adicionados, removidos, modificados).
- ``dataset.tar.gz`` — os bytes, prontos para ser materializados no Colab.

Versão publicada NUNCA é editada. Correção vira v+1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from src.config import load as load_config

VERSION_RE = re.compile(r"^v(\d+)$")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _next_version(versions_dir: Path) -> str:
    if not versions_dir.is_dir():
        return "v1"
    nums = [int(m.group(1)) for p in versions_dir.iterdir() if (m := VERSION_RE.match(p.name))]
    return f"v{(max(nums) + 1) if nums else 1}"


def _scan(live_dir: Path) -> dict[str, dict]:
    """Índice ``rel_path -> {sha256, size}`` do dataset vivo."""
    entries: dict[str, dict] = {}
    if not live_dir.is_dir():
        return entries
    for p in sorted(live_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(live_dir).as_posix()
        entries[rel] = {"sha256": _sha256(p), "size": p.stat().st_size}
    return entries


def _diff(prev: dict[str, dict], curr: dict[str, dict]) -> dict:
    prev_keys = set(prev)
    curr_keys = set(curr)
    added = sorted(curr_keys - prev_keys)
    removed = sorted(prev_keys - curr_keys)
    modified = sorted(k for k in prev_keys & curr_keys if prev[k]["sha256"] != curr[k]["sha256"])
    return {
        "adicionados": added,
        "removidos": removed,
        "modificados": modified,
        "resumo": {
            "adicionados": len(added),
            "removidos": len(removed),
            "modificados": len(modified),
            "labels_modificados": sum(1 for m in modified if m.startswith("labels/")),
        },
    }


def _write_tarball(live_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(live_dir, arcname=".")


def create(live_dir: Path, versions_dir: Path, note: str) -> Path:
    """Cria uma nova versão imutável do dataset vivo. Retorna a pasta da versão."""
    if not live_dir.is_dir():
        raise FileNotFoundError(f"live_dir não encontrado: {live_dir}")
    versions_dir.mkdir(parents=True, exist_ok=True)

    version = _next_version(versions_dir)
    vdir = versions_dir / version
    vdir.mkdir()

    entries = _scan(live_dir)
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "git_commit": _git_commit(),
        "count_images": sum(1 for k in entries if k.startswith("images/")),
        "count_labels": sum(1 for k in entries if k.startswith("labels/")),
        "files": entries,
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    prev_num = int(version[1:]) - 1
    prev_manifest_path = versions_dir / f"v{prev_num}" / "manifest.json"
    if prev_manifest_path.exists():
        prev = json.loads(prev_manifest_path.read_text(encoding="utf-8"))["files"]
        changelog = {"de": f"v{prev_num}", "para": version, **_diff(prev, entries)}
    else:
        changelog = {"de": None, "para": version, "resumo": {"snapshot_inicial": True}}
    (vdir / "changelog.json").write_text(json.dumps(changelog, indent=2), encoding="utf-8")

    _write_tarball(live_dir, vdir / "dataset.tar.gz")
    return vdir


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Congela uma nova versão imutável do dataset vivo")
    parser.add_argument("--note", required=True, help="nota descritiva do que motivou a versão")
    args = parser.parse_args()

    vdir = create(cfg.paths.live_dir, cfg.paths.versions_dir, args.note)
    manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    changelog = json.loads((vdir / "changelog.json").read_text(encoding="utf-8"))
    print(f"versão criada: {vdir}")
    print(f"  imagens: {manifest['count_images']}")
    print(f"  labels:  {manifest['count_labels']}")
    print(f"  resumo:  {changelog.get('resumo')}")


if __name__ == "__main__":
    main()
