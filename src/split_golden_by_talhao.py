"""Move todos os tiles do(s) talhão(ões) de validação para o golden set.

O golden set aqui NÃO é o "duplo passe" ideal do manual, mas é o mais próximo
que conseguimos com o dataset atual: hold-out completo por talhão. Ao segurar
um talhão inteiro fora do train, medimos generalização entre campos (voos,
iluminação, solo), que é o cenário de produção — não vazamento espacial dentro
do mesmo ortomosaico.

O talhão de val é a "régua". Uma vez movido, não volta ao live sem versionar.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from src.config import load as load_config


def _talhao_of(stem: str) -> str:
    return re.split(r"_r\d+_c\d+$", stem)[0]


def split(live_dir: Path, golden_dir: Path, val_talhoes: list[str]) -> dict[str, int]:
    images_src = live_dir / "images"
    labels_src = live_dir / "labels"
    images_dst = golden_dir / "images"
    labels_dst = golden_dir / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    val_set = set(val_talhoes)
    counts: dict[str, int] = {}
    for img in list(images_src.glob("*.jpg")):
        stem = img.stem
        talhao = _talhao_of(stem)
        counts[talhao] = counts.get(talhao, 0) + 1
        if talhao not in val_set:
            continue
        for ext, src_dir, dst_dir in [
            (".jpg", images_src, images_dst),
            (".txt", labels_src, labels_dst),
        ]:
            src = src_dir / f"{stem}{ext}"
            if src.exists():
                shutil.move(str(src), dst_dir / f"{stem}{ext}")

    # atualiza manifest de background
    manifest = live_dir / "background_tiles.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        keep, moved = [], []
        for stem in data.get("stems", []):
            (moved if _talhao_of(stem) in val_set else keep).append(stem)
        manifest.write_text(json.dumps({"stems": sorted(keep)}, indent=2), encoding="utf-8")
        (golden_dir / "background_tiles.json").write_text(
            json.dumps({"stems": sorted(moved)}, indent=2), encoding="utf-8"
        )

    return counts


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Move talhão(ões) de val para golden set")
    parser.add_argument("--val-talhoes", nargs="+", default=list(cfg.split.val_talhoes))
    parser.add_argument("--live", type=Path, default=cfg.paths.live_dir)
    parser.add_argument("--golden", type=Path, default=cfg.paths.golden_dir)
    args = parser.parse_args()

    counts = split(args.live, args.golden, args.val_talhoes)
    val_set = set(args.val_talhoes)
    total = sum(counts.values())
    val = sum(n for t, n in counts.items() if t in val_set)
    print("Distribuição por talhão:")
    for t in sorted(counts):
        marker = " ← VAL" if t in val_set else ""
        print(f"  {t:<14} {counts[t]:>5} tiles{marker}")
    print(f"\ntotal={total}  train={total - val}  val={val}  ({val/total*100:.0f}%)")
    print(f"train -> {args.live}")
    print(f"val   -> {args.golden}")


if __name__ == "__main__":
    main()
