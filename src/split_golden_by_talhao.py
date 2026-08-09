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


def _restore_golden_to_live(live_dir: Path, golden_dir: Path) -> int:
    """Move tudo do golden de volta para live. Torna a operação idempotente:
    chamadas sucessivas sempre produzem o estado correto."""
    moved = 0
    for sub in ("images", "labels"):
        g = golden_dir / sub
        l = live_dir / sub
        if not g.is_dir():
            continue
        l.mkdir(parents=True, exist_ok=True)
        for f in list(g.glob("*")):
            if f.is_file():
                shutil.move(str(f), l / f.name)
                moved += 1
    return moved


def split(live_dir: Path, golden_dir: Path, val_talhoes: list[str]) -> dict[str, int]:
    # idempotente: sempre restaura o estado unificado antes de re-splitar
    restored = _restore_golden_to_live(live_dir, golden_dir)
    if restored:
        print(f"[reset] {restored} arquivos movidos do golden de volta para live")

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

    # merge dos manifests antes de re-splitar (idempotente)
    live_manifest = live_dir / "background_tiles.json"
    gold_manifest = golden_dir / "background_tiles.json"
    all_stems: set[str] = set()
    for m in (live_manifest, gold_manifest):
        if m.exists():
            try:
                all_stems.update(json.loads(m.read_text(encoding="utf-8")).get("stems", []))
            except json.JSONDecodeError:
                pass
    keep, moved = [], []
    for stem in sorted(all_stems):
        (moved if _talhao_of(stem) in val_set else keep).append(stem)
    live_manifest.write_text(json.dumps({"stems": keep}, indent=2), encoding="utf-8")
    gold_manifest.write_text(json.dumps({"stems": moved}, indent=2), encoding="utf-8")

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
