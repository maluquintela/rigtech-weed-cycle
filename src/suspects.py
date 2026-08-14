"""Detecta tiles suspeitas rodando um modelo treinado contra o golden set.

Diferença de xval_flag.py: aquele TREINA N modelos out-of-fold (custoso).
Este apenas CARREGA um best.pt já treinado e roda inference no golden.
Uso típico: pós-treino, para gerar cards no Linear que anotador vai revisar.

Uma tile é marcada como suspeita se qualquer critério for atingido:
- mean_iou < cfg.xval.thresholds.mean_iou_below
- gt_perdidos_ratio > cfg.xval.thresholds.missed_gt_ratio_above
- ghost_conf > cfg.xval.thresholds.ghost_pred_conf_above

Reutiliza a função de scoring de xval_flag (mesma lógica, mesmo score).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.config import load as load_config
from src.xval_flag import ImageSuspicion, _load_gt_polys, score_image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _predict_polys(
    weights: Path,
    images: Iterable[Path],
    imgsz: int,
    conf: float = 0.25,
    device: str | None = None,
) -> dict[str, list[tuple[int, list[float], float]]]:
    """Retorna ``{nome_imagem: [(class_id, coords_xyn, conf), ...]}``."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions: dict[str, list[tuple[int, list[float], float]]] = {}
    for img in images:
        kwargs = {"source": str(img), "imgsz": imgsz, "conf": conf, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        polys: list[tuple[int, list[float], float]] = []
        for r in results:
            if r.masks is None:
                continue
            for i, seg in enumerate(r.masks.xyn):
                cls = int(r.boxes.cls[i].item())
                pconf = float(r.boxes.conf[i].item())
                coords = seg.reshape(-1).tolist()
                polys.append((cls, coords, pconf))
        predictions[img.name] = polys
    return predictions


def find_suspects(
    weights: Path,
    golden_dir: Path,
    cfg,
    conf: float = 0.25,
    device: str | None = None,
) -> list[ImageSuspicion]:
    """Roda inference com ``weights`` no golden e retorna tiles suspeitas
    ordenadas por score decrescente."""
    images_dir = golden_dir / "images"
    labels_dir = golden_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"golden/images não encontrado: {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"nenhuma imagem em {images_dir}")

    preds = _predict_polys(
        weights=weights, images=images, imgsz=cfg.model.imgsz,
        conf=conf, device=device,
    )

    thr = cfg.xval.thresholds
    suspicions: list[ImageSuspicion] = []
    for img in images:
        gt = _load_gt_polys(labels_dir / f"{img.stem}.txt")
        pred = preds.get(img.name, [])
        s = score_image(gt, pred)
        s.image = img.name
        if (
            s.mean_iou < thr.mean_iou_below
            or s.gt_perdidos_ratio > thr.missed_gt_ratio_above
            or s.ghost_conf > thr.ghost_pred_conf_above
        ):
            suspicions.append(s)
    suspicions.sort(key=lambda x: x.score, reverse=True)
    return suspicions


def write_report(suspicions: list[ImageSuspicion], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "score", "mean_iou", "gt_perdidos_ratio", "ghost_conf", "n_gt", "n_pred"])
        for s in suspicions:
            writer.writerow([
                s.image, f"{s.score:.4f}", f"{s.mean_iou:.4f}",
                f"{s.gt_perdidos_ratio:.4f}", f"{s.ghost_conf:.4f}", s.n_gt, s.n_pred,
            ])


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Detecta tiles suspeitas usando um best.pt já treinado")
    parser.add_argument("--weights", type=Path, required=True, help="caminho para best.pt")
    parser.add_argument("--golden", type=Path, default=cfg.paths.golden_dir, help="dir com images/ e labels/")
    parser.add_argument("--out", type=Path, default=cfg.paths.suspects_report)
    parser.add_argument("--conf", type=float, default=0.25, help="conf mínima para considerar predição")
    parser.add_argument("--device", default=None, help="device Ultralytics (cpu | mps | 0)")
    parser.add_argument("--top-n", type=int, default=None, help="limita saída às N piores")
    args = parser.parse_args()

    suspicions = find_suspects(
        weights=args.weights, golden_dir=args.golden, cfg=cfg,
        conf=args.conf, device=args.device,
    )
    if args.top_n:
        suspicions = suspicions[: args.top_n]

    write_report(suspicions, args.out)
    print(f"{len(suspicions)} tile(s) suspeita(s) -> {args.out}")
    for s in suspicions[:10]:
        print(f"  {s.image:40} score={s.score:.3f} iou={s.mean_iou:.2f} miss={s.gt_perdidos_ratio:.2f} ghost={s.ghost_conf:.2f}")


if __name__ == "__main__":
    main()
