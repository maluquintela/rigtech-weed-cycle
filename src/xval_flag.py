"""Detecção de anotações suspeitas por validação cruzada out-of-fold.

Divide o dataset em k partições agrupadas por ``group_key`` (para não vazar
frames sequenciais entre folds), treina um modelo por fold e prevê nas imagens
que ele NÃO viu. Combina três sinais em um score de suspeita:

- mean_iou             : máscara existe dos dois lados mas contorno diverge
- gt_perdidos_ratio    : instâncias anotadas que o modelo não achou
- ghost_conf           : predição de alta confiança onde não há rótulo

Sem o modelo julgando só o que não viu, ele memoriza a máscara errada e o erro
some do radar (Princípio 3).
"""
from __future__ import annotations

import argparse
import csv
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import load as load_config


_TILE_SUFFIX_RE = re.compile(r"_r\d+_c\d+$")


# ---------------------------------------------------------------------------
# group_key — agrupa por TALHÃO. Ver Fase 0 para o porquê.
# ---------------------------------------------------------------------------
def group_key(path: Path) -> str:
    """Chave de agrupamento: todos os tiles do mesmo talhão caem no mesmo fold.

    Os dados brutos são ortomosaicos GeoTIFF gerados pelo ODM — 1 arquivo por
    talhão. O ``src/convert_to_yoloseg.py`` tileia cada ortomosaico em janelas
    de ``convert.tile_size`` píxeis com nomes no formato
    ``{talhao}_r{linha}_c{coluna}.jpg``. Tiles adjacentes do mesmo ortomosaico
    são visualmente quase idênticos — se caírem em folds diferentes, o modelo
    "acerta" o rótulo do vizinho por memória, o desacordo fica baixo, e um erro
    de anotação passa despercebido.

    Exemplos reais do dataset RigTech:

        CelsoSTE2_r012_c034.jpg   ->  CelsoSTE2
        Flaviano01_r003_c009.jpg  ->  Flaviano01
        Giasa_r045_c012.jpg       ->  Giasa

    TODO(multi-altitude): se um mesmo talhão passar a ter múltiplos voos
    (altitudes/datas diferentes), o conversor deve prefixar a data
    (``CelsoSTE2_20260527_r012_c034.jpg``) e esta função continuar retornando
    apenas ``CelsoSTE2`` — voos diferentes do mesmo talhão são o MESMO local
    geográfico e não podem cair em folds diferentes (o vazamento seria ainda
    pior que entre tiles adjacentes).
    """
    stem = path.stem
    # Casa APENAS o sufixo canônico do conversor: _r{digitos}_c{digitos} no fim.
    # Isso evita falso-positivo em talhões cujo nome contenha "_r" (ex.: MorroRedondo).
    m = _TILE_SUFFIX_RE.search(stem)
    if m:
        return stem[: m.start()]
    return stem


# --- máscaras e IoU ---------------------------------------------------------


def poly_to_mask(coords, size: int = 256):
    import cv2
    import numpy as np

    pts = np.array(coords, dtype=np.float32).reshape(-1, 2) * size
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_iou(a, b) -> float:
    import numpy as np

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


# --- particionamento --------------------------------------------------------


def make_folds(images: list[Path], k: int) -> list[list[Path]]:
    """Distribui imagens em k folds por grupo, em rodízio determinístico."""
    groups: dict[str, list[Path]] = {}
    for p in sorted(images):
        groups.setdefault(group_key(p), []).append(p)
    folds: list[list[Path]] = [[] for _ in range(k)]
    for i, g in enumerate(sorted(groups)):
        folds[i % k].extend(groups[g])
    return folds


# --- suspeita ---------------------------------------------------------------


@dataclass
class ImageSuspicion:
    image: str
    mean_iou: float
    gt_perdidos_ratio: float
    ghost_conf: float
    score: float
    n_gt: int
    n_pred: int


def _load_gt_polys(label_path: Path) -> list[tuple[int, list[float]]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, list[float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        out.append((cls, coords))
    return out


def score_image(
    gt_polys: list[tuple[int, list[float]]],
    pred_polys: list[tuple[int, list[float], float]],
    iou_low: float = 0.1,
) -> ImageSuspicion:
    """Casa instâncias com algoritmo guloso e combina os três sinais."""
    import numpy as np

    gt_masks = [poly_to_mask(coords) for _, coords in gt_polys]
    pred_masks = [poly_to_mask(coords) for _, coords, _ in pred_polys]

    used_pred: set[int] = set()
    ious: list[float] = []
    missed = 0
    for gm in gt_masks:
        best_iou = 0.0
        best_j = -1
        for j, pm in enumerate(pred_masks):
            if j in used_pred:
                continue
            iou = mask_iou(gm, pm)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_low:
            used_pred.add(best_j)
            ious.append(best_iou)
        else:
            missed += 1
            ious.append(0.0)

    mean_iou = float(np.mean(ious)) if ious else 1.0
    gt_perdidos_ratio = missed / len(gt_polys) if gt_polys else 0.0
    ghost_conf = max(
        (conf for j, (_, _, conf) in enumerate(pred_polys) if j not in used_pred),
        default=0.0,
    )
    score = (1.0 - mean_iou) + gt_perdidos_ratio + ghost_conf
    return ImageSuspicion(
        image="",
        mean_iou=mean_iou,
        gt_perdidos_ratio=gt_perdidos_ratio,
        ghost_conf=float(ghost_conf),
        score=float(score),
        n_gt=len(gt_polys),
        n_pred=len(pred_polys),
    )


# --- treino/inferência (Ultralytics) ---------------------------------------


def _write_fold_yaml(
    train_imgs: list[Path], val_imgs: list[Path], cfg, tmpdir: Path
) -> Path:
    """Monta um data.yaml apontando para links simbólicos das imagens do fold."""
    import yaml

    train_dir = tmpdir / "train" / "images"
    val_dir = tmpdir / "val" / "images"
    train_lbl = tmpdir / "train" / "labels"
    val_lbl = tmpdir / "val" / "labels"
    for d in (train_dir, val_dir, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    def _link(src_img: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
        (dst_img_dir / src_img.name).symlink_to(src_img.resolve())
        lbl = src_img.parents[1] / "labels" / f"{src_img.stem}.txt"
        if lbl.exists():
            (dst_lbl_dir / lbl.name).symlink_to(lbl.resolve())

    for p in train_imgs:
        _link(p, train_dir, train_lbl)
    for p in val_imgs:
        _link(p, val_dir, val_lbl)

    data_yaml = tmpdir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "path": str(tmpdir),
                "train": "train/images",
                "val": "val/images",
                "names": {i: n for i, n in enumerate(cfg.class_names)},
            },
            fh,
        )
    return data_yaml


def run_fold(
    train_imgs: list[Path], val_imgs: list[Path], cfg, workdir: Path
) -> dict[str, list[tuple[int, list[float], float]]]:
    """Treina um modelo nano no fold e retorna predições por imagem de val."""
    from ultralytics import YOLO

    data_yaml = _write_fold_yaml(train_imgs, val_imgs, cfg, workdir)
    model = YOLO(cfg.xval.arch)
    model.train(
        data=str(data_yaml),
        epochs=cfg.xval.epochs,
        imgsz=cfg.xval.imgsz,
        batch=cfg.xval.batch,
        seed=cfg.model.seed,
        project=str(workdir),
        name="fold",
        verbose=False,
    )

    predictions: dict[str, list[tuple[int, list[float], float]]] = {}
    for img in val_imgs:
        results = model.predict(source=str(img), imgsz=cfg.xval.imgsz, verbose=False)
        polys: list[tuple[int, list[float], float]] = []
        for r in results:
            if r.masks is None:
                continue
            for i, seg in enumerate(r.masks.xyn):
                cls = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                coords = seg.reshape(-1).tolist()
                polys.append((cls, coords, conf))
        predictions[img.name] = polys
    return predictions


def run(cfg) -> list[ImageSuspicion]:
    live = cfg.paths.live_dir
    images = sorted((live / "images").rglob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        raise FileNotFoundError(f"nenhuma imagem em {live/'images'}")

    folds = make_folds(images, cfg.xval.k)
    all_preds: dict[str, list[tuple[int, list[float], float]]] = {}

    for i, val_imgs in enumerate(folds):
        train_imgs = [p for j, fold in enumerate(folds) if j != i for p in fold]
        with tempfile.TemporaryDirectory(prefix=f"xval_fold{i}_") as td:
            print(f"[fold {i+1}/{cfg.xval.k}] treino={len(train_imgs)} val={len(val_imgs)}")
            preds = run_fold(train_imgs, val_imgs, cfg, Path(td))
            all_preds.update(preds)

    thr = cfg.xval.thresholds
    suspicions: list[ImageSuspicion] = []
    for img in images:
        gt = _load_gt_polys(live / "labels" / f"{img.stem}.txt")
        pred = all_preds.get(img.name, [])
        s = score_image(gt, pred)
        s.image = img.name
        enters = (
            s.mean_iou < thr.mean_iou_below
            or s.gt_perdidos_ratio > thr.missed_gt_ratio_above
            or s.ghost_conf > thr.ghost_pred_conf_above
        )
        if enters:
            suspicions.append(s)

    suspicions.sort(key=lambda x: x.score, reverse=True)
    return suspicions


def write_report(suspicions: list[ImageSuspicion], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "score", "mean_iou", "gt_perdidos_ratio", "ghost_conf", "n_gt", "n_pred"])
        for s in suspicions:
            writer.writerow(
                [s.image, f"{s.score:.4f}", f"{s.mean_iou:.4f}", f"{s.gt_perdidos_ratio:.4f}",
                 f"{s.ghost_conf:.4f}", s.n_gt, s.n_pred]
            )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Validação cruzada out-of-fold para achar anotações suspeitas")
    parser.add_argument("--out", type=Path, default=cfg.paths.suspects_report)
    args = parser.parse_args()

    suspicions = run(cfg)
    write_report(suspicions, args.out)
    print(f"{len(suspicions)} imagem(ns) suspeita(s) -> {args.out}")


if __name__ == "__main__":
    main()
