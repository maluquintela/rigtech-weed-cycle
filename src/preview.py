"""Gera PNG de comparação (tile + GT verde + predição vermelha) por tile suspeita.

Uso principal: acompanhar link_suspects_to_linear.py — antes de postar o card,
gera o preview e sobe pro Linear. Anotador vê a imagem inline no card e decide
em segundos se o rótulo está errado.

Layout do PNG: uma única figura mostrando a tile crua com dois overlays:
- polígonos GT: contorno verde grosso
- polígonos preditos: contorno vermelho fino, com conf no canto do polígono

Sem depender de matplotlib (evita dependência pesada). Usa PIL puro.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from src.config import load as load_config


CLASS_COLORS = {
    0: (46, 204, 113),   # folha_larga
    1: (52, 152, 219),   # folha_estreita
    2: (241, 196, 15),   # mamona
}


def _load_gt(label_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, list[tuple[float, float]]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        pts = list(zip(coords[0::2], coords[1::2]))
        if len(pts) >= 3:
            out.append((cls, pts))
    return out


def _predict(weights: Path, image: Path, imgsz: int, conf: float, device: str | None):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = {"source": str(image), "imgsz": imgsz, "conf": conf, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    polys: list[tuple[int, list[tuple[float, float]], float]] = []
    for r in results:
        if r.masks is None:
            continue
        for i, seg in enumerate(r.masks.xyn):
            cls = int(r.boxes.cls[i].item())
            pconf = float(r.boxes.conf[i].item())
            pts = [(float(x), float(y)) for x, y in seg]
            if len(pts) >= 3:
                polys.append((cls, pts, pconf))
    return polys


def render_preview(
    image_path: Path,
    label_path: Path,
    pred_polys: list[tuple[int, list[tuple[float, float]], float]],
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Compõe o PNG e salva. Todas as coordenadas de entrada em [0,1]."""
    im = Image.open(image_path).convert("RGB").copy()
    w, h = im.size
    draw = ImageDraw.Draw(im, "RGBA")

    # GT: contorno verde grosso, preenchimento translúcido
    for cls, pts in _load_gt(label_path):
        color = CLASS_COLORS.get(cls, (46, 204, 113))
        pixel_pts = [(x * w, y * h) for x, y in pts]
        draw.polygon(pixel_pts, outline=color + (255,), fill=color + (50,), width=3)

    # Pred: contorno vermelho, sem preenchimento
    for cls, pts, conf in pred_polys:
        pixel_pts = [(x * w, y * h) for x, y in pts]
        draw.polygon(pixel_pts, outline=(231, 76, 60, 255), width=2)
        try:
            font = ImageFont.load_default()
            draw.text(pixel_pts[0], f"{cls}·{conf:.2f}", fill=(231, 76, 60, 255), font=font)
        except Exception:
            pass

    if title:
        draw.rectangle([(0, 0), (w, 24)], fill=(0, 0, 0, 180))
        draw.text((6, 4), title, fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")
    return out_path


def generate_for_tile(
    tile_stem: str,
    weights: Path,
    golden_dir: Path,
    out_dir: Path,
    imgsz: int,
    conf: float = 0.25,
    device: str | None = None,
    title: str | None = None,
) -> Path:
    """Conveniência: pega uma tile pelo stem, gera preview e retorna o path."""
    img_candidates = [
        p for p in (golden_dir / "images").iterdir()
        if p.stem == tile_stem and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not img_candidates:
        raise FileNotFoundError(f"tile {tile_stem} não encontrada em {golden_dir}/images")
    img = img_candidates[0]
    lbl = golden_dir / "labels" / f"{tile_stem}.txt"
    preds = _predict(weights, img, imgsz, conf, device)
    return render_preview(img, lbl, preds, out_dir / f"{tile_stem}.png", title=title or tile_stem)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Gera preview PNG (GT verde vs predição vermelha) para uma tile")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tile", required=True, help="stem da tile (sem extensão)")
    parser.add_argument("--golden", type=Path, default=cfg.paths.golden_dir)
    parser.add_argument("--out-dir", type=Path, default=Path(cfg.paths.runs_dir) / "previews")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = generate_for_tile(
        tile_stem=args.tile, weights=args.weights, golden_dir=args.golden,
        out_dir=args.out_dir, imgsz=cfg.model.imgsz, conf=args.conf, device=args.device,
    )
    print(f"preview -> {out}")


if __name__ == "__main__":
    main()
