"""Inferencia DINOv3 + head Transformer treinada nos ciclos.

Uso:
    # por poligonos (1 crop por centroide)
    python -m src.infer \
        --ckpt /path/train_ckpt_c1.pt \
        --tif  /path/Giasa.tif \
        --polygons /path/algum.geojson \
        --out-prefix /path/pred_giasa

    # sliding window (grid de crops sobre a imagem, opcionalmente mascarado por plantacao)
    python -m src.infer \
        --ckpt /path/train_ckpt_c1.pt \
        --tif  /path/Giasa.tif \
        --sliding --stride 112 \
        --mask /path/Giasa_plantacao.geojson \
        --out-prefix /path/pred_giasa

Saidas:
    <out-prefix>.csv     -- id, centroid_col/row, centroid_lon/lat, pred_class, prob_<classe>
    <out-prefix>.geojson -- Points com as mesmas propriedades

Requer as mesmas dependencias do treino (torch, transformers, rasterio, geopandas, shapely).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from shapely.geometry import Point, box, mapping
from shapely.ops import unary_union
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ---- constantes fixas do ciclo (bater com o notebook de treino) --------------
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"
CROP_PIXELS = 224
PATCH = 16
CLASS_NAMES = {0: "cultivo", 1: "folha_larga", 2: "folha_estreita"}
N_CLASSES = len(CLASS_NAMES)
TRANSFORMER_LAYERS = 2
NHEAD = 8
TRANSFORMER_DROPOUT = 0.1
CLASSIFIER_DROPOUT = 0.0


class CropTransformerHead(nn.Module):
    def __init__(self, hidden_size, n_tokens, n_layers=2, nhead=8, dropout=0.1,
                 classifier_dropout=0.0, n_classes=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, n_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.class_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, tokens):
        x = self.input_norm(tokens) + self.pos_encoding
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.class_dropout(x)
        return self.classifier(x)


def load_backbone(device):
    proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
    mean = torch.tensor([float(x) for x in proc.image_mean], device=device).view(1, 3, 1, 1)
    std = torch.tensor([float(x) for x in proc.image_std], device=device).view(1, 3, 1, 1)

    model = AutoModel.from_pretrained(DINO_MODEL).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    patch = model.config.patch_size
    hidden = model.config.hidden_size
    n_register = getattr(model.config, "num_register_tokens", 0)
    supports_interp = "interpolate_pos_encoding" in inspect.signature(model.forward).parameters
    assert patch == PATCH and CROP_PIXELS % PATCH == 0
    n_tokens = (CROP_PIXELS // PATCH) ** 2
    return model, mean, std, hidden, n_register, n_tokens, supports_interp


def load_head(ckpt_path, hidden, n_tokens, device):
    ck = torch.load(ckpt_path, map_location=device)
    # aceita tanto o checkpoint de retomada (com best_state) quanto um state_dict cru
    if isinstance(ck, dict) and "best_state" in ck and ck["best_state"] is not None:
        state = ck["best_state"]
        print(f"  head: usando best_state (best_f1={ck.get('best_f1', float('nan')):.4f})")
    elif isinstance(ck, dict) and "head_state_dict" in ck:
        state = ck["head_state_dict"]
        print("  head: usando head_state_dict (ultima epoca)")
    else:
        state = ck
        print("  head: state_dict cru")

    head = CropTransformerHead(
        hidden_size=hidden, n_tokens=n_tokens, n_layers=TRANSFORMER_LAYERS,
        nhead=NHEAD, dropout=TRANSFORMER_DROPOUT, classifier_dropout=CLASSIFIER_DROPOUT,
        n_classes=N_CLASSES,
    ).to(device).eval()
    head.load_state_dict(state)
    return head


@torch.no_grad()
def extract_features(crops_uint8_bhw3, backbone, mean_t, std_t, n_register, n_tokens, supports_interp, device):
    t = torch.from_numpy(crops_uint8_bhw3.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    t = (t - mean_t) / std_t
    kwargs = {"interpolate_pos_encoding": True} if supports_interp else {}
    out = backbone(pixel_values=t, **kwargs)
    tokens = out.last_hidden_state.float()
    patch_tokens = tokens[:, 1 + n_register:]
    if patch_tokens.shape[1] != n_tokens:
        raise RuntimeError(f"esperava {n_tokens} tokens, vieram {patch_tokens.shape[1]}")
    return patch_tokens


def read_centered_crop(src, center_col, center_row):
    half = CROP_PIXELS // 2
    col_off = int(round(center_col - half))
    row_off = int(round(center_row - half))
    win = Window(col_off, row_off, CROP_PIXELS, CROP_PIXELS)
    tile = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
    return np.moveaxis(tile, 0, -1)


def centers_from_polygons(src, polygons_path):
    gdf = gpd.read_file(polygons_path)
    if gdf.crs is not None and gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
    centers = []
    for i, g in enumerate(gdf.geometry):
        if g is None or g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
            if g.is_empty:
                continue
        c = g.centroid
        col, row = ~src.transform * (c.x, c.y)
        centers.append({"id": i, "col": col, "row": row, "x": c.x, "y": c.y})
    return centers


def centers_from_sliding(src, stride, mask_path=None):
    half = CROP_PIXELS // 2
    mask_geom = None
    if mask_path:
        gdf = gpd.read_file(mask_path)
        if gdf.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        mask_geom = unary_union([g.buffer(0) if not g.is_valid else g for g in gdf.geometry if g and not g.is_empty])

    centers = []
    idx = 0
    rows = range(half, src.height - half, stride)
    cols = range(half, src.width - half, stride)
    for r in rows:
        for c in cols:
            x, y = src.transform * (c, r)
            if mask_geom is not None and not mask_geom.intersects(Point(x, y)):
                continue
            centers.append({"id": idx, "col": float(c), "row": float(r), "x": x, "y": y})
            idx += 1
    return centers


def batched(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def to_lonlat(src, x, y):
    """Converte coords do CRS do raster pra lon/lat (WGS84) via GeoSeries."""
    pt = gpd.GeoSeries([Point(x, y)], crs=src.crs).to_crs("EPSG:4326").iloc[0]
    return float(pt.x), float(pt.y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train_ckpt_c{N}.pt salvo pelo treino")
    ap.add_argument("--tif", required=True, help="ortomosaico GeoTIFF (RGB bandas 1,2,3)")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--polygons", help="geojson: prev por centroide de cada poligono")
    src_group.add_argument("--sliding", action="store_true", help="sliding window sobre o raster")
    ap.add_argument("--stride", type=int, default=CROP_PIXELS // 2, help="stride do sliding em pixels")
    ap.add_argument("--mask", default=None, help="geojson opcional pra restringir sliding (ex: plantacao)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-prefix", required=True, help="prefixo dos arquivos de saida (sem extensao)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--weed-threshold", type=float, default=0.5,
                    help="prob min de folha_larga OU folha_estreita pra tile virar positivo (sliding)")
    ap.add_argument("--merge-buffer", type=float, default=0.0,
                    help="buffer (unidades do CRS do raster) aplicado antes do union pra fechar gaps entre tiles")
    ap.add_argument("--min-polygon-area", type=float, default=0.0,
                    help="descarta poligonos com area < este valor (unidades^2 do CRS do raster)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("carregando DINOv3...")
    backbone, mean_t, std_t, hidden, n_register, n_tokens, supports_interp = load_backbone(device)
    print(f"  hidden={hidden} tokens={n_tokens} register={n_register} interp={supports_interp}")

    print("carregando head do checkpoint...")
    head = load_head(args.ckpt, hidden, n_tokens, device)

    with rasterio.open(args.tif) as src:
        print(f"raster: {src.width}x{src.height} crs={src.crs}")

        if args.polygons:
            centers = centers_from_polygons(src, args.polygons)
            print(f"{len(centers)} centroides carregados de {args.polygons}")
        else:
            centers = centers_from_sliding(src, args.stride, args.mask)
            print(f"{len(centers)} tiles de sliding (stride={args.stride}, mask={bool(args.mask)})")

        if not centers:
            print("nada pra inferir. saindo.")
            return

        results = []
        for chunk in tqdm(list(batched(centers, args.batch)), desc="infer"):
            crops = np.stack([read_centered_crop(src, c["col"], c["row"]) for c in chunk], axis=0)
            feats = extract_features(crops, backbone, mean_t, std_t, n_register, n_tokens, supports_interp, device)
            with torch.no_grad():
                logits = head(feats)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            for c, p, pr in zip(chunk, preds, probs):
                lon, lat = to_lonlat(src, c["x"], c["y"])
                row = {
                    "id": c["id"],
                    "centroid_col": c["col"],
                    "centroid_row": c["row"],
                    "centroid_x": c["x"],
                    "centroid_y": c["y"],
                    "centroid_lon": lon,
                    "centroid_lat": lat,
                    "pred_class_id": int(p),
                    "pred_class": CLASS_NAMES[int(p)],
                }
                for k in range(N_CLASSES):
                    row[f"prob_{CLASS_NAMES[k]}"] = float(pr[k])
                results.append(row)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    csv_path = args.out_prefix + ".csv"
    geojson_path = args.out_prefix + ".geojson"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"csv: {csv_path}")

    features = []
    for r in results:
        props = {k: v for k, v in r.items() if k not in ("centroid_x", "centroid_y")}
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(r["centroid_lon"], r["centroid_lat"])),
            "properties": props,
        })
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    with open(geojson_path, "w") as f:
        json.dump(fc, f)
    print(f"geojson: {geojson_path}")

    counts = {name: 0 for name in CLASS_NAMES.values()}
    for r in results:
        counts[r["pred_class"]] += 1
    print("distribuicao:", counts)

    if args.sliding:
        aggregate_polygons(args, results)


def aggregate_polygons(args, results):
    """Constroi caixas (CROP_PIXELS) em torno dos centros positivos, agrupa por
    classe daninha e faz unary_union pra virar poligonos contiguos."""
    with rasterio.open(args.tif) as src:
        px_w = abs(src.transform.a)
        px_h = abs(src.transform.e)
        raster_crs = src.crs

    half_w = (CROP_PIXELS / 2.0) * px_w
    half_h = (CROP_PIXELS / 2.0) * px_h

    weed_classes = {cid: name for cid, name in CLASS_NAMES.items() if cid != 0}

    per_class_boxes = {cid: [] for cid in weed_classes}
    kept = 0
    for r in results:
        pred_id = r["pred_class_id"]
        if pred_id == 0:
            continue
        best_weed_prob = max(r[f"prob_{name}"] for name in weed_classes.values())
        if best_weed_prob < args.weed_threshold:
            continue
        b = box(r["centroid_x"] - half_w, r["centroid_y"] - half_h,
                r["centroid_x"] + half_w, r["centroid_y"] + half_h)
        per_class_boxes[pred_id].append((b, best_weed_prob))
        kept += 1
    print(f"tiles positivos apos threshold {args.weed_threshold}: {kept}")

    if kept == 0:
        print("nenhum poligono a exportar.")
        return

    features = []
    for cid, boxes in per_class_boxes.items():
        if not boxes:
            continue
        geoms = [b for b, _ in boxes]
        if args.merge_buffer > 0:
            geoms = [g.buffer(args.merge_buffer) for g in geoms]
        merged = unary_union(geoms)
        if args.merge_buffer > 0:
            merged = merged.buffer(-args.merge_buffer)

        parts = list(merged.geoms) if merged.geom_type.startswith("Multi") else [merged]
        parts = [p for p in parts if not p.is_empty and p.area >= args.min_polygon_area]
        if not parts:
            continue

        gs = gpd.GeoSeries(parts, crs=raster_crs).to_crs("EPSG:4326")
        for i, geom in enumerate(gs.geometry):
            area_native = parts[i].area
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "class_id": cid,
                    "class": CLASS_NAMES[cid],
                    "area_crs": float(area_native),
                    "n_tiles_hint": len([1 for b, _ in per_class_boxes[cid] if b.intersects(parts[i])]),
                },
            })

    poly_path = args.out_prefix + "_polygons.geojson"
    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    with open(poly_path, "w") as f:
        json.dump(fc, f)
    print(f"poligonos agregados: {poly_path} ({len(features)} features)")


if __name__ == "__main__":
    main()
