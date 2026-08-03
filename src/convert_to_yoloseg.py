"""Conversor: ortomosaicos GeoTIFF + polígonos GeoJSON -> dataset YOLO-seg.

Estrutura de entrada (``paths.geosource_dir``):

    DaninhasTreinoClientes/
    ├── {talhao}/
    │   ├── imagem/{talhao}.tif        # 1 ortomosaico GeoTIFF (RGBA), ODM
    │   └── daninhas/*.geojson         # 1 GeoJSON por classe (nome codifica classe)

Estrutura de saída (``paths.live_dir``):

    work/live/
    ├── images/{talhao}_r{row}_c{col}.jpg
    └── labels/{talhao}_r{row}_c{col}.txt   # YOLO-seg, coords em [0, 1]

Regras invioláveis:

- Originais nunca são tocados. O conversor só lê.
- CRS: cada geojson é reprojetado para o CRS do ortomosaico correspondente
  antes da rasterização.
- Talhões cujo par (tif + pelo menos um geojson) estiver incompleto são
  ignorados com log — a ausência de dado nunca vira dado inventado.
- Alpha do ortomosaico é respeitado: tiles com fração de píxel válido abaixo
  de ``convert.min_valid_ratio`` são descartados.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import load as load_config


# ---------------------------------------------------------------------------
# Núcleo geométrico — puro Python/numpy, sem rasterio. Testável isoladamente.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileWindow:
    """Janela de tile em coordenadas de píxel do ortomosaico."""

    row: int
    col: int
    x_off: int
    y_off: int
    size: int


def iter_tile_windows(width: int, height: int, tile_size: int, stride: int) -> Iterable[TileWindow]:
    """Gera janelas cobrindo o ortomosaico com o stride pedido.

    A última linha/coluna encosta na borda direita/inferior (ancoragem à direita)
    para não perder pixels e não gerar tile de tamanho diferente.
    """
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size e stride precisam ser positivos")

    def _starts(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        starts = list(range(0, total - tile_size + 1, stride))
        if starts[-1] != total - tile_size:
            starts.append(total - tile_size)
        return starts

    xs = _starts(width)
    ys = _starts(height)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            yield TileWindow(row=r, col=c, x_off=x, y_off=y, size=tile_size)


def polygon_intersects_window(bounds: tuple[float, float, float, float], window: TileWindow) -> bool:
    """AABB (axis-aligned bounding box) simples — descarta polígonos claramente fora."""
    minx, miny, maxx, maxy = bounds
    wx0, wy0 = window.x_off, window.y_off
    wx1, wy1 = wx0 + window.size, wy0 + window.size
    return not (maxx < wx0 or minx > wx1 or maxy < wy0 or miny > wy1)


def polygon_to_yoloseg_line(
    class_id: int, ring_px: list[tuple[float, float]], window: TileWindow
) -> str | None:
    """Converte um anel poligonal em coord. de píxel do ortomosaico para uma
    linha YOLO-seg. Retorna None se o anel resultante for inválido
    (menos de 3 vértices ou área nula depois do clip).

    NOTA: este é o caminho simples — assume que o clip contra a borda do tile
    é feito pelo chamador (com shapely). Aqui só normaliza e formata.
    """
    if len(ring_px) < 3:
        return None
    coords_norm: list[str] = []
    for x, y in ring_px:
        nx = (x - window.x_off) / window.size
        ny = (y - window.y_off) / window.size
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        coords_norm.append(f"{nx:.6f}")
        coords_norm.append(f"{ny:.6f}")
    return " ".join([str(class_id)] + coords_norm)


# ---------------------------------------------------------------------------
# Descoberta de pares talhão-nível
# ---------------------------------------------------------------------------


@dataclass
class GeoSourcePair:
    talhao: str
    tif_path: Path
    geojson_by_class: dict[int, list[Path]]

    @property
    def is_complete(self) -> bool:
        return self.tif_path.exists() and any(self.geojson_by_class.values())


def _match_class(name: str, class_map: dict[str, int]) -> int | None:
    """Casamento por substring case-insensitive. Chaves mais longas ganham."""
    name_lower = name.lower()
    best: tuple[int, int] | None = None
    for key, cid in class_map.items():
        if key.lower() in name_lower:
            score = len(key)
            if best is None or score > best[0]:
                best = (score, cid)
    return best[1] if best else None


def discover_pairs(geosource_dir: Path, class_map: dict[str, int]) -> list[GeoSourcePair]:
    """Varre a árvore ``geosource_dir/{talhao}/`` procurando pares tif+geojson."""
    pairs: list[GeoSourcePair] = []
    for talhao_dir in sorted(p for p in geosource_dir.iterdir() if p.is_dir()):
        talhao = talhao_dir.name
        img_dir = talhao_dir / "imagem"
        gj_dir = talhao_dir / "daninhas"

        tifs = sorted(img_dir.glob("*.tif")) if img_dir.is_dir() else []
        tif = tifs[0] if tifs else Path("__missing__.tif")

        by_class: dict[int, list[Path]] = {}
        if gj_dir.is_dir():
            for gj in sorted(gj_dir.glob("*.geojson")):
                cid = _match_class(gj.stem, class_map)
                if cid is None:
                    print(f"  [aviso] {gj.name}: nome não casa com nenhuma classe conhecida")
                    continue
                by_class.setdefault(cid, []).append(gj)
        pairs.append(GeoSourcePair(talhao=talhao, tif_path=tif, geojson_by_class=by_class))
    return pairs


# ---------------------------------------------------------------------------
# Conversão de um talhão — depende de rasterio/shapely/pyproj. Isolada aqui
# para que o resto do módulo seja importável sem essas deps.
# ---------------------------------------------------------------------------


def convert_talhao(
    pair: GeoSourcePair,
    cfg,
    out_dir: Path,
    rng: random.Random,
    background_tiles: list[str] | None = None,
) -> tuple[int, int]:
    """Tileia o ortomosaico do talhão e gera pares imagem/label YOLO-seg.

    ``background_tiles``, se fornecido, é preenchido com os stems dos tiles
    salvos como background intencional — usado pelo caller para escrever o
    manifest ``background_tiles.json`` (lido por qa_static para evitar falso
    positivo em ``label_vazio``).

    Retorna (tiles_com_polígono, tiles_background_mantidos).
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.windows import Window
    from shapely.geometry import Polygon, box, shape
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pair.tif_path) as src:
        width, height = src.width, src.height
        tif_crs = src.crs
        tif_transform = src.transform
        band_count = src.count
        has_alpha = band_count >= 4

        # carrega e reprojeta todos os polígonos, ANOTANDO a classe
        polys_px: list[tuple[int, Polygon]] = []
        for class_id, gj_paths in pair.geojson_by_class.items():
            for gj_path in gj_paths:
                gj = json.loads(gj_path.read_text(encoding="utf-8"))
                src_crs_name = gj.get("crs", {}).get("properties", {}).get("name")
                # Default: FeatureCollection sem CRS = WGS84 (RFC 7946)
                src_crs = src_crs_name or "OGC:CRS84"
                to_tif = Transformer.from_crs(src_crs, tif_crs, always_xy=True).transform

                for feat in gj.get("features", []):
                    geom = shape(feat["geometry"])
                    geom_proj = shp_transform(to_tif, geom)
                    # geom_proj está em coord. do CRS do tif; converter para píxel:
                    for sub in _flatten_polygons(geom_proj):
                        ring_px = _crs_polygon_to_pixel(sub, tif_transform)
                        if ring_px is None:
                            continue
                        polys_px.append((class_id, ring_px))
        print(f"  {pair.talhao}: {len(polys_px)} polígonos após reprojeção/rasterização")

        n_with_poly = 0
        n_bg_kept = 0

        for window in iter_tile_windows(width, height, cfg.convert.tile_size, cfg.convert.stride):
            # lê o tile
            rw = Window(window.x_off, window.y_off, window.size, window.size)
            arr = src.read(window=rw)  # shape (bands, size, size)
            rgb = arr[:3].transpose(1, 2, 0)  # HWC
            if has_alpha:
                alpha = arr[3]
                valid_ratio = float((alpha > 0).mean())
                if valid_ratio < cfg.convert.min_valid_ratio:
                    continue

            # clip dos polígonos contra a janela do tile
            tile_box = box(window.x_off, window.y_off, window.x_off + window.size, window.y_off + window.size)
            lines: list[str] = []
            tile_area_px = window.size * window.size
            min_area_px = cfg.convert.min_polygon_area_norm * tile_area_px
            for class_id, ring_px in polys_px:
                poly = Polygon(ring_px)
                if not poly.is_valid or poly.is_empty:
                    continue
                if not poly.intersects(tile_box):
                    continue
                clipped = poly.intersection(tile_box)
                for sub in _flatten_polygons(clipped):
                    # descarta slivers minúsculos gerados pelo clip contra a borda
                    if sub.area < min_area_px:
                        continue
                    ext = list(sub.exterior.coords)[:-1]  # anel fechado -> abrir
                    line = polygon_to_yoloseg_line(class_id, ext, window)
                    if line is not None:
                        lines.append(line)

            stem = f"{pair.talhao}_r{window.row:03d}_c{window.col:03d}"
            if not lines:
                if rng.random() > cfg.convert.background_keep_ratio:
                    continue
                n_bg_kept += 1
                if background_tiles is not None:
                    background_tiles.append(stem)
            else:
                n_with_poly += 1

            Image.fromarray(rgb, mode="RGB").save(images_dir / f"{stem}.jpg", quality=90)
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        return n_with_poly, n_bg_kept


def _flatten_polygons(geom):
    """Achata Polygon/MultiPolygon/GeometryCollection numa lista de Polygons."""
    from shapely.geometry import MultiPolygon, Polygon, GeometryCollection

    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []


def _crs_polygon_to_pixel(poly, tif_transform) -> list[tuple[float, float]] | None:
    """Converte anel externo de um Polygon (em coord. CRS) para coord. de píxel."""
    if poly.is_empty:
        return None
    inv = ~tif_transform
    ring_px: list[tuple[float, float]] = []
    for x, y in list(poly.exterior.coords)[:-1]:
        px, py = inv * (x, y)
        ring_px.append((px, py))
    if len(ring_px) < 3:
        return None
    return ring_px


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Converte ortomosaicos + geojsons em dataset YOLO-seg")
    parser.add_argument("--geosource", type=Path, default=cfg.paths.geosource_dir)
    parser.add_argument("--out", type=Path, default=cfg.paths.live_dir)
    parser.add_argument("--seed", type=int, default=cfg.model.seed, help="semente para amostragem de background")
    args = parser.parse_args()

    class_map = vars(cfg.convert.geojson_class_map)
    pairs = discover_pairs(args.geosource, class_map)

    print(f"talhões encontrados: {len(pairs)}")
    for p in pairs:
        status = "COMPLETO" if p.is_complete else "INCOMPLETO"
        print(f"  [{status}] {p.talhao}")
        print(f"      tif: {p.tif_path if p.tif_path.exists() else '— AUSENTE'}")
        for cid, gjs in p.geojson_by_class.items():
            for gj in gjs:
                print(f"      geojson class={cid}: {gj.name}")

    rng = random.Random(args.seed)
    total_with = 0
    total_bg = 0
    background_tiles: list[str] = []
    for p in pairs:
        if not p.is_complete:
            print(f"[skip] {p.talhao}: par incompleto")
            continue
        with_poly, bg = convert_talhao(p, cfg, args.out, rng, background_tiles)
        total_with += with_poly
        total_bg += bg
        print(f"[ok]   {p.talhao}: {with_poly} tiles c/ polígono + {bg} background")

    # manifest de background: qa_static.py o lê para não gerar falso positivo
    # em label_vazio nesses tiles.
    manifest_path = args.out / "background_tiles.json"
    manifest_path.write_text(
        json.dumps({"stems": sorted(background_tiles)}, indent=2), encoding="utf-8"
    )
    print(f"\nTOTAL: {total_with} tiles com polígono, {total_bg} background -> {args.out}")
    print(f"manifest background: {manifest_path}")


if __name__ == "__main__":
    main()
