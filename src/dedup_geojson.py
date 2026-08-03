"""Remove features duplicadas de um arquivo GeoJSON.

Duas features são consideradas duplicatas quando o primeiro anel do primeiro
polígono coincide (com arredondamento a 7 casas decimais) em todos os vértices.
O original nunca é tocado — a saída vai para um novo arquivo.

Uso:
    python -m src.dedup_geojson --in ARQUIVO.geojson [--out SAIDA.geojson]

Se ``--out`` não for informado, a saída é ``ARQUIVO.dedup.geojson`` ao lado do
original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _signature(feature: dict, decimals: int = 7) -> tuple:
    """Assinatura estável de uma feature GeoJSON para detectar duplicatas."""
    geom = feature["geometry"]
    gtype = geom["type"]
    coords = geom["coordinates"]
    # normaliza para lista de vértices do primeiro anel externo do primeiro polígono
    if gtype == "MultiPolygon":
        ring = coords[0][0]
    elif gtype == "Polygon":
        ring = coords[0]
    else:
        # geometrias diferentes de polígono — retorna algo único para não colapsar
        return (gtype, json.dumps(coords, sort_keys=True))
    return (gtype, tuple((round(x, decimals), round(y, decimals)) for x, y in ring))


def dedup(features: list[dict]) -> tuple[list[dict], list[dict]]:
    """Retorna (features_unicas, features_removidas) preservando ordem de entrada."""
    seen: dict[tuple, int] = {}
    unique: list[dict] = []
    removed: list[dict] = []
    for f in features:
        sig = _signature(f)
        if sig in seen:
            removed.append(f)
            continue
        seen[sig] = len(unique)
        unique.append(f)
    return unique, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove features duplicadas de um GeoJSON")
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="dst", type=Path, default=None)
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst or src.with_suffix(".dedup.geojson")

    gj = json.loads(src.read_text(encoding="utf-8"))
    features = gj.get("features", [])
    unique, removed = dedup(features)

    print(f"entrada:   {src}")
    print(f"features:  {len(features)}  →  únicas: {len(unique)}  |  removidas: {len(removed)}")

    if not removed:
        print("nada a fazer — não há duplicatas.")
        return

    gj_out = dict(gj)
    gj_out["features"] = unique
    dst.write_text(json.dumps(gj_out, indent=2), encoding="utf-8")
    print(f"saida:     {dst}")


if __name__ == "__main__":
    main()
