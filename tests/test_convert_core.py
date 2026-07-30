"""Testes do núcleo geométrico do conversor.

Apenas funções puras aqui — nada que dependa de rasterio/shapely/pyproj (o
teste completo do conversor exige um ortomosaico real e fica fora do CI).
"""
from __future__ import annotations

from pathlib import Path

from src.convert_to_yoloseg import (
    TileWindow,
    _match_class,
    discover_pairs,
    iter_tile_windows,
    polygon_intersects_window,
    polygon_to_yoloseg_line,
)


# ---------------------------------------------------------------------------
# iter_tile_windows
# ---------------------------------------------------------------------------


def test_iter_tile_windows_ortomosaico_menor_que_tile() -> None:
    """Se a imagem é menor que o tile, gera exatamente 1 tile na origem."""
    windows = list(iter_tile_windows(width=500, height=500, tile_size=1024, stride=896))
    assert len(windows) == 1
    assert windows[0].x_off == 0 and windows[0].y_off == 0


def test_iter_tile_windows_ancora_ultima_janela_na_borda() -> None:
    """Última janela deve tocar a borda direita/inferior — não perder píxeis."""
    windows = list(iter_tile_windows(width=2000, height=1500, tile_size=1024, stride=896))
    xs = sorted({w.x_off for w in windows})
    ys = sorted({w.y_off for w in windows})
    assert xs[-1] + 1024 == 2000
    assert ys[-1] + 1024 == 1500


def test_iter_tile_windows_row_col_sao_indices_sequenciais() -> None:
    windows = list(iter_tile_windows(width=2000, height=2000, tile_size=1024, stride=896))
    n_rows = len({w.row for w in windows})
    n_cols = len({w.col for w in windows})
    assert n_rows * n_cols == len(windows)


# ---------------------------------------------------------------------------
# polygon_intersects_window
# ---------------------------------------------------------------------------


def test_bbox_completamente_fora() -> None:
    w = TileWindow(row=0, col=0, x_off=0, y_off=0, size=100)
    assert not polygon_intersects_window((200, 200, 300, 300), w)


def test_bbox_encostando() -> None:
    w = TileWindow(row=0, col=0, x_off=0, y_off=0, size=100)
    assert polygon_intersects_window((50, 50, 150, 150), w)


# ---------------------------------------------------------------------------
# polygon_to_yoloseg_line
# ---------------------------------------------------------------------------


def test_polygon_line_normaliza_para_zero_um() -> None:
    w = TileWindow(row=0, col=0, x_off=100, y_off=100, size=1000)
    # triângulo em coord. de píxel do ortomosaico
    line = polygon_to_yoloseg_line(1, [(100, 100), (1100, 100), (600, 1100)], w)
    assert line is not None
    parts = line.split()
    assert parts[0] == "1"
    # (100,100) - offset (100,100) / 1000 = (0.0, 0.0)
    assert parts[1] == "0.000000"
    assert parts[2] == "0.000000"
    # (1100,100) -> (1.0, 0.0)
    assert parts[3] == "1.000000"
    assert parts[4] == "0.000000"


def test_polygon_line_recusa_menos_de_tres_vertices() -> None:
    w = TileWindow(row=0, col=0, x_off=0, y_off=0, size=100)
    assert polygon_to_yoloseg_line(0, [(1.0, 1.0), (2.0, 2.0)], w) is None


def test_polygon_line_clampeia_fora_do_tile() -> None:
    """Se um vértice cai fora do tile por 1 píxel (arredondamento), clampeia."""
    w = TileWindow(row=0, col=0, x_off=0, y_off=0, size=100)
    line = polygon_to_yoloseg_line(0, [(-1, -1), (101, -1), (50, 101)], w)
    assert line is not None
    for v in line.split()[1:]:
        f = float(v)
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# _match_class
# ---------------------------------------------------------------------------


def test_match_class_por_substring_case_insensitive() -> None:
    class_map = {"FolhaLarga": 0, "FolhaEstreita": 1, "Mamona": 2, "Mamonas": 2}
    assert _match_class("FolhaLargaCelsoSTE-2", class_map) == 0
    assert _match_class("FolhaEstreitaGiasa (1)", class_map) == 1
    assert _match_class("MamonasDoisRiosFlaviano", class_map) == 2
    assert _match_class("random.geojson", class_map) is None


def test_match_class_chave_mais_longa_vence() -> None:
    """Ambíguo por design: 'FolhasLargas' contém 'Larga' — o mapeamento explícito
    de chave mais longa protege contra colisão silenciosa."""
    class_map = {"Folha": 99, "FolhaLarga": 0}
    assert _match_class("FolhaLargaX", class_map) == 0


# ---------------------------------------------------------------------------
# discover_pairs
# ---------------------------------------------------------------------------


def test_discover_pairs_detecta_completos_e_incompletos(tmp_path: Path) -> None:
    class_map = {"FolhaLarga": 0, "FolhaEstreita": 1, "Mamona": 2, "Mamonas": 2}

    # talhão completo
    (tmp_path / "Bom" / "imagem").mkdir(parents=True)
    (tmp_path / "Bom" / "imagem" / "Bom.tif").write_bytes(b"")
    (tmp_path / "Bom" / "daninhas").mkdir(parents=True)
    (tmp_path / "Bom" / "daninhas" / "FolhaLargaBom.geojson").write_text("{}")

    # talhão só com geojson (sem imagem)
    (tmp_path / "SoGJ" / "daninhas").mkdir(parents=True)
    (tmp_path / "SoGJ" / "daninhas" / "FolhaLargaSoGJ.geojson").write_text("{}")

    # talhão só com imagem (sem geojson)
    (tmp_path / "SoIMG" / "imagem").mkdir(parents=True)
    (tmp_path / "SoIMG" / "imagem" / "SoIMG.tif").write_bytes(b"")

    pairs = {p.talhao: p for p in discover_pairs(tmp_path, class_map)}
    assert pairs["Bom"].is_complete
    assert not pairs["SoGJ"].is_complete
    assert not pairs["SoIMG"].is_complete
