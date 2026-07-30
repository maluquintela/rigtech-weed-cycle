"""Testes de qa_static com dataset sintético gerado no próprio teste.

Cada teste cria uma pasta temporária com um mini dataset YOLO-seg
(imagens PNG minúsculas + labels .txt inventados) e verifica que os
achados esperados aparecem — e que os NÃO-erros não geram falso positivo.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src import qa_static


NC = 2  # folha_larga, folha_estreita — suficiente para os testes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_dataset(tmp: Path) -> tuple[Path, Path]:
    images = tmp / "images"
    labels = tmp / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    return images, labels


def _mk_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    Image.new("RGB", size, color=(120, 200, 90)).save(path)


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _issues(findings: list[qa_static.Finding], image: str | None = None) -> set[str]:
    return {f.issue for f in findings if image is None or f.image.startswith(image)}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_dataset_sem_problemas_nao_gera_achado(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "a.png")
    # triângulo válido, dentro de [0,1], área não-degenerada
    _write_label(labels / "a.txt", ["0 0.1 0.1 0.9 0.1 0.5 0.9"])
    findings = qa_static.run(tmp_path, NC)
    assert findings == [], f"achou algo indevido: {findings}"


# ---------------------------------------------------------------------------
# defeitos estruturais
# ---------------------------------------------------------------------------


def test_label_orfao(tmp_path: Path) -> None:
    _, labels = _mk_dataset(tmp_path)
    _write_label(labels / "solto.txt", ["0 0.1 0.1 0.9 0.1 0.5 0.9"])
    assert "label_orfao" in _issues(qa_static.run(tmp_path, NC))


def test_imagem_sem_label(tmp_path: Path) -> None:
    images, _ = _mk_dataset(tmp_path)
    _mk_image(images / "b.png")
    assert "imagem_sem_label" in _issues(qa_static.run(tmp_path, NC))


def test_classe_invalida(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "c.png")
    _write_label(labels / "c.txt", ["5 0.1 0.1 0.9 0.1 0.5 0.9"])  # class_id=5, nc=2
    assert "classe_invalida" in _issues(qa_static.run(tmp_path, NC))


def test_poligono_degenerado(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "d.png")
    _write_label(labels / "d.txt", ["0 0.1 0.1 0.2 0.2"])  # apenas 2 vértices
    assert "poligono_degenerado" in _issues(qa_static.run(tmp_path, NC))


def test_coord_fora_do_range(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "e.png")
    _write_label(labels / "e.txt", ["0 0.1 0.1 1.2 0.1 0.5 0.9"])  # 1.2 > 1
    assert "coord_fora_do_range" in _issues(qa_static.run(tmp_path, NC))


def test_linha_malformada(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "f.png")
    _write_label(labels / "f.txt", ["esta linha nao eh numerica"])
    assert "linha_malformada" in _issues(qa_static.run(tmp_path, NC))


def test_instancia_minuscula(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "g.png")
    # triângulo minúsculo perto de (0,0): área << 1e-5
    _write_label(labels / "g.txt", ["0 0.001 0.001 0.002 0.001 0.001 0.002"])
    assert "instancia_minuscula" in _issues(qa_static.run(tmp_path, NC))


def test_label_vazio(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "h.png")
    (labels / "h.txt").write_text("", encoding="utf-8")
    assert "label_vazio" in _issues(qa_static.run(tmp_path, NC))


# ---------------------------------------------------------------------------
# conflito e duplicata (usam bbox-IoU >= 0.85 entre instâncias)
# ---------------------------------------------------------------------------


def test_conflito_de_classe(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "i.png")
    # duas instâncias com bbox quase idêntica, classes diferentes
    _write_label(
        labels / "i.txt",
        [
            "0 0.1 0.1 0.9 0.1 0.5 0.9",
            "1 0.1 0.1 0.9 0.1 0.5 0.9",
        ],
    )
    assert "conflito_de_classe" in _issues(qa_static.run(tmp_path, NC))


def test_instancia_duplicada(tmp_path: Path) -> None:
    images, labels = _mk_dataset(tmp_path)
    _mk_image(images / "j.png")
    _write_label(
        labels / "j.txt",
        [
            "0 0.1 0.1 0.9 0.1 0.5 0.9",
            "0 0.1 0.1 0.9 0.1 0.5 0.9",
        ],
    )
    assert "instancia_duplicada" in _issues(qa_static.run(tmp_path, NC))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_write_report_grava_csv(tmp_path: Path) -> None:
    out = tmp_path / "report.csv"
    findings = [qa_static.Finding("a.png", "label_vazio", qa_static.SEVERITY_MED, "detalhe")]
    qa_static.write_report(findings, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "image,issue,severity,detail"
    assert "a.png,label_vazio,media,detalhe" in lines[1]
