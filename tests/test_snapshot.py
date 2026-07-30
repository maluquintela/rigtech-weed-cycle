"""Testes de snapshot: imutabilidade da versão e correção do diff."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from src import snapshot


def _seed_live(root: Path, files: dict[str, str]) -> Path:
    live = root / "live"
    for rel, content in files.items():
        p = live / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return live


def test_primeira_versao_grava_manifest_changelog_e_tarball(tmp_path: Path) -> None:
    live = _seed_live(
        tmp_path,
        {
            "images/talhao_r000_c000.jpg": "img-bytes",
            "labels/talhao_r000_c000.txt": "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
        },
    )
    versions = tmp_path / "versions"

    vdir = snapshot.create(live, versions, note="baseline")
    assert vdir.name == "v1"
    assert (vdir / "manifest.json").exists()
    assert (vdir / "changelog.json").exists()
    assert (vdir / "dataset.tar.gz").exists()

    manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count_images"] == 1
    assert manifest["count_labels"] == 1
    assert manifest["note"] == "baseline"
    assert "images/talhao_r000_c000.jpg" in manifest["files"]

    changelog = json.loads((vdir / "changelog.json").read_text(encoding="utf-8"))
    assert changelog["de"] is None
    assert changelog["para"] == "v1"


def test_segunda_versao_calcula_diff_correto(tmp_path: Path) -> None:
    live = _seed_live(
        tmp_path,
        {
            "images/a.jpg": "A",
            "labels/a.txt": "0 0.1 0.1 0.9 0.1 0.5 0.9",
        },
    )
    versions = tmp_path / "versions"
    snapshot.create(live, versions, note="v1")

    # modifica um label, acrescenta um par, remove nada
    (live / "labels" / "a.txt").write_text("1 0.1 0.1 0.9 0.1 0.5 0.9", encoding="utf-8")
    (live / "images" / "b.jpg").write_text("B", encoding="utf-8")
    (live / "labels" / "b.txt").write_text("0 0.1 0.1 0.9 0.1 0.5 0.9", encoding="utf-8")

    v2 = snapshot.create(live, versions, note="v2")
    assert v2.name == "v2"
    changelog = json.loads((v2 / "changelog.json").read_text(encoding="utf-8"))
    assert changelog["de"] == "v1"
    assert changelog["resumo"]["modificados"] == 1
    assert changelog["resumo"]["adicionados"] == 2
    assert changelog["resumo"]["removidos"] == 0
    assert changelog["resumo"]["labels_modificados"] == 1
    assert "labels/a.txt" in changelog["modificados"]


def test_arquivos_nao_alterados_nao_aparecem_no_diff(tmp_path: Path) -> None:
    """Garante que o diff é por SHA-256, não por mtime — imprescindível no Drive."""
    live = _seed_live(tmp_path, {"images/a.jpg": "A", "labels/a.txt": "0 .1 .1 .9 .1 .5 .9"})
    versions = tmp_path / "versions"
    snapshot.create(live, versions, note="v1")

    # simula sync do Drive: toca mtime sem mudar conteúdo
    (live / "images" / "a.jpg").touch()
    (live / "labels" / "a.txt").touch()

    v2 = snapshot.create(live, versions, note="v2")
    changelog = json.loads((v2 / "changelog.json").read_text(encoding="utf-8"))
    assert changelog["resumo"]["modificados"] == 0
    assert changelog["resumo"]["adicionados"] == 0


def test_tarball_contem_todos_os_arquivos(tmp_path: Path) -> None:
    live = _seed_live(tmp_path, {"images/a.jpg": "A", "labels/a.txt": "0 .1 .1 .9 .1 .5 .9"})
    versions = tmp_path / "versions"
    vdir = snapshot.create(live, versions, note="v1")
    with tarfile.open(vdir / "dataset.tar.gz") as tar:
        names = {n.lstrip("./") for n in tar.getnames() if n not in {".", "./"}}
    assert "images/a.jpg" in names
    assert "labels/a.txt" in names
