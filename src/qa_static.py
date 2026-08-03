"""Faxina estrutural do dataset — CPU, segundos, sem carregar modelo.

Encontra defeitos que não são erro de julgamento do anotador, mas defeito
estrutural do arquivo (label órfão, polígono degenerado, class_id inválido,
conflito de classe, etc.). Rodar antes de qualquer coisa é uma decisão
econômica: não faz sentido gastar quatro treinos de GPU na validação cruzada
para descobrir que uma imagem tem polígono de dois pontos.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from src.config import load as load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SEVERITY_HIGH = "alta"
SEVERITY_MED = "media"


@dataclass(frozen=True)
class Finding:
    image: str
    issue: str
    severity: str
    detail: str


# --- geometria --------------------------------------------------------------


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Área do polígono pela fórmula do cadarço, coordenadas normalizadas."""
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _polygon_bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --- IO ---------------------------------------------------------------------


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _list_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        return []
    return sorted(labels_dir.rglob("*.txt"))


def _stem_map(paths: Iterable[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


# --- verificações por arquivo ----------------------------------------------


def _check_image(img_path: Path) -> list[Finding]:
    """Detecta corrupção sem carregar a imagem inteira em memória."""
    try:
        with Image.open(img_path) as im:
            im.verify()
        return []
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return [Finding(img_path.name, "imagem_corrompida", SEVERITY_HIGH, str(exc))]


def _parse_label_line(line: str, nc: int) -> tuple[int | None, list[tuple[float, float]] | None, str | None]:
    """Retorna (class_id, pontos, mensagem_de_erro). Erro exclusivo com sucesso."""
    parts = line.strip().split()
    if not parts:
        return None, None, None  # linha em branco: ignorar silenciosamente
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError as exc:
        return None, None, f"linha_malformada: {exc}"
    if class_id < 0 or class_id >= nc:
        return class_id, None, f"classe_invalida: class_id={class_id} (nc={nc})"
    if len(coords) < 6 or len(coords) % 2 != 0:
        return class_id, None, f"poligono_degenerado: {len(coords)} coordenadas"
    pts = list(zip(coords[0::2], coords[1::2]))
    for x, y in pts:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return class_id, pts, f"coord_fora_do_range: ({x:.4f}, {y:.4f})"
    return class_id, pts, None


def _check_label(label_path: Path, nc: int) -> list[Finding]:
    findings: list[Finding] = []
    try:
        raw = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Finding(label_path.name, "linha_malformada", SEVERITY_HIGH, f"leitura falhou: {exc}")]

    instances: list[tuple[int, list[tuple[float, float]], tuple[float, float, float, float], float]] = []
    non_empty_lines = 0
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        non_empty_lines += 1
        class_id, pts, err = _parse_label_line(line, nc)
        if err is not None:
            issue = err.split(":", 1)[0]
            findings.append(Finding(label_path.name, issue, SEVERITY_HIGH, f"L{lineno}: {err}"))
            continue
        assert class_id is not None and pts is not None
        area = _polygon_area(pts)
        if area < 1e-5:
            findings.append(
                Finding(label_path.name, "instancia_minuscula", SEVERITY_MED, f"L{lineno}: área={area:.2e}")
            )
            continue
        instances.append((class_id, pts, _polygon_bbox(pts), area))

    if non_empty_lines == 0:
        findings.append(Finding(label_path.name, "label_vazio", SEVERITY_MED, "arquivo sem instâncias"))

    # duplicatas e conflitos de classe via bbox-IoU
    for i in range(len(instances)):
        ci, _, bi, _ = instances[i]
        for j in range(i + 1, len(instances)):
            cj, _, bj, _ = instances[j]
            iou = _bbox_iou(bi, bj)
            if iou < 0.85:
                continue
            if ci == cj:
                findings.append(
                    Finding(
                        label_path.name,
                        "instancia_duplicada",
                        SEVERITY_MED,
                        f"L{i+1} e L{j+1}: bbox-IoU={iou:.2f}",
                    )
                )
            else:
                findings.append(
                    Finding(
                        label_path.name,
                        "conflito_de_classe",
                        SEVERITY_HIGH,
                        f"L{i+1}(cls={ci}) x L{j+1}(cls={cj}): bbox-IoU={iou:.2f}",
                    )
                )
    return findings


# --- pipeline ---------------------------------------------------------------


def _load_background_stems(dataset_dir: Path) -> set[str]:
    """Lê ``background_tiles.json`` (opcional) gerado pelo conversor.

    Tiles listados aqui foram salvos DELIBERADAMENTE como background pelo
    conversor — não devem gerar ``label_vazio`` no QA.
    """
    manifest = dataset_dir / "background_tiles.json"
    if not manifest.exists():
        return set()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return set(data.get("stems", []))
    except json.JSONDecodeError:
        return set()


def run(dataset_dir: Path, nc: int) -> list[Finding]:
    """Roda todas as verificações. Espera estrutura ``dataset_dir/{images,labels}/``."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset não encontrado: {dataset_dir}")

    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)
    img_by_stem = _stem_map(images)
    lbl_by_stem = _stem_map(labels)
    background_stems = _load_background_stems(dataset_dir)

    findings: list[Finding] = []

    for stem, img_path in img_by_stem.items():
        findings.extend(_check_image(img_path))
        if stem not in lbl_by_stem:
            findings.append(
                Finding(img_path.name, "imagem_sem_label", SEVERITY_HIGH, "declarar como fundo se intencional")
            )

    for stem, lbl_path in lbl_by_stem.items():
        if stem not in img_by_stem:
            findings.append(Finding(lbl_path.name, "label_orfao", SEVERITY_HIGH, "sem imagem correspondente"))
            continue
        for f in _check_label(lbl_path, nc):
            # tiles marcados como background pelo conversor são vazios de
            # propósito — não gerar falso positivo em label_vazio.
            if f.issue == "label_vazio" and stem in background_stems:
                continue
            findings.append(f)

    return findings


def write_report(findings: list[Finding], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "issue", "severity", "detail"])
        for f in findings:
            writer.writerow([f.image, f.issue, f.severity, f.detail])


def _summary(findings: list[Finding]) -> str:
    from collections import Counter

    by_issue = Counter(f.issue for f in findings)
    high = sum(1 for f in findings if f.severity == SEVERITY_HIGH)
    lines = [f"{len(findings)} achado(s)"]
    for issue, n in by_issue.most_common():
        lines.append(f"  {n:>4} {issue}")
    lines.append("")
    lines.append(f"{high} de severidade ALTA — corrigir antes de qualquer treino.")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="QA estrutural do dataset (CPU, sem modelo)")
    parser.add_argument("--dataset", type=Path, default=cfg.paths.live_dir, help="pasta com images/ e labels/")
    parser.add_argument("--out", type=Path, default=cfg.paths.qa_report, help="CSV de saída")
    args = parser.parse_args()

    findings = run(args.dataset, cfg.nc)
    write_report(findings, args.out)
    print(_summary(findings))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
