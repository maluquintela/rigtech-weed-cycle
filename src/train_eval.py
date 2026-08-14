"""Treino, avaliação no golden set e decisão de promoção.

Duas regras não-negociáveis:

1. O treino lê de um tarball imutável ``versions/vN/dataset.tar.gz``, extraído
   em disco LOCAL. Nunca lê o Drive montado — a latência por arquivo do Drive
   transforma cada época em horas de I/O.
2. A avaliação roda sempre no golden set congelado, que nunca entra em treino.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import load as load_config


def materialize(version: str, cfg) -> Path:
    """Extrai ``versions/vN/dataset.tar.gz`` para o disco local. Retorna o destino."""
    vdir = Path(cfg.paths.versions_dir) / version
    tar_path = vdir / "dataset.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"tarball ausente: {tar_path}")
    dest = Path(cfg.paths.work_dir) / "materialized" / version
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    return dest


def _write_data_yaml(materialized: Path, cfg, out_path: Path) -> Path:
    """Monta o data.yaml do Ultralytics — validação SEMPRE aponta para o golden set."""
    golden_images = Path(cfg.paths.golden_dir) / "images"
    if not golden_images.is_dir():
        raise FileNotFoundError(
            f"golden set ausente em {golden_images}. O golden set é pré-requisito de qualquer avaliação."
        )
    data = {
        "path": str(materialized),
        "train": "images",
        "val": str(golden_images),
        "names": {i: n for i, n in enumerate(cfg.class_names)},
    }
    out_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out_path


def _extract_metrics(results) -> dict[str, float]:
    """Achata as métricas do Ultralytics em um dict plano."""
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    return {
        "seg_mAP50-95": float(rd.get("metrics/mAP50-95(M)", rd.get("metrics/mAP50-95(B)", 0.0))),
        "seg_mAP50": float(rd.get("metrics/mAP50(M)", rd.get("metrics/mAP50(B)", 0.0))),
        "seg_precision": float(rd.get("metrics/precision(M)", rd.get("metrics/precision(B)", 0.0))),
        "seg_recall": float(rd.get("metrics/recall(M)", rd.get("metrics/recall(B)", 0.0))),
    }


def _append_history(history_path: Path, entry: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if history_path.exists():
        try:
            entries = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = []
    entries.append(entry)
    history_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _decide_promotion(history: list[dict], current: dict, cfg) -> tuple[bool, str]:
    """Aplica min_delta e guard_metrics contra o melhor histórico anterior."""
    if not history:
        return True, "primeiro run: promovido como baseline"

    metric = cfg.promotion.metric
    prev_best = max(history, key=lambda h: h["metrics"].get(metric, float("-inf")))
    delta = current["metrics"][metric] - prev_best["metrics"].get(metric, 0.0)

    if delta < cfg.promotion.min_delta:
        return False, f"ganho de {delta:+.4f} em {metric} abaixo de min_delta={cfg.promotion.min_delta}"

    for guard, threshold in vars(cfg.promotion.guard_metrics).items():
        prev_val = prev_best["metrics"].get(guard, 0.0)
        curr_val = current["metrics"].get(guard, 0.0)
        drop = curr_val - prev_val
        if drop < threshold:
            return False, f"métrica de guarda violada: {guard} caiu {drop:+.4f} (limite {threshold})"

    return True, f"ganho real de {delta:+.4f} em {metric}"


def run(
    version: str,
    cfg,
    arch: str | None = None,
    tag: str | None = None,
    batch: int | None = None,
    device: str | None = None,
    amp: bool | None = None,
    imgsz: int | None = None,
    epochs: int | None = None,
) -> dict:
    """Roda treino + avaliação no golden set. Retorna o registro salvo em history.json."""
    from ultralytics import YOLO

    materialized = materialize(version, cfg)
    run_id = f"{version}_{tag or 'run'}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(cfg.paths.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = _write_data_yaml(materialized, cfg, run_dir / "data.yaml")
    arch_name = arch or cfg.model.arch

    model = YOLO(arch_name)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs if epochs is not None else cfg.model.epochs,
        imgsz=imgsz if imgsz is not None else cfg.model.imgsz,
        batch=batch if batch is not None else cfg.model.batch,
        patience=cfg.model.patience,
        seed=cfg.model.seed,
        device=device if device is not None else cfg.model.device,
        project=str(run_dir),
        name="train",
        verbose=True,
    )
    # amp: override CLI vence; senão cai no config; senão default do Ultralytics
    if amp is not None:
        train_kwargs["amp"] = amp
    elif hasattr(cfg.model, "amp"):
        train_kwargs["amp"] = cfg.model.amp
    model.train(**train_kwargs)
    metrics = _extract_metrics(model.metrics)
    weights_path = run_dir / "train" / "weights" / "best.pt"

    entry = {
        "run_id": run_id,
        "version": version,
        "arch": arch_name,
        "seed": cfg.model.seed,
        "imgsz": cfg.model.imgsz,
        "epochs": cfg.model.epochs,
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "weights": str(weights_path),
    }

    history_path = Path(cfg.paths.history)
    prev_history: list[dict] = []
    if history_path.exists():
        try:
            prev_history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_history = []

    promoted, verdict = _decide_promotion(prev_history, entry, cfg)
    entry["promoted"] = promoted
    entry["verdict"] = verdict
    _append_history(history_path, entry)

    # Registro opcional em Linear: só roda se LINEAR_API_KEY estiver no ambiente.
    # Falha do post NÃO derruba o treino — a run já está no history.json.
    import os
    if os.environ.get("LINEAR_API_KEY"):
        try:
            from src.linear_versions import register_version
            ciclo_str = os.environ.get("CICLO_ATUAL")
            ciclo = int(ciclo_str) if ciclo_str and ciclo_str.isdigit() else None
            issue_id = register_version(
                entry=entry,
                team_key=cfg.linear.team_key,
                endpoint=cfg.linear.endpoint,
                ciclo=ciclo,
            )
            entry["linear_issue_id"] = issue_id
            print(f"[linear] issue criada: {issue_id}")
        except Exception as exc:
            # Post-treino é best-effort. Não vazar erro do Linear e perder o treino.
            print(f"[linear] falha ao postar (ignorado): {exc}")

    return entry


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Treino + avaliação no golden set")
    parser.add_argument("--version", required=True, help="identificador da versão (ex.: v3)")
    parser.add_argument("--arch", default=None, help="sobrescreve model.arch (para etapa 6)")
    parser.add_argument("--tag", default=None, help="rótulo livre para identificar o run")
    parser.add_argument("--batch", type=int, default=None, help="sobrescreve model.batch")
    parser.add_argument("--device", default=None, help="sobrescreve model.device (cpu | mps | 0)")
    parser.add_argument("--imgsz", type=int, default=None, help="sobrescreve model.imgsz")
    parser.add_argument("--epochs", type=int, default=None, help="sobrescreve model.epochs")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    entry = run(
        args.version, cfg,
        arch=args.arch, tag=args.tag,
        batch=args.batch, device=args.device, amp=args.amp,
        imgsz=args.imgsz, epochs=args.epochs,
    )
    print(json.dumps(entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
