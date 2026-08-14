"""Posta tiles suspeitas (saída de src.suspects) como issues em Linear.

Cada tile vira um card em state `Selecionadas` com labels `suspeita` e
`ciclo-N` (se passado --ciclo). Se --parent-issue for passado, os cards
viram sub-issues da issue-mãe da versão (linkados na UI).

Uso:
    python -m src.link_suspects_to_linear --weights work/runs/vX/train/weights/best.pt \\
        --ciclo 1 [--parent-issue <UUID>] [--top-n 20] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load as load_config
from src.linear_client import LinearClient
from src.suspects import find_suspects
from src.xval_flag import ImageSuspicion


def _build_body(s: ImageSuspicion, weights: Path) -> str:
    reasons = []
    if s.gt_perdidos_ratio > 0:
        reasons.append(f"- **{s.gt_perdidos_ratio:.0%}** das anotações não foram detectadas")
    if s.ghost_conf > 0.5:
        reasons.append(f"- predição fantasma com conf **{s.ghost_conf:.2f}** onde não há rótulo")
    if s.mean_iou < 0.9 and s.n_gt > 0:
        reasons.append(f"- IoU médio entre GT e predição **{s.mean_iou:.2f}** (baixo)")
    reasons_md = "\n".join(reasons) if reasons else "- (motivo genérico)"
    return (
        f"**Tile**: `{s.image}`\n"
        f"**Score de suspeita**: {s.score:.3f}\n"
        f"**Instâncias GT**: {s.n_gt}  ·  **Predições**: {s.n_pred}\n\n"
        f"### Motivos\n{reasons_md}\n\n"
        f"### Ação\n"
        f"1. Abrir `{s.image}` no anotador\n"
        f"2. Comparar rótulo atual com a predição do modelo\n"
        f"3. Corrigir se rótulo estiver errado; mover card para **Em anotação**\n"
        f"4. Depois de revisado, mover para **Em revisão (QA)**\n\n"
        f"**Modelo avaliador**: `{weights.name}`"
    )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Posta suspeitas como cards em Linear")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=cfg.paths.golden_dir)
    parser.add_argument("--ciclo", type=int, default=None, help="número do ciclo (label ciclo-N)")
    parser.add_argument("--parent-issue", default=None, help="UUID da issue-mãe (versão)")
    parser.add_argument("--top-n", type=int, default=20, help="limita a N piores tiles")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suspicions = find_suspects(
        weights=args.weights, golden_dir=args.golden, cfg=cfg,
        conf=args.conf, device=args.device,
    )
    suspicions = suspicions[: args.top_n]
    print(f"detectadas {len(suspicions)} tile(s) suspeita(s):")
    for s in suspicions:
        print(f"  {s.image:40} score={s.score:.3f}")

    if args.dry_run:
        print("[dry-run] nada foi postado.")
        return

    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)
    client.create_label("suspeita", color="#eb5757")
    labels = ["suspeita"]
    if args.ciclo is not None:
        ciclo_label = f"ciclo-{args.ciclo}"
        client.create_label(ciclo_label, color="#8777d9")
        labels.append(ciclo_label)

    for s in suspicions:
        title = f"suspeita · {s.image} · score={s.score:.2f}"
        body = _build_body(s, args.weights)
        issue_id = client.create_issue(
            title=title,
            description=body,
            state="Selecionadas",
            labels=labels,
            parent_id=args.parent_issue,
        )
        print(f"  [ok] {issue_id}  {title}")


if __name__ == "__main__":
    main()
