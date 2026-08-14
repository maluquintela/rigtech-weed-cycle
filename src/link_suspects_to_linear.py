"""Posta tiles suspeitas como issues em Linear, com preview PNG anexado.

Fluxo por tile:
1. `find_suspects` roda o modelo no golden e classifica tiles suspeitas
2. Para as top-N, gera preview PNG (GT vs predição) via `src.preview`
3. Sobe o PNG pro Linear via fileUpload
4. Cria a issue em `Selecionadas` com labels `suspeita`+`ciclo-N`, corpo
   com métricas e o PNG embedado via markdown

Auto-detect de ciclo: se --ciclo não for informado, conta issues com label
'versao' já criadas e usa esse número (= ciclo atual em curso).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load as load_config
from src.linear_client import LinearClient
from src.preview import generate_for_tile
from src.suspects import find_suspects
from src.xval_flag import ImageSuspicion


def _resolve_ciclo(client: LinearClient, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    # heurística: contamos versões já registradas. Ciclo em curso = N+1.
    n = client.count_issues_by_label("versao")
    return max(1, n)


def _build_body(s: ImageSuspicion, weights: Path, preview_url: str | None) -> str:
    reasons = []
    if s.gt_perdidos_ratio > 0:
        reasons.append(f"- **{s.gt_perdidos_ratio:.0%}** das anotações não foram detectadas")
    if s.ghost_conf > 0.5:
        reasons.append(f"- predição fantasma com conf **{s.ghost_conf:.2f}** onde não há rótulo")
    if s.mean_iou < 0.9 and s.n_gt > 0:
        reasons.append(f"- IoU médio entre GT e predição **{s.mean_iou:.2f}** (baixo)")
    reasons_md = "\n".join(reasons) if reasons else "- (motivo genérico)"

    preview_md = f"\n![preview]({preview_url})\n" if preview_url else ""
    return (
        f"**Tile**: `{s.image}`\n"
        f"**Score de suspeita**: {s.score:.3f}\n"
        f"**Instâncias GT**: {s.n_gt}  ·  **Predições**: {s.n_pred}\n"
        f"{preview_md}\n"
        f"### Motivos\n{reasons_md}\n\n"
        f"### Legenda do preview\n"
        f"- **Verde**: anotação atual (GT)\n"
        f"- **Vermelho**: predição do modelo\n\n"
        f"### Ação\n"
        f"1. Abrir `{s.image}` no anotador\n"
        f"2. Comparar rótulo atual com a predição do modelo\n"
        f"3. Corrigir se rótulo estiver errado; mover card para **Em anotação**\n"
        f"4. Depois de revisado, mover para **Em revisão (QA)**\n\n"
        f"**Modelo avaliador**: `{weights.name}`"
    )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Posta suspeitas em Linear com preview PNG")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=cfg.paths.golden_dir)
    parser.add_argument("--ciclo", type=int, default=None, help="se omitido, detecta pelo Linear")
    parser.add_argument("--parent-issue", default=None, help="UUID da issue-mãe da versão")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="pula geração+upload do preview")
    parser.add_argument("--preview-dir", type=Path, default=None)
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
    ciclo = _resolve_ciclo(client, args.ciclo)
    print(f"[ciclo] usando ciclo-{ciclo}")

    client.create_label("suspeita", color="#eb5757")
    ciclo_label = f"ciclo-{ciclo}"
    client.create_label(ciclo_label, color="#8777d9")
    labels = ["suspeita", ciclo_label]

    preview_dir = args.preview_dir or (Path(cfg.paths.runs_dir) / "previews" / f"ciclo_{ciclo}")

    for s in suspicions:
        preview_url = None
        if not args.no_preview:
            try:
                png_path = generate_for_tile(
                    tile_stem=Path(s.image).stem,
                    weights=args.weights, golden_dir=args.golden,
                    out_dir=preview_dir, imgsz=cfg.model.imgsz,
                    conf=args.conf, device=args.device,
                    title=f"{s.image} · score={s.score:.2f}",
                )
                preview_url = client.upload_file(png_path)
            except Exception as exc:
                print(f"  [preview-fail] {s.image}: {exc}")

        title = f"suspeita · {s.image} · score={s.score:.2f}"
        body = _build_body(s, args.weights, preview_url)
        issue_id = client.create_issue(
            title=title, description=body,
            state="Selecionadas", labels=labels,
            parent_id=args.parent_issue,
        )
        marker = "📎" if preview_url else "  "
        print(f"  [ok]{marker} {issue_id}  {title}")


if __name__ == "__main__":
    main()
