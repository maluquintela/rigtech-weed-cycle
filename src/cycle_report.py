"""Gera relatório de fechamento de ciclo e posta como issue-mãe em Linear.

Percorre history.json e agrupa runs por ciclo (via labels 'ciclo-N' nas issues
em Linear). Produz tabela comparativa das versões, delta da métrica principal,
melhor arquitetura, verdicts de promoção. Cria uma issue tipo:

    Ciclo N — resumo
    ------------------
    | versão | tag | seg_mAP50-95 | delta | promoted |
    | v1 | baseline | 0.017 | — | ✓ |
    | v2 | pos_correcao_giasa | 0.089 | +0.072 | ✓ |
    ...
    Melhor run: v2 (delta +0.072)

Executar ao FIM de cada ciclo (depois das versões daquele ciclo terem sido
registradas). Idempotente: rodar duas vezes cria duas issues (Linear não
bloqueia duplicatas — se quiser, use --overwrite-issue <UUID>).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load as load_config
from src.linear_client import LinearClient


def _entries_for_ciclo(entries: list[dict], ciclo: int) -> list[dict]:
    """Filtra history.json entries pelo campo tag/ciclo. Heurística:
    - se a entry tem 'ciclo' explicito, filtra
    - senão, retorna todas (ciclo=None significa "todas as versões")
    """
    if ciclo is None:
        return entries
    tag_match = [e for e in entries if str(e.get("ciclo") or "") == str(ciclo)]
    return tag_match or entries  # se ninguém tem ciclo explícito, mostra tudo


def build_report(entries: list[dict], ciclo: int | None) -> tuple[str, str]:
    """Retorna (title, body_markdown)."""
    metric = "seg_mAP50-95"
    rows = []
    prev = None
    best = None
    for e in entries:
        m = e["metrics"].get(metric, 0.0)
        delta = f"{m - prev:+.4f}" if prev is not None else "—"
        promoted = "✓" if e.get("promoted") else "✗"
        rows.append(f"| `{e['version']}` | `{e.get('tag') or '-'}` | {m:.4f} | {delta} | {promoted} |")
        if best is None or m > best[1]:
            best = (e, m)
        prev = m

    body_lines = [
        f"**Runs incluídos**: {len(entries)}",
        f"**Métrica principal**: `{metric}`",
        "",
        "| versão | tag | seg_mAP50-95 | Δ | promoted |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    if best is not None:
        e, m = best
        body_lines.append(f"**Melhor run**: `{e['run_id']}` — {metric}={m:.4f}")
    if entries:
        first_m = entries[0]["metrics"].get(metric, 0.0)
        last_m = entries[-1]["metrics"].get(metric, 0.0)
        body_lines.append(f"**Progressão do ciclo**: {first_m:.4f} → {last_m:.4f}  ({last_m - first_m:+.4f})")

    ciclo_label = f"Ciclo {ciclo}" if ciclo is not None else "Todos os ciclos"
    title = f"{ciclo_label} — resumo · {len(entries)} run(s)"
    return title, "\n".join(body_lines)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Gera resumo do ciclo e posta em Linear")
    parser.add_argument("--history", type=Path, default=cfg.paths.history)
    parser.add_argument("--ciclo", type=int, default=None,
                        help="filtra runs deste ciclo. Omitido: usa todos.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"history.json não encontrado: {args.history}")
    entries = json.loads(args.history.read_text(encoding="utf-8"))
    entries = _entries_for_ciclo(entries, args.ciclo)
    if not entries:
        print(f"nenhum run para ciclo={args.ciclo}")
        return

    title, body = build_report(entries, args.ciclo)
    print(f"### {title}\n\n{body}")

    if args.dry_run:
        print("\n[dry-run] nada foi postado.")
        return

    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)
    client.create_label("resumo-ciclo", color="#27ae60")
    labels = ["resumo-ciclo"]
    if args.ciclo is not None:
        client.create_label(f"ciclo-{args.ciclo}", color="#8777d9")
        labels.append(f"ciclo-{args.ciclo}")
    issue_id = client.create_issue(
        title=title, description=body, state="Avaliado", labels=labels,
    )
    print(f"\n[ok] resumo posto em Linear: {issue_id}")


if __name__ == "__main__":
    main()
