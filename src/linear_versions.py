"""Registro de versões de treino em Linear.

Cada run de treino vira uma issue com estado, métricas, e link pros pesos.
A issue serve como memória compartilhada do ciclo — todo mundo do time vê no
Linear a evolução das métricas versão a versão sem precisar do repo/logs.

Diferença de nomenclatura:
- "run" = uma execução de treino (arquivo em work/runs/history.json)
- "versão" = o snapshot do dataset (v1, v2, ...) usado por esse run
Uma versão do dataset pode ter várias runs (arquitetura diferente, LOO folds).
"""
from __future__ import annotations

from typing import Any

from src.linear_client import LinearClient


def _metrics_table(metrics: dict[str, float]) -> str:
    """Renderiza métricas do run como tabela markdown."""
    rows = ["| métrica | valor |", "|---|---|"]
    for k in ("seg_mAP50-95", "seg_mAP50", "seg_precision", "seg_recall"):
        v = metrics.get(k)
        if v is None:
            continue
        rows.append(f"| {k} | {v:.4f} |")
    return "\n".join(rows)


def _state_for_entry(entry: dict[str, Any]) -> str:
    """Decide o state Linear a partir do resultado da promoção."""
    if entry.get("promoted"):
        return "Aprovado"
    return "Avaliado"


def build_description(entry: dict[str, Any], extra: dict[str, str] | None = None) -> str:
    """Corpo em markdown da issue de versão. ``extra`` mescla campos custom."""
    lines = [
        f"**run_id**: `{entry['run_id']}`",
        f"**versão do dataset**: `{entry['version']}`",
        f"**tag**: `{entry.get('tag') or '-'}`",
        f"**arquitetura**: `{entry['arch']}`",
        f"**imgsz**: {entry['imgsz']}  ·  **seed**: {entry['seed']}  ·  **epochs**: {entry['epochs']}",
        f"**criado em**: {entry['created_at']}",
        "",
        "## Métricas no golden set",
        _metrics_table(entry["metrics"]),
        "",
        f"**Veredito de promoção**: {entry.get('verdict', '-')}",
        f"**Promoted**: {'sim' if entry.get('promoted') else 'não'}",
        "",
        f"**Pesos**: `{entry.get('weights', '-')}`",
    ]
    if extra:
        lines.extend(["", "## Contexto adicional"])
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def build_title(entry: dict[str, Any]) -> str:
    """Título da issue. Ex.: `v1 · baseline · seg_mAP50-95=0.0170`"""
    tag = entry.get("tag") or "run"
    m = entry["metrics"].get("seg_mAP50-95", 0.0)
    return f"{entry['version']} · {tag} · seg_mAP50-95={m:.4f}"


def register_version(
    entry: dict[str, Any],
    team_key: str,
    endpoint: str = "https://api.linear.app/graphql",
    api_key: str | None = None,
    extra: dict[str, str] | None = None,
    ciclo: int | None = None,
) -> str:
    """Cria a issue de versão em Linear e retorna o id."""
    client = LinearClient(team_key=team_key, endpoint=endpoint, api_key=api_key)

    # garante labels necessárias
    client.create_label("versao", color="#5e6ad2")
    client.create_label("run-modelo", color="#57d9a3")
    labels = ["versao", "run-modelo"]
    if ciclo is not None:
        ciclo_label = f"ciclo-{ciclo}"
        client.create_label(ciclo_label, color="#8777d9")
        labels.append(ciclo_label)

    tag = entry.get("tag") or ""
    if tag:
        arch_label = f"tag-{tag.replace(' ', '-').replace('/', '-')}"
        client.create_label(arch_label, color="#95a5a6")
        labels.append(arch_label)

    return client.create_issue(
        title=build_title(entry),
        description=build_description(entry, extra=extra),
        state=_state_for_entry(entry),
        labels=labels,
    )
