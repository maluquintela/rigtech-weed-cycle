"""CLI: registra uma run existente do history.json em Linear.

Use após um treino no Colab (ou local) para postar a versão + métricas em
Linear. Idempotência: se você rodar duas vezes com o mesmo run_id, cria duas
issues (Linear não bloqueia duplicatas). Passe --dry-run pra inspecionar o
que seria postado antes de gastar API calls.

Exemplos:
    python -m src.link_run_to_linear --run-id v1_baseline_colab_c81b22
    python -m src.link_run_to_linear --last --ciclo 1
    python -m src.link_run_to_linear --run-id v2_loo_Giasa_xxx --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load as load_config
from src.linear_versions import build_description, build_title, register_version


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"history.json não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_entry(entries: list[dict], run_id: str | None, use_last: bool) -> dict:
    if use_last:
        if not entries:
            raise ValueError("history.json está vazio, nada para pegar como 'last'")
        return entries[-1]
    matches = [e for e in entries if e.get("run_id") == run_id]
    if not matches:
        raise ValueError(f"run_id '{run_id}' não encontrado em history.json")
    return matches[-1]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Registra run existente em Linear")
    parser.add_argument("--run-id", help="run_id específico do history.json")
    parser.add_argument("--last", action="store_true", help="pega a última entrada de history.json")
    parser.add_argument("--history", type=Path, default=cfg.paths.history)
    parser.add_argument("--ciclo", type=int, default=None, help="número do ciclo (para label ciclo-N)")
    parser.add_argument("--dry-run", action="store_true", help="imprime o que seria postado, não envia")
    args = parser.parse_args()

    if not args.run_id and not args.last:
        parser.error("informe --run-id ou --last")

    entries = _load_history(args.history)
    entry = _find_entry(entries, args.run_id, args.last)

    print(f"run selecionado: {entry['run_id']}")
    print(f"titulo: {build_title(entry)}")
    print("---")
    print(build_description(entry))
    print("---")

    if args.dry_run:
        print("[dry-run] nada foi postado.")
        return

    issue_id = register_version(
        entry=entry,
        team_key=cfg.linear.team_key,
        endpoint=cfg.linear.endpoint,
        ciclo=args.ciclo,
    )
    print(f"[ok] issue criada em Linear: {issue_id}")


if __name__ == "__main__":
    main()
