"""Monitora o estado dos cards do ciclo no Linear.

Duas modos:
- ``--once``: imprime a tabela de contagem por state e sai.
- ``--interval N``: fica em loop, imprime a tabela a cada N segundos.

Também alerta quando cards em ``Pronto para treino`` cruzam um limiar
(``--trigger-at 40`` = "quando N ≥ 40, imprime alerta pra treinar").

Uso típico: rodar `python -m src.watch_cards --once` no terminal quando
quiser saber "está pronto pra treinar?". Ou colocar em cron (macOS launchd
ou crontab) rodando de hora em hora para receber notificação por email
quando o limiar for cruzado.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from src.config import load as load_config
from src.linear_client import LinearClient


STATES_TO_WATCH = [
    "Selecionadas",
    "Em anotação",
    "Em revisão (QA)",
    "Pronto para treino",
    "Em treinamento",
    "Avaliado",
    "Aprovado",
]


def _snapshot(client: LinearClient) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in STATES_TO_WATCH:
        try:
            issues = client.issues_in_state(state)
            counts[state] = len(issues)
        except Exception as exc:
            counts[state] = -1  # marca como erro
            print(f"  [erro] state '{state}': {exc}")
    return counts


def _print_table(counts: dict[str, int], trigger_at: int | None = None, ready_state: str = "Pronto para treino") -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}]")
    print(f"{'state':<24}{'cards':>8}")
    print("-" * 32)
    for state in STATES_TO_WATCH:
        n = counts.get(state, 0)
        marker = ""
        if state == ready_state and trigger_at is not None and n >= trigger_at:
            marker = "  ← 🚨 rodar treino"
        print(f"{state:<24}{n:>8}{marker}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Monitor de cards no Linear")
    parser.add_argument("--once", action="store_true", help="uma tabela e sai (default se --interval não passar)")
    parser.add_argument("--interval", type=int, default=None, help="loop, segundos entre snapshots")
    parser.add_argument("--trigger-at", type=int, default=None,
                        help="imprime alerta quando 'Pronto para treino' >= N")
    args = parser.parse_args()

    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)

    if args.interval is None or args.once:
        counts = _snapshot(client)
        _print_table(counts, trigger_at=args.trigger_at)
        return

    print(f"pollando a cada {args.interval}s. Ctrl-C para parar.")
    while True:
        try:
            counts = _snapshot(client)
            _print_table(counts, trigger_at=args.trigger_at)
        except Exception as exc:
            print(f"[erro no ciclo] {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
