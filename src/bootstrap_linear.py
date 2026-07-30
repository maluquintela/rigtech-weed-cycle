"""Provisiona os estados/labels do Linear a partir de seed_linear.yaml.

Executar UMA VEZ ao configurar o time. Idempotente: labels que já existam são
puladas; estados são apenas conferidos (a criação de estados custom requer
permissão de admin do time e o Linear não expõe todos pela API pública em
todos os planos, por isso o script se limita a verificar e reportar).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config import load as load_config
from src.linear_client import LinearClient


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Provisiona labels do Linear e verifica estados")
    parser.add_argument(
        "--seed",
        type=Path,
        default=cfg.repo_root / "seed_linear.yaml",
        help="arquivo com estados esperados e labels a criar",
    )
    args = parser.parse_args()

    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)

    print(f"time: {cfg.linear.team_key} -> {client.team_id()}")

    existing_states = client.states()
    print("\nestados no workflow do time:")
    for name in sorted(existing_states):
        print(f"  [ok] {name}")

    print("\nestados esperados por config.yaml:")
    for expected in vars(cfg.linear.states).values():
        marker = "[ok]" if expected in existing_states else "[FALTA]"
        print(f"  {marker} {expected}")

    print("\nlabels:")
    for label in seed.get("labels", []):
        lid = client.create_label(label["name"], color=label.get("color", "#95a5a6"))
        print(f"  [ok] {label['name']} -> {lid}")


if __name__ == "__main__":
    main()
