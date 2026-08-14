"""Provisiona os estados/labels do Linear a partir de seed_linear.yaml.

Executar UMA VEZ ao configurar o time. Idempotente: labels e states que já
existam são pulados. Com ``--create-missing-states``, states ausentes são
criados via API (requer permissão de criar workflow states no time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.config import load as load_config
from src.linear_client import LinearClient


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Provisiona labels/states do Linear")
    parser.add_argument(
        "--seed",
        type=Path,
        default=cfg.repo_root / "seed_linear.yaml",
        help="arquivo com states e labels",
    )
    parser.add_argument(
        "--create-missing-states",
        action="store_true",
        help="cria via API os states listados no seed que não existirem no workflow",
    )
    args = parser.parse_args()

    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)

    print(f"time: {cfg.linear.team_key} -> {client.team_id()}")

    existing_states = client.states()
    print("\nstates no workflow do time:")
    for name in sorted(existing_states):
        print(f"  [ok] {name}")

    print("\nstates esperados por seed:")
    seed_states = seed.get("states", [])
    for state_def in seed_states:
        name = state_def["name"]
        if name in existing_states:
            print(f"  [ok] {name}")
            continue
        if args.create_missing_states:
            sid = client.create_state(
                name=name,
                type_=state_def["type"],
                color=state_def.get("color", "#95a5a6"),
            )
            print(f"  [criado] {name} -> {sid}")
        else:
            print(f"  [FALTA] {name}  (use --create-missing-states)")

    print("\nlabels:")
    for label in seed.get("labels", []):
        lid = client.create_label(label["name"], color=label.get("color", "#95a5a6"))
        print(f"  [ok] {label['name']} -> {lid}")


if __name__ == "__main__":
    main()
