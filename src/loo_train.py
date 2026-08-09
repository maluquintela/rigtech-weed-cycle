"""Leave-one-out por talhão: treina N modelos, cada um com 1 talhão como val.

Diagnóstico: se todos os folds tiverem métricas ruins, o modelo simplesmente
não aprende a tarefa. Se um talhão for muito pior, aquele talhão é OOD
(voo/sensor/cultura diferente). Se todos forem OK menos um, o dataset é OK
mas tem um outlier.

Cada fold:
1. Reseta split: move tudo para live/, depois move val_talhao para golden/
2. Snapshot como versão vN_loo_{talhao}
3. Treina via train_eval.run(...)
4. Registra em history.json (append)

Uso:
    python -m src.loo_train --talhoes Celso01 CelsoSTE2 DoisRiosFlaviano Flaviano01 Giasa \\
        --device 0 --batch 16 --imgsz 1024 --amp --epochs 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import load as load_config
from src.split_golden_by_talhao import split as split_by_talhao
from src.train_eval import run as train_run


def discover_talhoes(live_dir: Path, golden_dir: Path) -> list[str]:
    """Descobre talhões presentes olhando os stems em live/ + golden/."""
    from src.split_golden_by_talhao import _talhao_of
    stems: set[str] = set()
    for d in (live_dir / "images", golden_dir / "images"):
        if d.is_dir():
            stems.update(p.stem for p in d.glob("*.jpg"))
    return sorted({_talhao_of(s) for s in stems})


def run_snapshot(note: str) -> str:
    """Chama snapshot.py como subprocess e devolve o nome da versão criada."""
    r = subprocess.run(
        [sys.executable, "-m", "src.snapshot", "--note", note],
        check=True, capture_output=True, text=True,
    )
    # snapshot imprime "versão criada: .../vN"
    for line in r.stdout.splitlines():
        if "versão criada" in line:
            return line.rsplit("/", 1)[-1].strip()
    raise RuntimeError(f"snapshot não retornou versão. stdout:\n{r.stdout}\nstderr:\n{r.stderr}")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Leave-one-out por talhão")
    parser.add_argument("--talhoes", nargs="*", default=None,
                        help="talhões a iterar (default: todos descobertos)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--skip-existing", action="store_true",
                        help="pula fold se já houver run com tag loo_{talhao} em history.json")
    args = parser.parse_args()

    live = Path(cfg.paths.live_dir)
    golden = Path(cfg.paths.golden_dir)
    talhoes = args.talhoes or discover_talhoes(live, golden)
    print(f"talhões: {talhoes}")

    history_path = Path(cfg.paths.history)
    existing_tags: set[str] = set()
    if args.skip_existing and history_path.exists():
        try:
            existing_tags = {e.get("tag") for e in json.loads(history_path.read_text(encoding="utf-8"))}
        except json.JSONDecodeError:
            pass

    results: list[dict] = []
    for i, val_talhao in enumerate(talhoes, start=1):
        tag = f"loo_{val_talhao}"
        if tag in existing_tags:
            print(f"[skip] fold {i}/{len(talhoes)} {val_talhao}: já tem run com tag={tag}")
            continue

        print(f"\n===== fold {i}/{len(talhoes)}: val={val_talhao} =====")
        counts = split_by_talhao(live, golden, [val_talhao])
        total = sum(counts.values())
        val_n = counts.get(val_talhao, 0)
        print(f"  train={total - val_n}  val={val_n}  ({val_n/total*100:.0f}%)")

        version = run_snapshot(f"loo fold: val={val_talhao}")
        print(f"  snapshot={version}")

        entry = train_run(
            version=version, cfg=cfg,
            tag=tag,
            batch=args.batch, device=args.device, amp=args.amp,
            imgsz=args.imgsz, epochs=args.epochs,
        )
        print(f"  {tag}: {entry['metrics']}")
        results.append(entry)

    print("\n===== RESUMO LOO =====")
    print(f"{'val_talhao':<20}{'mAP50-95':>10}{'mAP50':>10}{'P':>8}{'R':>8}")
    for e in results:
        m = e["metrics"]
        print(f"{e['tag'].replace('loo_',''):<20}"
              f"{m['seg_mAP50-95']:>10.4f}{m['seg_mAP50']:>10.4f}"
              f"{m['seg_precision']:>8.3f}{m['seg_recall']:>8.3f}")


if __name__ == "__main__":
    main()
