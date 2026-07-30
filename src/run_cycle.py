"""Orquestrador do ciclo — três subcomandos: flag, watch, train.

flag  : monta a fila de revisão (QA + validação cruzada) e cria cards no Linear.
watch : fica de pé consultando o Linear; quando um lote atinge "Pronto para
        treino", congela snapshot, treina e comenta o veredito no card.
train : treino avulso em uma versão específica — para baseline e para a
        comparação ampla de arquiteturas (etapa 6 da implantação).
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

from src.config import load as load_config
from src import qa_static, snapshot, train_eval, xval_flag
from src.linear_client import LinearClient


# --- flag -------------------------------------------------------------------


def _load_qa_high_severity(csv_path: Path) -> dict[str, list[str]]:
    """Lê o CSV do QA e devolve ``{imagem: [issue, ...]}`` apenas para severidade alta."""
    if not csv_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["severity"] != qa_static.SEVERITY_HIGH:
                continue
            out.setdefault(row["image"], []).append(f"{row['issue']}: {row['detail']}")
    return out


def cmd_flag(cfg, args) -> None:
    qa_findings = qa_static.run(cfg.paths.live_dir, cfg.nc)
    qa_static.write_report(qa_findings, cfg.paths.qa_report)
    qa_map = _load_qa_high_severity(cfg.paths.qa_report)

    suspicions = xval_flag.run(cfg)
    xval_flag.write_report(suspicions, cfg.paths.suspects_report)

    # imagens com erro estrutural grave vão à frente da fila
    ordered: list[dict] = []
    seen: set[str] = set()
    for img, issues in qa_map.items():
        ordered.append(
            {"image": img, "score": float("inf"), "reason": "qa_alta", "issues": issues,
             "mean_iou": None, "gt_perdidos_ratio": None, "ghost_conf": None, "n_gt": None, "n_pred": None}
        )
        seen.add(img)
    for s in suspicions:
        if s.image in seen:
            continue
        ordered.append({
            "image": s.image, "score": s.score, "reason": "xval",
            "issues": [],
            "mean_iou": s.mean_iou, "gt_perdidos_ratio": s.gt_perdidos_ratio,
            "ghost_conf": s.ghost_conf, "n_gt": s.n_gt, "n_pred": s.n_pred,
        })

    batch = ordered[: cfg.xval.batch_size]
    batch_csv = Path(cfg.paths.runs_dir) / f"lote_ciclo_{args.cycle}.csv"
    batch_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "reason", "score", "mean_iou", "gt_perdidos_ratio",
                         "ghost_conf", "n_gt", "n_pred", "issues"])
        for item in batch:
            writer.writerow([
                item["image"], item["reason"], item["score"], item["mean_iou"],
                item["gt_perdidos_ratio"], item["ghost_conf"], item["n_gt"], item["n_pred"],
                " | ".join(item["issues"]),
            ])

    print(f"lote com {len(batch)} imagens -> {batch_csv}")
    if args.dry_run:
        print("--dry-run: nenhum card criado no Linear.")
        return

    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)
    parent_title = f"Lote ciclo {args.cycle}"
    parent_desc = f"Lote gerado por run_cycle flag. Total: {len(batch)} imagens."
    parent_id = client.create_issue(
        title=parent_title,
        description=parent_desc,
        state=cfg.linear.states.selecionadas,
        labels=[cfg.linear.labels.lote, cfg.linear.labels.ciclo],
    )
    print(f"card-pai criado: {parent_id}")

    for item in batch:
        body_lines = [f"**imagem:** `{item['image']}`", f"**motivo:** {item['reason']}"]
        if item["score"] != float("inf"):
            body_lines += [
                f"score: {item['score']:.4f}",
                f"mean_iou: {item['mean_iou']:.4f}",
                f"gt_perdidos_ratio: {item['gt_perdidos_ratio']:.4f}",
                f"ghost_conf: {item['ghost_conf']:.4f}",
                f"n_gt: {item['n_gt']} · n_pred: {item['n_pred']}",
            ]
        if item["issues"]:
            body_lines.append("**achados QA:**")
            body_lines += [f"- {msg}" for msg in item["issues"]]
        client.create_issue(
            title=f"Revisar {item['image']}",
            description="\n".join(body_lines),
            state=cfg.linear.states.selecionadas,
            parent_id=parent_id,
        )
    print(f"{len(batch)} sub-cards criados.")


# --- watch ------------------------------------------------------------------


def _process_ready_batch(cfg, client: LinearClient, issue: dict) -> None:
    issue_id = issue["id"]
    identifier = issue["identifier"]
    print(f"[{identifier}] processando lote pronto para treino")
    client.move(issue_id, cfg.linear.states.em_treinamento)

    vdir = snapshot.create(cfg.paths.live_dir, cfg.paths.versions_dir, note=f"Lote {identifier}")
    version = vdir.name
    changelog = (vdir / "changelog.json").read_text(encoding="utf-8")
    client.comment(issue_id, f"**snapshot criado:** `{version}`\n\n```json\n{changelog}\n```")

    entry = train_eval.run(version, cfg, tag=identifier)
    metrics_md = "\n".join(f"- **{k}:** {v:.4f}" for k, v in entry["metrics"].items())
    verdict = entry["verdict"]
    promoted = "✔ promovido" if entry["promoted"] else "✘ não promovido"
    client.comment(
        issue_id,
        f"**run:** `{entry['run_id']}`\n**versão:** `{version}`\n\n{metrics_md}\n\n"
        f"**veredito:** {promoted} — {verdict}",
    )
    client.move(issue_id, cfg.linear.states.avaliado)


def cmd_watch(cfg, args) -> None:
    client = LinearClient(team_key=cfg.linear.team_key, endpoint=cfg.linear.endpoint)
    processed: set[str] = set()
    interval = args.interval or cfg.linear.watch.interval_seconds
    print(f"watch ativo — pollando a cada {interval}s. Ctrl-C para encerrar.")
    while True:
        try:
            ready = client.issues_in_state(cfg.linear.states.pronto_para_treino, label=cfg.linear.labels.lote)
            for issue in ready:
                if issue["id"] in processed:
                    continue
                try:
                    _process_ready_batch(cfg, client, issue)
                except Exception:
                    traceback.print_exc()
                processed.add(issue["id"])
        except Exception:
            traceback.print_exc()
        time.sleep(interval)


# --- train ------------------------------------------------------------------


def cmd_train(cfg, args) -> None:
    entry = train_eval.run(args.version, cfg, arch=args.arch, tag=args.tag)
    import json as _json

    print(_json.dumps(entry, indent=2, ensure_ascii=False))


# --- CLI --------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Orquestrador do ciclo de melhoria do dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_flag = sub.add_parser("flag", help="monta a fila de revisão e cria cards no Linear")
    p_flag.add_argument("--cycle", type=int, required=True, help="número do ciclo (aparece no título do lote)")
    p_flag.add_argument("--dry-run", action="store_true", help="não criar cards no Linear")
    p_flag.set_defaults(func=cmd_flag)

    p_watch = sub.add_parser("watch", help="observa o Linear e dispara treino ao ficar 'Pronto para treino'")
    p_watch.add_argument("--interval", type=int, default=None, help="segundos entre consultas")
    p_watch.set_defaults(func=cmd_watch)

    p_train = sub.add_parser("train", help="treino avulso em uma versão específica")
    p_train.add_argument("--version", required=True)
    p_train.add_argument("--arch", default=None)
    p_train.add_argument("--tag", default=None)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(cfg, args)


if __name__ == "__main__":
    main()
