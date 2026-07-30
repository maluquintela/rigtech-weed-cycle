# rigtech-weed-cycle

Ciclo automatizado de melhoria do dataset de segmentação de plantas daninhas
(folha larga vs. folha estreita). Implementação operacional dos documentos
`docs/documento_conceitual.pdf` e `docs/manual_tecnico.pdf`.

## Princípios inegociáveis

1. **A régua não pode se mover.** Avaliação sempre no golden set congelado.
2. **Uma variável por vez.** Enquanto provamos que o dado é o gargalo,
   arquitetura e seed ficam fixas.
3. **O modelo só pode julgar o que não viu.** Detecção de suspeitas via
   predição out-of-fold.
4. **Reprodutibilidade.** Treino lê de tarball imutável, nunca da pasta que
   os anotadores estão editando.

## Estrutura

```
rigtech-weed-cycle/
├── config.yaml                fonte única de verdade
├── requirements.txt
├── seed_linear.yaml           conteúdo inicial do board
├── src/
│   ├── config.py              carregamento e resolução de caminhos
│   ├── qa_static.py           faxina estrutural (CPU, sem modelo)
│   ├── snapshot.py            versionamento imutável do dataset
│   ├── xval_flag.py           detecção de suspeitas via k-fold out-of-fold
│   ├── train_eval.py          treino + avaliação no golden + promoção
│   ├── linear_client.py       cliente GraphQL do Linear
│   ├── bootstrap_linear.py    provisiona labels/verifica estados do board
│   └── run_cycle.py           orquestrador (flag / watch / train)
├── notebooks/
│   └── runner_colab.ipynb     único notebook — só chama scripts
├── docs/
│   ├── golden_set.md
│   ├── manual_tecnico.pdf
│   └── documento_conceitual.pdf
├── tests/                     populado na Fase 1
└── work/                      fora do Git — dados, versões, runs
```

## O que cada script faz (uma frase)

- **`config.py`** — carrega `config.yaml`, resolve caminhos contra a raiz do
  repo, expõe `cfg.class_names` e `cfg.nc`.
- **`qa_static.py`** — verifica defeitos estruturais dos arquivos (label
  órfão, polígono degenerado, classe inválida, conflito de classe, etc.) e
  grava `work/runs/qa_static.csv`.
- **`snapshot.py`** — congela o dataset vivo em `work/versions/vN/` com
  `manifest.json` (sha256), `changelog.json` (diff vs. vN-1) e
  `dataset.tar.gz`.
- **`xval_flag.py`** — divide em k folds por `group_key`, treina k modelos
  nano, prevê out-of-fold, combina `mean_iou` + `gt_perdidos_ratio` +
  `ghost_conf` em um score e grava a fila de revisão.
- **`train_eval.py`** — materializa uma versão em disco local, treina com
  seed fixa, avalia no golden set, aplica `min_delta` e `guard_metrics`,
  grava em `history.json`.
- **`linear_client.py`** — cliente GraphQL mínimo do Linear (auth sem Bearer,
  cache de UUIDs de estados e labels).
- **`bootstrap_linear.py`** — cria as labels e verifica que os estados do
  workflow do time batem com `config.yaml`.
- **`run_cycle.py`** — três subcomandos: `flag` monta a fila e cria cards;
  `watch` reage a lotes "Pronto para treino" disparando snapshot + treino +
  comentário; `train` roda treino avulso.

## Ciclo no dia a dia

```bash
# início do ciclo
python -m src.qa_static
python -m src.run_cycle flag --cycle 3 --dry-run
python -m src.run_cycle flag --cycle 3

# durante o ciclo
python -m src.run_cycle watch --interval 300

# fora do fluxo automático
python -m src.snapshot --note "correções do lote RIG-231"
python -m src.run_cycle train --version v3 --tag baseline
```

## TODOs explícitos antes da Fase 1

- **Dataset:** `paths.live_dir` aponta para `work/live/` — apontar para a
  cópia local do Drive
  (https://drive.google.com/drive/folders/1P3EblQUrd3s4r6JXeHtU1SKyUIX-oTLr).
- **Golden set:** `work/golden/` ainda não existe. Pré-requisito de tudo
  (ver `docs/golden_set.md`).
- **`group_key()` em `src/xval_flag.py`:** implementação padrão junta os dois
  primeiros segmentos do nome do arquivo. Reescrever conforme o padrão real
  de nomes do projeto — se houver frames sequenciais da mesma passagem no
  talhão, vazamento entre folds apaga o sinal silenciosamente.
- **Linear:** `config.yaml → linear.team_key` está como `RIG` — confirmar. A
  chave pessoal vem via `LINEAR_API_KEY` no ambiente.
- **Formato de labels:** o repo assume YOLO-seg canônico
  (`labels/*.txt`, polígonos normalizados em `[0,1]`). Se o formato real for
  outro (COCO, máscaras PNG), criar `src/convert_to_yoloseg.py` sem alterar
  os originais.
- **Testes:** `tests/` populado na Fase 1 com dataset sintético.

## Ordem de implantação

| # | Etapa | O que ela bloqueia se faltar |
|---|-------|------------------------------|
| 0 | Golden set congelado | tudo |
| 1 | QA estático e correção dos erros graves | k-fold gastaria GPU medindo lixo geométrico |
| 2 | Versionamento e baseline fixo | comparação entre ciclos |
| 3 | Validação cruzada | priorização da anotação |
| 4 | Integração com o Linear | centralização (até aqui funciona por CSV) |
| 5 | Watcher (disparo automático) | automação de fato |
| 6 | Comparação ampla de arquiteturas | nada — mas só faz sentido com todo o resto estável |
