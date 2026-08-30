# rigtech-weed-cycle

Ciclo automatizado de melhoria do dataset de plantas daninhas em ortomosaicos de drone. O modelo funciona como **instrumento fixo de auditoria de rótulos** — mesma arquitetura, mesmos hiperparâmetros, todo ciclo. Ganho de métrica entre ciclos significa **rótulos melhores**, não tuning.

## Filosofia

1. **Instrumento fixo.** DINOv3-ViT-L/16 satélite (congelado) + head Transformer de 2 layers, mesma seed, mesmos hiperparâmetros em todos os ciclos. Só o dataset muda.
2. **Uma métrica de referência.** macro-F1 no split espacial de validação (faixa de 20% da coluna direita de cada fazenda, margem de 1 tile de gap).
3. **Reprodutibilidade.** Cada ciclo gera manifest SHA256 dos geojsons + tarball versionado. Extraindo o tarball, o mesmo modelo é retreinado idêntico.
4. **O modelo julga o que revisar.** Cada rodada exporta um CSV de suspeitas ordenadas: polígonos onde o modelo prediz classe diferente da anotada (`misclass`) ou tem confiança baixa na classe verdadeira (`low_conf`).
5. **Linear como fila de revisão.** Suspeitas viram cards agrupados por site+tipo, com tabela top-N, coords pra abrir no QGIS e instruções de revisão.

## Arquitetura da tarefa

- **Task**: classificação per-crop 224×224, 3 classes:
  - `0` = cultivo (sampleado da plantação)
  - `1` = folha_larga
  - `2` = folha_estreita
- **Mamona** fica fora do escopo atual — geojsons são carregados só pra filtrar amostras negativas (evita cultivo cair em cima de mamona real).
- **Backbone**: `facebook/dinov3-vitl16-pretrain-sat493m`, congelado. Grid 14×14 = 196 patch tokens de 1024 dim por crop.
- **Head treinável**: LayerNorm de entrada + positional encoding aprendível + 2× `TransformerEncoderLayer` (8 heads, GELU, `norm_first=True`, dropout 0.15) + mean pool + LayerNorm + Dropout(0.2) + Linear(1024, 3).
- **Loss**: `CrossEntropyLoss` com peso automático por classe + `label_smoothing=0.05`.
- **Otimização**: AdamW (LR=1e-4, weight_decay=1e-4), warmup 2 + cosine decay 25 épocas, gradient clipping 1.0, early stop patience=7 em macro-F1.
- **Split**: faixa espacial de 20% dentro de cada fazenda (não leave-one-farm-out) — mede generalização espacial dentro do talhão, todas as fazendas contribuem pra treino E val.

## Fluxo do ciclo (dia a dia)

```
1. Abrir notebook no Colab (A100).
2. CICLO = N no topo da célula de config.
3. Run all.
4. Notebook faz:
   - Snapshot do dataset (manifest + tarball versionado + diff vs c{N-1})
   - Extração de features via DINOv3 (cache por site — mtime-based, só recalcula quem mudou)
   - Split treino/val + treino da head (checkpoint por epoch, retoma após disconnect)
   - Relatório de métricas (per-class, matriz de confusão)
   - CSV de suspeitas ordenadas
   - Postagem no Linear (1 card por site+kind)
5. Anotador revisa cards no Linear, corrige geojsons no Drive.
6. Bump CICLO = N+1, roda de novo. Diff automático mostra o que mudou.
7. Compara relatório c{N+1} vs c{N} — macro-F1 subiu? Suspeitas caíram?
```

## Baseline (ciclo 1, 2026-08-25)

Referência pra comparação nos próximos ciclos.

| Métrica | Valor |
|---|---|
| **macro-F1 (val)** | **0.7786** |
| accuracy (val) | 0.8153 |
| F1 cultivo | 0.90 |
| F1 folha_larga | 0.80 |
| F1 folha_estreita | 0.64 |
| Total crops | 14.075 |
| Positivos analisados | 6.575 |
| **Suspeitas (misclass + low_conf)** | **655** (534 + 121) |
| Total polígonos daninha (com mamona) | 6.698 |

Achado principal do c1: **top suspeitas são folha_larga ↔ folha_estreita com `prob_pred > 0.99`** — padrão de troca de arquivo, provável ruído sistemático de anotação. Concentração em DoisRiosFlaviano. Ver `relatorio_dinov3_3class_c1.txt` no Drive.

## Estrutura do repo

```
rigtech-weed-cycle/
├── README.md
├── notebooks/
│   ├── runner_colab_continuacao.ipynb    ← ENTRY POINT (Colab)
│   └── runner_colab.ipynb                pipeline YOLO-seg antigo (deprecated)
├── config.yaml                           config do pipeline YOLO antigo (deprecated)
├── src/                                  módulos do pipeline YOLO antigo (deprecated)
│   ├── convert_to_yoloseg.py             conversor GeoTIFF+GeoJSON -> YOLO
│   ├── qa_static.py                      QA de labels YOLO
│   ├── train_eval.py                     treino Ultralytics YOLO-seg
│   ├── suspects.py                       detecção de suspeitas YOLO
│   ├── linear_client.py                  cliente Linear (reusável)
│   ├── link_suspects_to_linear.py        posta suspeitas YOLO no Linear
│   └── ... (bootstrap_linear, cycle_report, snapshot, etc.)
├── docs/
│   ├── documento_conceitual.pdf
│   ├── manual_tecnico.pdf
│   └── golden_set.md
├── tests/
└── work/                                 fora do Git (dados + artefatos locais)
```

**O único arquivo relevante hoje é `notebooks/runner_colab_continuacao.ipynb`.** Todo o resto (`src/`, `config.yaml`, `notebooks/runner_colab.ipynb`) é do pipeline YOLO-seg anterior — mantido no repo pra histórico e caso a gente precise reaproveitar o `linear_client.py`. Não roda mais em produção.

## Pré-requisitos no Google Drive

```
MyDrive/
├── Datasets/DaninhasTreinoClientes/
│   ├── Giasa/{imagem,daninhas,plantacao}
│   ├── DoisRiosFlaviano/{imagem,daninhas,plantacao}
│   ├── Flaviano01/{imagem,daninhas,plantacao}
│   ├── CelsoSTE2/{imagem,daninhas,plantacao}
│   └── Celso01/{imagem,daninhas,plantacao}
└── (artefatos criados automaticamente pelo notebook)
    ├── modelo_dinov3_3class_c{N}.pt
    ├── relatorio_dinov3_3class_c{N}.txt
    ├── suspeitas_c{N}.csv
    ├── dinov3_3class_checkpoint/{site}_{hash}.joblib
    └── rigtech_ciclos/
        ├── dataset_manifest_c{N}.json
        ├── dataset_geojsons_c{N}.tar.gz
        └── train_ckpt_c{N}.pt (limpo ao fim do treino)
```

## Secrets necessários no Colab

Ícone chave à esquerda no Colab, com "Notebook access" ligado:

- `HF_TOKEN` — token do Hugging Face com permissão pra gated repos. Aceitar termos em https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m antes de gerar o token.
- `LINEAR_API_KEY` (opcional) — chave pessoal do Linear pra postar suspeitas como cards no time RIG. Criar em `linear.app/rigtech/settings/account/security` → **API keys** → **New API key**.

## Como rodar

1. Abrir o notebook no Colab: https://colab.research.google.com/github/maluquintela/rigtech-weed-cycle/blob/main/notebooks/runner_colab_continuacao.ipynb
2. **File → Save a copy in Drive** (senão não salva alterações).
3. **Runtime → Change runtime type → A100 GPU** (T4 funciona mas é mais lento).
4. **Runtime → Restart session**.
5. Verificar `CICLO = N` no topo da célula de config.
6. **Runtime → Run all**.

Tempo esperado no primeiro ciclo (sem cache): ~30-45 min pra extração de features + ~15-25 min pra treino. Ciclos subsequentes reaproveitam cache: só sites com geojsons alterados recalculam features.

## Meta pro próximo ciclo

Corrigindo ~20-30 suspeitas top do c1 (foco em DoisRiosFlaviano folha_larga↔folha_estreita):

- macro-F1 > 0.79
- F1 folha_estreita > 0.68
- Suspeitas totais < 550

Se essas metas forem batidas com correções manuais no dataset, o método está funcionando. Se não subir apesar das correções, o teto da tarefa não é rótulo — é GSD/resolução, e a decisão vira sobre pipeline de captura em campo.

## Repos

- **Público (espelho pessoal)**: https://github.com/maluquintela/rigtech-weed-cycle — usado pelo Colab pra abrir o notebook sem autenticação.
- **Privado (org)**: https://github.com/Rigtech-Solutions/rigtech-weed-cycle — canonical.

Ambos sincronizados. Commits vão pros dois com `git push origin main && git push personal main`.
