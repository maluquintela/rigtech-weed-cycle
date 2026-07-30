# tests/

Vazio de propósito — populado na **Fase 1** com testes para:

- `src/qa_static.py` (dataset sintético gerado no próprio teste)
- `src/snapshot.py` (verificar imutabilidade e correção do changelog)
- `src/xval_flag.py::group_key` (verificar que o padrão real de nomes
  agrupa corretamente e não vaza entre folds)

Rodar com `pytest -q` a partir da raiz do repo.
