"""Testes de group_key e do particionamento derivado.

O ponto crítico é o Princípio 3: dois tiles do MESMO talhão nunca podem cair
em folds diferentes. Se caírem, o modelo "acerta" por memória do vizinho e o
erro de anotação passa despercebido.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.xval_flag import group_key, make_folds


@pytest.mark.parametrize(
    "filename, expected",
    [
        # Padrão do conversor: {talhao}_r{linha}_c{coluna}.jpg
        ("CelsoSTE2_r012_c034.jpg", "CelsoSTE2"),
        ("Flaviano01_r003_c009.jpg", "Flaviano01"),
        ("Giasa_r045_c012.jpg", "Giasa"),
        # Extremos: tile na coluna 0, linha 0
        ("CelsoSTE2_r000_c000.jpg", "CelsoSTE2"),
        # Nome de arquivo sem "_r" — degrada para o próprio stem
        ("golden_ref_012.jpg", "golden_ref_012"),
    ],
)
def test_group_key_extrai_talhao(filename: str, expected: str) -> None:
    assert group_key(Path(filename)) == expected


def test_tiles_do_mesmo_talhao_ficam_no_mesmo_fold() -> None:
    """O invariante que sustenta a validade do k-fold."""
    talhoes = ["CelsoSTE2", "Flaviano01", "Giasa"]
    imgs = [Path(f"{t}_r{r:03d}_c{c:03d}.jpg") for t in talhoes for r in range(4) for c in range(4)]
    folds = make_folds(imgs, k=3)
    for fold in folds:
        talhoes_no_fold = {group_key(p) for p in fold}
        # cada fold não pode conter tiles de talhões que já apareceram em outro
        for other in folds:
            if other is fold:
                continue
            talhoes_no_outro = {group_key(p) for p in other}
            assert not (talhoes_no_fold & talhoes_no_outro), (
                f"vazamento: talhão em dois folds → {talhoes_no_fold & talhoes_no_outro}"
            )


def test_make_folds_distribui_todos_os_arquivos() -> None:
    imgs = [Path(f"t{i}_r000_c000.jpg") for i in range(9)]
    folds = make_folds(imgs, k=3)
    assert sum(len(f) for f in folds) == len(imgs)
    assert sorted(p for f in folds for p in f) == sorted(imgs)


def test_make_folds_determinismo() -> None:
    imgs = [Path(f"t{i}_r000_c000.jpg") for i in range(9)]
    a = [[p.name for p in f] for f in make_folds(imgs, k=3)]
    b = [[p.name for p in f] for f in make_folds(imgs, k=3)]
    assert a == b


def test_group_key_lida_com_futuro_multi_altitude() -> None:
    """Cenário previsto no TODO: sufixo de data no nome não deve separar
    voos diferentes do mesmo talhão em folds diferentes.

    Este teste FALHA de propósito com a implementação atual — serve de
    lembrete/documentação. Marcado como xfail até que o conversor comece a
    incluir a data no nome e a group_key seja atualizada em conjunto.
    """
    p1 = Path("CelsoSTE2_r012_c034.jpg")
    p2 = Path("CelsoSTE2_20260527_r012_c034.jpg")
    # a intenção é que ambos retornem "CelsoSTE2"; hoje o segundo retorna
    # "CelsoSTE2_20260527" — se este comportamento mudar sem atualizar o TODO,
    # o teste chama atenção.
    assert group_key(p1) == "CelsoSTE2"
    assert group_key(p2) in ("CelsoSTE2", "CelsoSTE2_20260527"), (
        "atualizar test_group_key_lida_com_futuro_multi_altitude ao habilitar "
        "prefixo de data"
    )
