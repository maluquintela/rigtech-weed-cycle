# Golden Set — a régua congelada

O golden set é o pré-requisito de tudo. Sem uma régua confiável, nenhuma
métrica significa nada e a comparação entre ciclos é ilusória.

## Tamanho

150 a 300 imagens. **Menor e impecável vale mais que grande e duvidoso.**

## Cobertura por eixo

| Eixo             | Faixas a cobrir                                       |
|------------------|-------------------------------------------------------|
| Iluminação       | sol pleno, nublado, sombra parcial, contraluz         |
| Estágio da planta| plântula, intermediário, desenvolvido                 |
| Densidade        | isolada, aglomerada, sobreposta à cultura             |
| Solo e fundo     | seco, úmido, com palhada, com resíduo                 |
| Cultura          | cada cultura em que o modelo vai operar               |
| Balanço de classe| folha larga e estreita em proporção comparável        |

Somar a isso um conjunto de **casos difíceis** escolhidos de propósito — as
fronteiras onde os anotadores discordam. Se a régua só tem caso fácil, ela
mede 85% e o campo entrega 60%.

## Duplo passe

1. Duas pessoas anotam independentemente, **sem ver o trabalho da outra**.
2. Calcula-se IoU entre as duas anotações, instância a instância.
3. IoU ≥ 0,85 → aceita.
4. IoU baixo, ou instância marcada por apenas uma → adjudicação por terceira
   pessoa; casos estruturais discutidos com o time. **Cada desacordo é
   registrado.**

A lista de desacordos é o subproduto mais valioso — é o diagnóstico do guia
de anotação. Cada desacordo recorrente vira uma regra explícita no guia, com
imagem-exemplo do caso-limite.

## Artefatos obrigatórios (junto ao golden set)

- `anotadores.json` — quem anotou o quê, em qual passe.
- `desacordos.csv` — imagem, instância, IoU entre passes, decisão da adjudicação.
- `guia_anotacao.md` — versão do guia vigente no momento do congelamento.

## Regras invioláveis

- **Nunca** entra em treino.
- **Nunca** é editado depois de congelado (correção seria admitir viés — nesse
  caso, congele um novo golden e mantenha o antigo).
- Vive em `work/golden/` (fora do Git), com estrutura `images/` + `labels/`.
- Referenciado por `train_eval.py` apenas como `val:` no `data.yaml`.
