"""
Seleção dos casos do benchmark B1 pela classe do rótulo algorítmico Opus (ESI v5).

Como só há ~7 casos ESI-2 registrados pelo enfermeiro, a classe positiva é
definida pelo GROUND TRUTH ALGORÍTMICO (urgencia_esi_v5_opus47 ∈ {urgente,
emergencia}), não pelo nível ESI do enfermeiro. Amostra N positivos + N normais
com seed fixo e salva os nratendimentos (um por linha) para o compare_models.py
consumir via --nr.

Uso:
    python esi-dataset/select_b1_cases.py --n-pos 50 --n-neg 50 --out esi-dataset/b1_cases.txt
"""

import argparse
import csv
import random
import sys
from collections import Counter

POSITIVE = {"urgente", "emergencia"}


def main():
    p = argparse.ArgumentParser(description="Seleciona casos do B1 pela classe do rótulo Opus")
    p.add_argument("--dataset", default="esi-dataset/dataset.csv")
    p.add_argument("--label-column", default="urgencia_esi_v5_opus47")
    p.add_argument("--n-pos", type=int, default=50, help="Casos positivos (urgente+emergencia)")
    p.add_argument("--n-neg", type=int, default=50, help="Casos normais")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="esi-dataset/b1_cases.txt")
    args = p.parse_args()

    pos, neg = [], []
    with open(args.dataset, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get(args.label_column) or "").strip()
            nr = str(row.get("nratendimento", "")).strip()
            if not lab or not nr:
                continue
            if lab in POSITIVE:
                pos.append(nr)
            elif lab == "normal":
                neg.append(nr)

    print(f"Pool rotulado por '{args.label_column}': {len(pos)} positivos, {len(neg)} normais")
    if len(pos) < args.n_pos:
        print(f"  AVISO: só {len(pos)} positivos disponíveis (< {args.n_pos}). Reclassifique mais com Opus.", file=sys.stderr)
    if len(neg) < args.n_neg:
        print(f"  AVISO: só {len(neg)} normais disponíveis (< {args.n_neg}).", file=sys.stderr)

    rng = random.Random(args.seed)
    sel_pos = rng.sample(pos, min(args.n_pos, len(pos)))
    sel_neg = rng.sample(neg, min(args.n_neg, len(neg)))
    selected = sel_pos + sel_neg

    with open(args.out, "w", encoding="utf-8") as f:
        for nr in selected:
            f.write(nr + "\n")

    print(f"Selecionados: {len(sel_pos)} positivos + {len(sel_neg)} normais = {len(selected)} casos")
    print(f"Lista salva em: {args.out}")
    print(f"Uso no B1:  --nr $(cat {args.out} | tr '\\n' ' ')  --label-column {args.label_column}")


if __name__ == "__main__":
    main()
