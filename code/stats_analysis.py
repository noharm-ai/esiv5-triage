"""
Análise estatística pós-experimento (B2 do plano IEEE BIBM).

Lê o CSV detalhado gerado por compare_models.py (colunas: model, nr, esi,
expected, predicted, correct, [coverage_score]) e reporta, por modelo:

- Accuracy com IC 95% bootstrap;
- Macro-F1 com IC 95% bootstrap;
- Recall da classe positiva (default "urgente") com IC de Wilson — a métrica
  de segurança central (sub-triagem). O lower bound de Wilson é o número que
  sustenta (ou não) a afirmação de "zero sub-triagem".

E, opcionalmente, o teste de McNemar pareado entre dois modelos (mesmos casos),
usando o teste binomial exato (sem dependências externas).

Tudo em stdlib puro (sem numpy/scipy), para rodar no venv do esi-dataset.

Uso:
    python esi-dataset/stats_analysis.py --results esi-dataset/compare_results.csv
    python esi-dataset/stats_analysis.py --results compare_results.csv --mcnemar claude_haiku4_5 qwen3-next
    python esi-dataset/stats_analysis.py --results compare_results.csv --positive-label urgente --bootstrap 10000
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict

# Garante saída UTF-8 mesmo em console Windows (cp1252) — evita crash em '→', '±'.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LABELS = ["normal", "urgente", "emergencia"]
Z = 1.959963984540054  # quantil normal para 95%


# ---------------------------------------------------------------------------
# Métricas pontuais
# ---------------------------------------------------------------------------
def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["expected"] == r["predicted"]) / len(rows)


def macro_f1(rows: list[dict]) -> float:
    present = [l for l in LABELS if any(r["expected"] == l for r in rows)]
    if not present:
        return 0.0
    f1s = []
    for label in present:
        tp = sum(1 for r in rows if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in rows if r["expected"] != label and r["predicted"] == label)
        fn = sum(1 for r in rows if r["expected"] == label and r["predicted"] != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(present)


def recall_counts(rows: list[dict], label: str) -> tuple[int, int]:
    """(acertos, total) de recall para uma classe: TP, TP+FN."""
    tp = sum(1 for r in rows if r["expected"] == label and r["predicted"] == label)
    n = sum(1 for r in rows if r["expected"] == label)
    return tp, n


# ---------------------------------------------------------------------------
# Intervalos de confiança
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = Z) -> tuple[float, float, float]:
    """IC de Wilson 95% para uma proporção. Retorna (p_hat, low, high)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (center - margin) / denom, (center + margin) / denom


def bootstrap_ci(rows: list[dict], metric_fn, B: int, rng: random.Random) -> tuple[float, float, float]:
    """IC 95% percentil por bootstrap (reamostragem dos casos com reposição)."""
    point = metric_fn(rows)
    if not rows:
        return point, float("nan"), float("nan")
    n = len(rows)
    samples = []
    for _ in range(B):
        boot = [rows[rng.randrange(n)] for _ in range(n)]
        samples.append(metric_fn(boot))
    samples.sort()
    lo = samples[int(0.025 * B)]
    hi = samples[min(B - 1, int(0.975 * B))]
    return point, lo, hi


# ---------------------------------------------------------------------------
# McNemar pareado (teste binomial exato)
# ---------------------------------------------------------------------------
def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Compara dois modelos nos mesmos casos (pareados por nr)."""
    by_nr_a = {r["nr"]: (r["expected"] == r["predicted"]) for r in rows_a}
    by_nr_b = {r["nr"]: (r["expected"] == r["predicted"]) for r in rows_b}
    common = sorted(set(by_nr_a) & set(by_nr_b))
    b = sum(1 for nr in common if by_nr_a[nr] and not by_nr_b[nr])  # A acerta, B erra
    c = sum(1 for nr in common if not by_nr_a[nr] and by_nr_b[nr])  # A erra, B acerta
    n = b + c
    # p-valor binomial exato bicaudal (H0: p=0.5)
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p_value = min(1.0, 2 * tail)
    return {"n_pairs": len(common), "b": b, "c": c, "discordant": n, "p_value": p_value}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_results(path: str) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_model[row["model"]].append({
                "nr": row.get("nr", ""),
                "expected": row.get("expected", "").strip(),
                "predicted": row.get("predicted", "").strip(),
            })
    return by_model


def fmt_ci(point: float, lo: float, hi: float, pct: bool = True) -> str:
    if math.isnan(point):
        return "n/a"
    if pct:
        return f"{point:6.1%} [{lo:5.1%}, {hi:5.1%}]"
    return f"{point:5.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI + Wilson CI + McNemar sobre resultados de triagem")
    parser.add_argument("--results", default="esi-dataset/compare_results.csv", help="CSV detalhado do compare_models.py")
    parser.add_argument("--positive-label", default="urgente", help="Classe positiva para recall de segurança (default: urgente)")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Iterações de bootstrap (default: 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcnemar", nargs=2, metavar=("MODELO_A", "MODELO_B"), default=None,
                        help="Par de modelos para teste de McNemar pareado")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_model = load_results(args.results)
    if not by_model:
        print("Nenhum resultado encontrado.")
        return

    label = args.positive_label
    print(f"Arquivo: {args.results}  |  bootstrap B={args.bootstrap}  |  classe positiva='{label}'")
    print(f"{'='*96}")
    print(f"{'modelo':<20}{'Accuracy [IC95%]':<26}{'Macro-F1 [IC95%]':<26}{'Recall('+label+') [Wilson95%]':<24}")
    print(f"{'-'*96}")

    for model in sorted(by_model):
        rows = by_model[model]
        acc, acc_lo, acc_hi = bootstrap_ci(rows, accuracy, args.bootstrap, rng)
        mf1, mf1_lo, mf1_hi = bootstrap_ci(rows, macro_f1, args.bootstrap, rng)
        tp, n_pos = recall_counts(rows, label)
        rec, rec_lo, rec_hi = wilson_ci(tp, n_pos)
        rec_str = f"{rec:5.2f} [{rec_lo:.2f},{rec_hi:.2f}] {tp}/{n_pos}" if not math.isnan(rec) else "n/a"
        print(f"{model:<20}{fmt_ci(acc, acc_lo, acc_hi):<26}{fmt_ci(mf1, mf1_lo, mf1_hi, pct=False):<26}{rec_str:<24}")

    if args.mcnemar:
        a, b = args.mcnemar
        if a not in by_model or b not in by_model:
            print(f"\nMcNemar: modelo não encontrado ({a} ou {b}).")
            return
        res = mcnemar(by_model[a], by_model[b])
        print(f"\nMcNemar pareado: {a} vs {b}")
        print(f"  pares comuns: {res['n_pairs']}  |  {a} acerta+{b} erra (b)={res['b']}  |  "
              f"{a} erra+{b} acerta (c)={res['c']}  |  discordantes={res['discordant']}")
        sig = "significativo" if res["p_value"] < 0.05 else "não significativo"
        print(f"  p-valor (binomial exato, bicaudal) = {res['p_value']:.4f}  →  {sig} a 5%")


if __name__ == "__main__":
    main()
