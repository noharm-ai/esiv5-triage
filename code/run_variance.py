"""
Experimento de variância run-a-run (B3 do plano IEEE BIBM).

Roda compare_models.py N vezes sobre os MESMOS casos (seed fixo => mesma
amostragem) e agrega, por modelo, média ± desvio-padrão das métricas. A
variação observada entre rodadas vem da estocasticidade dos modelos
(temperatura padrão do provider), não da amostragem de casos.

Pensado para o subconjunto de modelos SEM raciocínio, onde a variância e o
repeat-prompt mais importam.

Os argumentos após `--` são repassados literalmente ao compare_models.py.

Uso:
    python esi-dataset/run_variance.py --runs 5 --out-dir esi-dataset/variance -- \
        --n-urgent 50 --n-normal 50 \
        --models sabia_4 sabiazinho_4 minimax21 kimik25 deepseekv32 \
        --patient-model claude_sonnet4_5 --no-severity-hint --coverage \
        --prompt-file esi-dataset/prompt_esi_v5_cot_v2.txt \
        --classifier-prompt-file esi-dataset/prompt_classifier_esi_v5_v2.txt \
        --label-column urgencia_esi_v5_opus47
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Garante saída UTF-8 mesmo em console Windows (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

COMPARE = str(Path(__file__).parent / "compare_models.py")
# Métricas numéricas do *_summary.csv agregadas entre rodadas
METRICS = ["acc", "macro_f1", "recall_urgente", "f1_urgente", "avg_turns", "avg_coverage_score"]


def run_once(forward: list[str], out_csv: Path) -> Path:
    cmd = [sys.executable, COMPARE, *forward, "--out", str(out_csv)]
    print(f"\n$ {' '.join(cmd)}")
    env = {**os.environ, "PYTHONUTF8": "1"}  # saída UTF-8 no subprocesso (Windows)
    subprocess.run(cmd, check=True, env=env)
    return out_csv.with_name(out_csv.stem + "_summary.csv")


def read_summary(summary_path: Path) -> dict[str, dict]:
    by_model: dict[str, dict] = {}
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_model[row["model"]] = row
    return by_model


def main():
    parser = argparse.ArgumentParser(description="Variância run-a-run de compare_models.py")
    parser.add_argument("--runs", type=int, default=3, help="Número de repetições (3–5)")
    parser.add_argument("--out-dir", default="esi-dataset/variance", help="Diretório para os CSVs por rodada")
    parser.add_argument("forward", nargs=argparse.REMAINDER,
                        help="Args após -- repassados ao compare_models.py")
    args = parser.parse_args()

    forward = args.forward
    if forward and forward[0] == "--":
        forward = forward[1:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # metric_values[model][metric] = lista de valores entre rodadas
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i in range(1, args.runs + 1):
        print(f"\n{'='*70}\nRODADA {i}/{args.runs}\n{'='*70}")
        summary = run_once(forward, out_dir / f"run_{i}.csv")
        per_model = read_summary(summary)
        for model, row in per_model.items():
            for metric in METRICS:
                val = row.get(metric, "")
                if val not in ("", None):
                    try:
                        metric_values[model][metric].append(float(val))
                    except ValueError:
                        pass

    # ── Agregação ────────────────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print(f"VARIÂNCIA ENTRE {args.runs} RODADAS (média ± desvio-padrão)")
    print(f"{'='*88}")
    header = f"{'modelo':<20}" + "".join(f"{m:>16}" for m in ["acc", "macro_f1", "recall_urg", "coverage"])
    print(header)
    print("-" * len(header))

    agg_rows = []
    for model in sorted(metric_values):
        mv = metric_values[model]
        def cell(metric: str) -> str:
            vals = mv.get(metric, [])
            if not vals:
                return f"{'n/a':>16}"
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mean:>8.3f}±{sd:<6.3f}"
        print(f"{model:<20}" + cell("acc") + cell("macro_f1") + cell("recall_urgente") + cell("avg_coverage_score"))

        row = {"model": model, "n_runs": args.runs}
        for metric in METRICS:
            vals = mv.get(metric, [])
            if vals:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 4)
                row[f"{metric}_sd"] = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        agg_rows.append(row)

    agg_path = out_dir / "variance_summary.csv"
    if agg_rows:
        fieldnames: list[str] = []
        for r in agg_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_rows)
    print(f"\nAgregado salvo em: {agg_path}")


if __name__ == "__main__":
    main()
