"""
Análise da cobertura ESI v5 (Opus) — estatística + figura de trade-off.

Lê esi-dataset/opus_coverage.csv (uma linha por conversa: model, coverage_score,
n_perguntas) e produz:
  1) coverage_summary.csv — por modelo: cobertura média [IC95% bootstrap],
     nº médio de perguntas [IC95%], cobertura/pergunta;
  2) figures/coverage_pareto.(pdf|png) — scatter cobertura × perguntas, com a
     fronteira de Pareto (maior cobertura, menos perguntas) destacada.

Sem chamadas de API. Uso:
    python esi-dataset/coverage_analysis.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
IN = HERE / "opus_coverage.csv"
OUT_CSV = HERE / "coverage_summary.csv"
# pasta do paper (irmã de repo/, sob artigo-triagem/)
FIGDIR = HERE.parent.parent / "IEEE_BIBM_2026___LLM_for_ESI_Triage"

# Nomes de exibição
DISP = {
    "gpt_5_4": "GPT-5.4", "gpt_5_4_mini": "GPT-5.4-mini", "gpt_5_5": "GPT-5.5",
    "sabia_4": "Sabiá-4", "sabia_4_thinking": "Sabiá-4-thinking", "sabiazinho_4": "Sabiázinho-4",
    "claude_haiku4_5": "Haiku 4.5", "qwen3-next": "Qwen3-Next", "deepseekv32": "DeepSeek v3.2",
    "kimik25": "Kimi K2.5", "minimax21": "MiniMax M2.1",
}

# Paleta mínima, colorblind-safe e print-safe (ink + 1 accent)
INK = "#2c3a47"       # pontos dominados
ACCENT = "#0e6e78"    # fronteira de Pareto
WINNER = "#b8402c"    # melhor cobertura/pergunta
GRID = "#dce3e8"


def boot_ci(vals, B=10000, seed=42):
    a = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)
    return a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    cov = defaultdict(list)
    perg = defaultdict(list)
    with open(IN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                cov[r["model"]].append(float(r["coverage_score"]))
                perg[r["model"]].append(int(r["n_perguntas"]))
            except Exception:
                pass

    rows = []
    for m in cov:
        c, clo, chi = boot_ci(cov[m])
        p, plo, phi = boot_ci(perg[m])
        rows.append({
            "model": m, "disp": DISP.get(m, m),
            "coverage": c, "cov_lo": clo, "cov_hi": chi,
            "perguntas": p, "perg_lo": plo, "perg_hi": phi,
            "cov_por_pergunta": c / p if p else 0,
            "n": len(cov[m]),
        })

    # Fronteira de Pareto: não-dominado = ninguém tem cobertura >= E perguntas <=
    for r in rows:
        r["pareto"] = not any(
            (o["coverage"] >= r["coverage"] and o["perguntas"] <= r["perguntas"]
             and (o["coverage"] > r["coverage"] or o["perguntas"] < r["perguntas"]))
            for o in rows
        )
    winner = max(rows, key=lambda r: r["cov_por_pergunta"])["model"]

    # ── CSV resumo ────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: -r["coverage"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "coverage_mean", "coverage_ci95_lo", "coverage_ci95_hi",
                    "perguntas_mean", "perguntas_ci95_lo", "perguntas_ci95_hi",
                    "cov_por_pergunta", "pareto", "n"])
        for r in rows:
            w.writerow([r["model"], f"{r['coverage']:.4f}", f"{r['cov_lo']:.4f}", f"{r['cov_hi']:.4f}",
                        f"{r['perguntas']:.3f}", f"{r['perg_lo']:.3f}", f"{r['perg_hi']:.3f}",
                        f"{r['cov_por_pergunta']:.4f}", int(r["pareto"]), r["n"]])

    print(f"{'modelo':<20}{'cobertura [IC95%]':<26}{'perguntas':>11}{'cob/perg':>10}{'pareto':>8}")
    print("-" * 75)
    for r in rows:
        pf = "★" if r["model"] == winner else ("◆" if r["pareto"] else "")
        print(f"{r['disp']:<20}{r['coverage']:.1%} [{r['cov_lo']:.1%},{r['cov_hi']:.1%}]".ljust(46)
              + f"{r['perguntas']:>6.1f}{r['cov_por_pergunta']:>10.3f}{pf:>8}")
    print(f"\nVencedor (maior cob/pergunta): {DISP.get(winner,winner)}")
    print(f"Fronteira de Pareto: {[DISP.get(r['model']) for r in rows if r['pareto']]}")

    # ── Figura ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#8795a1", "axes.linewidth": 0.8})

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    # linha da fronteira de Pareto (ordenada por perguntas)
    pareto = sorted([r for r in rows if r["pareto"]], key=lambda r: r["perguntas"])
    if len(pareto) >= 2:
        ax.plot([r["perguntas"] for r in pareto], [r["coverage"] * 100 for r in pareto],
                "-", color=ACCENT, lw=1.4, alpha=0.7, zorder=2, label="Pareto frontier")

    # deslocamentos manuais de rótulo (dx, dy, ha) para evitar colisão no aglomerado central
    LOFF = {
        "sabia_4": (-0.09, 0.55, "right"), "kimik25": (0.0, -0.85, "center"),
        "gpt_5_4": (0.09, 0.15, "left"), "gpt_5_5": (-0.09, -0.15, "right"),
        "claude_haiku4_5": (0.09, 0.15, "left"), "minimax21": (0.09, 0.0, "left"),
    }
    for r in rows:
        is_w = r["model"] == winner
        color = WINNER if is_w else (ACCENT if r["pareto"] else INK)
        # barra de erro (IC95% da cobertura)
        ax.errorbar(r["perguntas"], r["coverage"] * 100,
                    yerr=[[(r["coverage"] - r["cov_lo"]) * 100], [(r["cov_hi"] - r["coverage"]) * 100]],
                    fmt="none", ecolor=color, elinewidth=1, capsize=2.5, alpha=0.55, zorder=3)
        ax.scatter(r["perguntas"], r["coverage"] * 100, s=(110 if is_w else 55),
                   marker=("*" if is_w else "o"),
                   facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
        # rótulo direto (com offset customizado quando há aglomeração)
        dx, dy, ha = LOFF.get(r["model"], (0.07, 0.0, "left"))
        ax.annotate(r["disp"], (r["perguntas"], r["coverage"] * 100),
                    xytext=(r["perguntas"] + dx, r["coverage"] * 100 + dy),
                    fontsize=7.6, va="center", ha=ha,
                    color=(color if (r["pareto"] or is_w) else "#3d4b58"),
                    fontweight=("bold" if (r["pareto"] or is_w) else "normal"))

    ax.set_xlabel("Questions per anamnesis (mean)")
    ax.set_ylabel("ESI v5 criteria coverage (%)")
    ax.set_xlim(4.9, 8.2)
    ax.set_ylim(74, 92)
    # "better" region hint (upper-left)
    ax.annotate("better\n(more coverage,\nfewer questions)", xy=(5.15, 90.5), fontsize=6.8,
                color="#5a6b7a", ha="left", va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("Anamnesis quality: ESI v5 coverage vs. effort (n=100 per model)",
                 fontsize=9.5, color=INK, pad=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"coverage_pareto.{ext}", dpi=200, bbox_inches="tight")
    print(f"\nFigura salva em: {FIGDIR/'coverage_pareto.pdf'} (+ .png)")
    print(f"Resumo salvo em: {OUT_CSV}")


if __name__ == "__main__":
    main()
