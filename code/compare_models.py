"""
Compara todos os modelos do default.yaml em N casos urgentes + N casos normais.

Separação de papéis:
- Modelo variado: gera as perguntas de anamnese e decide quando encerrar.
- Classificador fixo (--classifier-model, padrão: claude_sonnet4_5): lê o
  histórico completo e classifica a urgência. Isso isola a avaliação da
  qualidade das perguntas, independente da capacidade de classificação de risco.

Os casos são fixos para comparação justa entre os modelos.

Uso:
    python esi-dataset/compare_models.py [opções]

Exemplos:
    python esi-dataset/compare_models.py
    python esi-dataset/compare_models.py --n-urgent 5 --n-normal 5 --prompt-file esi-dataset/prompt_esi_v5.txt
    python esi-dataset/compare_models.py --models claude_haiku4_5 sabiazinho_4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_triage import (
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_CLASSIFIER_PROMPT_FILE,
    DEFAULT_COVERAGE_PROMPT_FILE,
    ESI_TO_URGENCIA,
    TRIAGE_SYSTEM_PROMPT,
    build_patient_system_prompt,
    compute_coverage,
    compute_token_cost,
    load_dataset,
    load_model,
    reset_token_usage,
    run_conversation,
)

DEFAULT_CONFIG = str(Path(__file__).parent / "default.yaml")


def compute_metrics(results: list[dict]) -> dict:
    labels = ["normal", "urgente", "emergencia"]
    per_label = {}
    for label in labels:
        tp = sum(1 for r in results if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in results if r["expected"] != label and r["predicted"] == label)
        fn = sum(1 for r in results if r["expected"] == label and r["predicted"] != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[label] = {"precision": prec, "recall": rec, "f1": f1}

    present = [l for l in labels if any(r["expected"] == l for r in results)]
    macro_f1 = sum(per_label[l]["f1"] for l in present) / len(present) if present else 0.0

    acc = sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    avg_turns = sum(r["n_turns"] for r in results) / len(results) if results else 0.0

    cov_values = [r["coverage_score"] for r in results if "coverage_score" in r]
    avg_coverage = sum(cov_values) / len(cov_values) if cov_values else None

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "avg_turns": avg_turns,
        "avg_coverage": avg_coverage,
        "per_label": per_label,
    }


def main():
    parser = argparse.ArgumentParser(description="Comparação de modelos de triagem")
    parser.add_argument("--dataset", default="esi-dataset/dataset.csv")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n-urgent", type=int, default=5, help="Casos ESI 2 (urgente)")
    parser.add_argument("--n-normal", type=int, default=5, help="Casos ESI 4 (normal)")
    parser.add_argument("--nr", nargs="*", default=None, help="Filtrar por nratendimentos específicos (ignora --n-urgent/--n-normal)")
    parser.add_argument("--label-column", default=None, help="Coluna do dataset a usar como ground truth (ex: urgencia_esi_v5_opus47)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--prompt-file", default=None, help="System prompt alternativo para triagem (.txt)")
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL, help=f"Modelo fixo para classificação de urgência (default: {DEFAULT_CLASSIFIER_MODEL})")
    parser.add_argument("--patient-model", default="claude_haiku4_5", help="Modelo fixo para simular o paciente (default: claude_haiku4_5)")
    parser.add_argument("--classifier-prompt-file", default=DEFAULT_CLASSIFIER_PROMPT_FILE, help=f"System prompt do classificador (.txt) (default: {DEFAULT_CLASSIFIER_PROMPT_FILE})")
    parser.add_argument("--coverage", action="store_true", help="Calcular coverage score ESI v5 por conversa (LLM-judge extra por caso)")
    parser.add_argument("--coverage-prompt-file", default=DEFAULT_COVERAGE_PROMPT_FILE, help=f"System prompt do avaliador de coverage (.txt) (default: {DEFAULT_COVERAGE_PROMPT_FILE})")
    parser.add_argument("--models", nargs="*", default=None, help="Subconjunto de modelos que geram anamnese (default: todos)")
    parser.add_argument("--repeat-prompt", action="store_true", help="Repetir o system prompt no início de cada mensagem de usuário (melhora cobertura em modelos sem reasoning)")
    parser.add_argument("--repeat-classifier", action="store_true", help="Repetir o prompt do classificador no início do conteúdo enviado ao classificador")
    parser.add_argument("--no-severity-hint", action="store_true", help="Não injetar a dica de severidade (derivada do nível ESI) no prompt do paciente — evita leakage do rótulo")
    parser.add_argument("--temperature", type=float, default=None, help="Temperatura aplicada a todos os modelos (default: padrão do provider)")
    parser.add_argument("--out", default="esi-dataset/compare_results.csv")
    parser.add_argument("--resume", action="store_true", help="Retoma: pula (modelo,caso) já presentes no --out e anexa os novos (robusto a interrupção)")
    args = parser.parse_args()

    overrides = {"temperature": args.temperature} if args.temperature is not None else None
    use_severity_hint = not args.no_severity_hint

    triage_system = (
        Path(args.prompt_file).read_text(encoding="utf-8").strip()
        if args.prompt_file
        else TRIAGE_SYSTEM_PROMPT
    )
    classifier_system = Path(args.classifier_prompt_file).read_text(encoding="utf-8").strip()
    coverage_system = Path(args.coverage_prompt_file).read_text(encoding="utf-8").strip() if args.coverage else None

    prompt_label = args.prompt_file or "default"
    cov_label = f"  |  Coverage: {args.coverage_prompt_file}" if args.coverage else ""
    print(f"Prompt triagem: {prompt_label}  |  Prompt classificador: {args.classifier_prompt_file}{cov_label}")

    if args.nr:
        cases = load_dataset(
            args.dataset, esi_levels=None,
            nr_filter=set(args.nr), n=0, seed=args.seed,
            label_column=args.label_column,
        )
        print(f"Casos: {len(cases)} (filtrados por --nr)")
        print(f"  nratendimentos: {[c['nratendimento'] for c in cases]}")
    else:
        urgent_cases = load_dataset(args.dataset, esi_levels=[2], nr_filter=None, n=args.n_urgent, seed=args.seed, label_column=args.label_column)
        normal_cases = load_dataset(args.dataset, esi_levels=[4], nr_filter=None, n=args.n_normal, seed=args.seed, label_column=args.label_column)
        cases = urgent_cases + normal_cases
        print(f"Casos: {len(urgent_cases)} urgentes (ESI 2) + {len(normal_cases)} normais (ESI 4) = {len(cases)} total")
        print(f"  nratendimentos urgentes: {[c['nratendimento'] for c in urgent_cases]}")
        print(f"  nratendimentos normais:  {[c['nratendimento'] for c in normal_cases]}")
    if args.label_column:
        print(f"  Ground truth: coluna '{args.label_column}'")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_keys = list(config.get("lms", {}).keys())
    model_keys = args.models if args.models else all_keys
    print(f"\nClassificador fixo: {args.classifier_model}")
    print(f"Paciente fixo:      {args.patient_model}")
    print(f"Modelos de anamnese ({len(model_keys)}): {', '.join(model_keys)}\n")

    if args.no_severity_hint:
        print("Severity hint:      DESLIGADO (--no-severity-hint)")
    if args.temperature is not None:
        print(f"Temperature:        {args.temperature}")

    try:
        classifier_model = load_model(args.classifier_model, args.config, overrides=overrides)
    except Exception as e:
        print(f"ERRO ao carregar classificador '{args.classifier_model}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        patient_model = load_model(args.patient_model, args.config, overrides=overrides)
    except Exception as e:
        print(f"ERRO ao carregar patient model '{args.patient_model}': {e}", file=sys.stderr)
        sys.exit(1)

    all_results: dict[str, list[dict] | None] = {}
    model_costs: dict[str, float] = {}  # custo total USD por modelo (triage + paciente + classificador + coverage)

    # ── CSV detalhado incremental (grava caso-a-caso; robusto a interrupção) ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_fields = ["model", "nr", "esi", "expected", "predicted", "correct", "n_turns"]
    if args.coverage:
        detail_fields.append("coverage_score")
    detail_fields.append("conversation_json")  # conversa completa (para reanálise/Opus/dataset)

    done: set[tuple[str, str]] = set()
    write_header = True
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, newline="", encoding="utf-8") as rf:
            for row in csv.DictReader(rf):
                done.add((row["model"], str(row["nr"])))
        write_header = False
        print(f"Resume: {len(done)} (modelo,caso) já feitos serão pulados")

    detail_f = open(out_path, "w" if write_header else "a", newline="", encoding="utf-8")
    detail_writer = csv.DictWriter(detail_f, fieldnames=detail_fields)
    if write_header:
        detail_writer.writeheader()
        detail_f.flush()

    for model_key in model_keys:
        print(f"{'─'*60}")
        print(f"Modelo de anamnese: {model_key}")

        try:
            model = load_model(model_key, args.config, overrides=overrides)
        except Exception as e:
            print(f"  ERRO ao carregar: {e}")
            all_results[model_key] = None
            continue

        reset_token_usage()  # custo por modelo (papel triage = este model_key)
        model_results: list[dict] = []
        errors = 0

        for i, case in enumerate(cases, 1):
            nr = case.get("nratendimento", "?")
            esi = int(case["esi_level"])
            expected = case[args.label_column] if args.label_column else ESI_TO_URGENCIA.get(esi, "normal")
            if (model_key, str(nr)) in done:
                continue
            print(f"  [{i:2d}/{len(cases)}] nr={nr} ESI={esi} esp={expected} ...", end=" ", flush=True)

            try:
                history, predicted, _ = run_conversation(
                    triage_model=model,
                    patient_model=patient_model,
                    triage_system=triage_system,
                    patient_system=build_patient_system_prompt(case, severity_hint=use_severity_hint),
                    max_turns=args.max_turns,
                    classifier_model=classifier_model,
                    classifier_system=classifier_system,
                    repeat_prompt=args.repeat_prompt,
                    repeat_classifier=args.repeat_classifier,
                )
            except Exception as e:
                print(f"ERRO: {e}")
                errors += 1
                continue

            correct = predicted == expected
            n_turns = len(history) // 2

            coverage_score = None
            if args.coverage:
                try:
                    cov = compute_coverage(classifier_model, history, coverage_system)
                    coverage_score = cov.get("coverage_score", 0.0)
                except Exception:
                    coverage_score = 0.0

            cov_str = f" cov={coverage_score:.0%}" if coverage_score is not None else ""
            print(f"pred={predicted} {'✓' if correct else '✗'} ({n_turns}t){cov_str}")

            entry = {
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "n_turns": n_turns,
                "nr": nr,
                "esi": esi,
            }
            if coverage_score is not None:
                entry["coverage_score"] = coverage_score
            model_results.append(entry)

            # grava imediatamente (não perde progresso se cair energia/rede)
            detail_row = {"model": model_key, "nr": nr, "esi": esi,
                          "expected": expected, "predicted": predicted,
                          "correct": correct, "n_turns": n_turns}
            if args.coverage:
                detail_row["coverage_score"] = coverage_score if coverage_score is not None else ""
            detail_row["conversation_json"] = json.dumps(history, ensure_ascii=False)
            detail_writer.writerow(detail_row)
            detail_f.flush()

        if errors:
            print(f"  {errors} erros neste modelo")

        role_model = {"triage": model_key, "patient": args.patient_model,
                      "classifier": args.classifier_model, "coverage": args.classifier_model}
        total_cost, _ = compute_token_cost(role_model)
        model_costs[model_key] = total_cost
        if total_cost > 0:
            per_case = total_cost / len(model_results) if model_results else 0.0
            print(f"  custo: ${total_cost:.4f} total | ${per_case:.4f}/caso")

        all_results[model_key] = model_results if model_results else None

    detail_f.close()

    # Re-lê o CSV completo (inclui casos de rodadas anteriores no modo --resume)
    all_results = {}
    with open(out_path, newline="", encoding="utf-8") as rf:
        for row in csv.DictReader(rf):
            e = {
                "expected": row["expected"],
                "predicted": row["predicted"],
                "correct": row["expected"] == row["predicted"],
                "n_turns": int(row["n_turns"]) if row.get("n_turns") else 0,
                "nr": row["nr"],
                "esi": int(row["esi"]) if row.get("esi") else 0,
            }
            if row.get("coverage_score") not in (None, ""):
                e["coverage_score"] = float(row["coverage_score"])
            all_results.setdefault(row["model"], []).append(e)

    # ── Tabela comparativa ──────────────────────────────────────────────────
    cov_header = f"  {'Cov':>6}" if args.coverage else ""
    print(f"\n{'='*80}")
    print("TABELA COMPARATIVA")
    print(f"{'='*80}")
    print(f"{'Modelo':<20} {'Acc':>6}  {'P(urg)':>7} {'R(urg)':>7} {'F1(urg)':>7}  {'MacroF1':>8}  {'Turnos':>7}{cov_header}")
    cov_sep = f"  {'─'*6}" if args.coverage else ""
    print(f"{'─'*20} {'─'*6}  {'─'*7} {'─'*7} {'─'*7}  {'─'*8}  {'─'*7}{cov_sep}")

    summary_rows = []
    for model_key in model_keys:
        results = all_results.get(model_key)
        if not results:
            print(f"{model_key:<20} {'ERRO/SEM DADOS':>40}")
            continue

        m = compute_metrics(results)
        urg = m["per_label"]["urgente"]
        cov_col = f"  {m['avg_coverage']:6.1%}" if args.coverage and m["avg_coverage"] is not None else ""
        print(
            f"{model_key:<20} {m['acc']:6.1%}  "
            f"{urg['precision']:7.2f} {urg['recall']:7.2f} {urg['f1']:7.2f}  "
            f"{m['macro_f1']:8.2f}  {m['avg_turns']:7.1f}{cov_col}"
        )
        row = {
            "model": model_key,
            "acc": round(m["acc"], 4),
            "f1_urgente": round(urg["f1"], 4),
            "precision_urgente": round(urg["precision"], 4),
            "recall_urgente": round(urg["recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "avg_turns": round(m["avg_turns"], 2),
            "n_cases": len(results),
        }
        if args.coverage and m["avg_coverage"] is not None:
            row["avg_coverage_score"] = round(m["avg_coverage"], 4)
        cost = model_costs.get(model_key, 0.0)
        if cost > 0:
            row["cost_usd_total"] = round(cost, 4)
            row["cost_usd_per_case"] = round(cost / len(results), 6) if results else 0.0
        summary_rows.append(row)

    # ── CSV detalhado já foi gravado incrementalmente durante o loop ──────────
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            # União das chaves (colunas como coverage/custo são condicionais por modelo)
            fieldnames: list[str] = []
            for r in summary_rows:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nDetalhado: {out_path}")
    print(f"Sumário:   {summary_path}")


if __name__ == "__main__":
    main()
