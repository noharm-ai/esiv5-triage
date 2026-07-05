"""
Reclassifica os atendimentos avaliados no CoT usando o classificador ESI v5 com Opus 4.7.

Motivação: as enfermeiras não seguiram exatamente o algoritmo ESI v5 ao registrar os
níveis no banco. Este script aplica o classificador LLM (prompt_classifier_esi_v5.txt)
sobre os dados clínicos do dataset para gerar uma coluna de ground truth alternativa.

Resultado: nova coluna `urgencia_esi_v5_opus47` adicionada ao dataset.csv para os
casos que aparecem em compare_results_cot.csv.

Uso:
    python esi-dataset/reclassify_cot.py [opções]

Exemplos:
    # Ground truth primário (Opus 4.7 via Bedrock, comportamento padrão):
    python esi-dataset/reclassify_cot.py
    # Ground truth cross-family (GPT-5.5 via default.yaml) numa coluna separada:
    python esi-dataset/reclassify_cot.py --classifier-model gpt_5_5 --output-column urgencia_esi_v5_gpt55
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_triage import load_model  # reaproveita o loader genérico do default.yaml

env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path)

DEFAULT_CLASSIFIER_PROMPT_FILE = str(Path(__file__).parent / "prompt_classifier_esi_v5.txt")
DEFAULT_COT_RESULTS = str(Path(__file__).parent / "compare_results_cot.csv")
DEFAULT_DATASET = str(Path(__file__).parent / "dataset.csv")
DEFAULT_CONFIG = str(Path(__file__).parent / "default.yaml")
OPUS_MODEL_ID = "us.anthropic.claude-opus-4-7"
DEFAULT_OUTPUT_COLUMN = "urgencia_esi_v5_opus47"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_classifier_model(model_id: str = OPUS_MODEL_ID):
    """Carrega Opus 4.7 direto via Bedrock (não faz parte do default.yaml).

    Uses ambient AWS credentials: boto3 reads as variáveis AWS_* do ambiente
    (carregadas de backend/.env), permitindo chaves de longa duração.
    """
    from langchain_aws import ChatBedrockConverse
    return ChatBedrockConverse(
        region_name="us-east-1",
        model_id=model_id,
    )


# Contabilização de tokens (Opus GT). Preço USD por 1M tokens (input, output).
OPUS_PRICE_IN_OUT = (15.0, 75.0)  # aprox. Opus 4.x — conferir antes de publicar
_USAGE = {"input": 0, "output": 0, "calls": 0}


def invoke(model, system_prompt: str, user_content: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    for attempt in range(3):
        try:
            response = model.invoke(messages)
            um = getattr(response, "usage_metadata", None)
            if um:
                _USAGE["input"] += int(um.get("input_tokens", 0) or 0)
                _USAGE["output"] += int(um.get("output_tokens", 0) or 0)
                _USAGE["calls"] += 1
            content = response.content
            if isinstance(content, list):
                parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                return "\n".join(parts)
            return content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Formatação do caso clínico como entrada para o classificador
# ---------------------------------------------------------------------------
def format_clinical_summary(row: dict) -> str:
    """
    Formata os dados clínicos do dataset como uma 'conversa de triagem' para
    o classificador ESI v5. Usa os campos disponíveis no dataset.csv.
    """
    parts = []

    chief = row.get("chief_complaint", "").strip()
    if chief:
        parts.append(f"Triagem: Qual é o principal problema que te trouxe aqui hoje?")
        parts.append(f"Paciente: {chief}")

    sintomas = []
    try:
        sintomas = json.loads(row.get("sintomas_nlp") or "[]")
    except Exception:
        pass
    if sintomas:
        parts.append(f"Triagem: Quais outros sintomas você está sentindo?")
        parts.append(f"Paciente: {', '.join(sintomas)}")

    # Contexto demográfico como nota de enfermagem
    ctx = []
    idade = row.get("idade", "")
    sexo = row.get("sexo", "")
    if idade:
        ctx.append(f"Idade: {idade} anos")
    if sexo:
        ctx.append(f"Sexo: {sexo}")
    gestante = row.get("gestante", "")
    if gestante and str(gestante).strip():
        ctx.append(f"Gestante: {gestante}")
    lactante = row.get("lactante", "")
    if lactante and str(lactante).strip():
        ctx.append("Lactante: sim")
    dialise = row.get("dialise", "")
    if dialise and str(dialise).strip():
        ctx.append("Em diálise: sim")

    if ctx:
        parts.append(f"[Dados do prontuário: {'; '.join(ctx)}]")

    return "Conversa de triagem:\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------
def classify(model, classifier_system: str, row: dict) -> tuple[str, str]:
    content = format_clinical_summary(row)
    raw = invoke(model, classifier_system, content)
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
            return data.get("urgencia", "normal"), data.get("resumo", "")
    except Exception:
        pass
    return "normal", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reclassifica casos do CoT com o classificador ESI v5 (ground truth algorítmico)")
    parser.add_argument("--cot-results", default=DEFAULT_COT_RESULTS)
    parser.add_argument("--nr", nargs="*", default=None, help="Lista explícita de nratendimentos a (re)classificar (ignora --cot-results)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML de modelos (usado com --classifier-model)")
    parser.add_argument("--classifier-model", default=None, help="Chave de modelo no default.yaml (ex: gpt_5_5). Habilita cross-family. Tem prioridade sobre --classifier-model-id")
    parser.add_argument("--classifier-model-id", default=OPUS_MODEL_ID, help=f"Model ID Bedrock direto, quando --classifier-model não é usado (default: {OPUS_MODEL_ID})")
    parser.add_argument("--output-column", default=DEFAULT_OUTPUT_COLUMN, help=f"Coluna de saída no dataset (default: {DEFAULT_OUTPUT_COLUMN}). Use uma coluna distinta por modelo de GT, ex: urgencia_esi_v5_gpt55")
    parser.add_argument("--classifier-prompt-file", default=DEFAULT_CLASSIFIER_PROMPT_FILE)
    parser.add_argument("--out", default=DEFAULT_DATASET, help="Arquivo de saída (default: sobrescreve dataset.csv)")
    parser.add_argument("--n", type=int, default=0, help="Amostrar N casos adicionais aleatórios (pula os já classificados)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_column = args.output_column

    classifier_system = Path(args.classifier_prompt_file).read_text(encoding="utf-8").strip()
    print(f"Prompt classificador: {args.classifier_prompt_file}")
    if args.classifier_model:
        print(f"Carregando modelo '{args.classifier_model}' via {args.config} (cross-family)...")
        model = load_model(args.classifier_model, args.config)
    else:
        print(f"Carregando modelo Bedrock {args.classifier_model_id}...")
        model = load_classifier_model(args.classifier_model_id)
    print(f"Coluna de saída: {output_column}")

    # Nratendimentos alvo: lista explícita (--nr) ou os avaliados no CoT
    cot_nrs: set[str] = set()
    if args.nr:
        cot_nrs = {str(x) for x in args.nr}
        print(f"Casos via --nr: {len(cot_nrs)} nratendimentos")
    else:
        with open(args.cot_results, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cot_nrs.add(row["nr"])
        print(f"Casos no CoT: {len(cot_nrs)} nratendimentos únicos")

    # Lê dataset completo
    with open(args.dataset, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        dataset = list(reader)

    if output_column not in fieldnames:
        fieldnames = fieldnames + [output_column]

    # Casos a classificar: CoT + amostra aleatória extra (não classificados ainda)
    target_nrs: set[str] = set(cot_nrs)
    if args.n > 0:
        unclassified = [
            r for r in dataset
            if str(r.get("nratendimento", "")) not in target_nrs
            and not r.get(output_column, "").strip()
        ]
        sample = random.Random(args.seed).sample(unclassified, min(args.n, len(unclassified)))
        extra_nrs = {str(r["nratendimento"]) for r in sample}
        target_nrs |= extra_nrs
        print(f"Amostra adicional: {len(extra_nrs)} casos (--n={args.n})")

    total = 0
    errors = 0

    for row in dataset:
        nr = str(row.get("nratendimento", ""))
        if nr not in target_nrs:
            continue
        if row.get(output_column, "").strip():
            continue  # já classificado

        esi = row.get("esi_level", "?")
        chief = row.get("chief_complaint", "")[:60]
        print(f"  nr={nr}  ESI={esi}  queixa={chief!r} ...", end=" ", flush=True)

        try:
            urgencia, resumo = classify(model, classifier_system, row)
            row[output_column] = urgencia
            print(f"→ {urgencia}")
        except Exception as e:
            print(f"ERRO: {e}")
            row[output_column] = ""
            errors += 1

        total += 1

    # Escreve dataset atualizado
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\nDataset salvo em: {out_path}")
    print(f"Casos reclassificados nesta rodada: {total - errors}/{total}  |  Erros: {errors}")

    if _USAGE["calls"]:
        pin, pout = OPUS_PRICE_IN_OUT
        cost = _USAGE["input"] / 1e6 * pin + _USAGE["output"] / 1e6 * pout
        n = _USAGE["calls"]
        print(f"Tokens Opus: in={_USAGE['input']} out={_USAGE['output']} chamadas={n}")
        print(f"Custo estimado: ${cost:.4f}  |  ${cost/n:.4f}/caso  →  projeção 6144 casos ≈ ${cost/n*6144:.2f}")

    # Sumário geral de todos classificados
    classified = [r for r in dataset if r.get(output_column, "").strip()]
    if classified:
        from collections import Counter
        print(f"\nTotal classificados ({output_column}): {len(classified)}")
        print(f"Distribuição {output_column}:")
        for urg, cnt in sorted(Counter(r[output_column] for r in classified).items()):
            print(f"  {urg}: {cnt}")
        print(f"Cruzamento ESI_enfermeira × {output_column}:")
        cross: dict = {}
        for r in classified:
            key = (r.get("esi_level"), r[output_column])
            cross[key] = cross.get(key, 0) + 1
        for (esi, urg), cnt in sorted(cross.items()):
            print(f"  ESI {esi} → {urg}: {cnt}")


if __name__ == "__main__":
    main()
