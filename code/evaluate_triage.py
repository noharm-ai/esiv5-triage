"""
Avaliação do sistema de triagem usando vinhetas clínicas do dataset ESI.

Para cada caso do dataset, simula uma conversa completa entre o LLM de triagem
(usando o SYSTEM_PROMPT do backend) e um paciente virtual (alimentado pelo
conteúdo clínico do dataset). Ao final, compara a urgencia predita com o
nível ESI real registrado pelo enfermeiro.

Separação de papéis:
- Modelo variado (--model): gera as perguntas de anamnese e decide quando encerrar.
- Modelo classificador fixo (--classifier-model, padrão: claude_sonnet4_5): lê o
  histórico completo da conversa e classifica a urgência. Isso isola a avaliação
  da qualidade das perguntas geradas, independente da capacidade de classificação
  de risco de cada modelo.

Modelos são configurados em default.yaml (LangChain). Suporta Bedrock e MariTalk.

Uso:
    python esi-dataset/evaluate_triage.py [opções]

Exemplos:
    python esi-dataset/evaluate_triage.py --n 10 --esi-level 4
    python esi-dataset/evaluate_triage.py --esi-level 2 --model sabiazinho_4
    python esi-dataset/evaluate_triage.py --n 30 --prompt-file esi-dataset/prompt_esi_v5.txt
    python esi-dataset/evaluate_triage.py --nr XXXXXXXX XXXXXXXX --model sabia_4
    python esi-dataset/evaluate_triage.py --model sabia_4 --n 10 --prompt-file esi-dataset/prompt_esi_v5_cot.txt --label-column urgencia_esi_v5_opus47 --coverage
    python esi-dataset/evaluate_triage.py --model sabia_4 --n 10 --prompt-file esi-dataset/prompt_esi_v5_cot_v2.txt --label-column urgencia_esi_v5_opus47 --coverage --repeat-prompt --patient-model claude_haiku4_5
    python esi-dataset/evaluate_triage.py --nr XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX --model sabia_4 --n 10 --prompt-file esi-dataset/prompt_esi_v5_cot_v2.txt --label-column urgencia_esi_v5_opus47 --coverage --repeat-prompt --patient-model claude_haiku4_5 --classifier-prompt-file esi-dataset/prompt_classifier_esi_v5_v2.txt --repeat-classifier --classifier-model sabia_4

"""

import argparse
import csv
import importlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path)

DEFAULT_MODEL = "claude_haiku4_5"
DEFAULT_CLASSIFIER_MODEL = "claude_sonnet4_5"
DEFAULT_CONFIG = str(Path(__file__).parent / "default.yaml")

ESI_TO_URGENCIA = {
    1: "emergencia",
    2: "urgente",
    3: "normal",
    4: "normal",
    5: "normal",
}

TRIAGE_SYSTEM_PROMPT = """Você é um assistente de pré-triagem hospitalar. Sua função é conduzir uma anamnese curta e objetiva com o paciente antes do atendimento médico presencial.

REGRAS:
1. Faça APENAS UMA pergunta por vez. Seja claro, simples e direto.
2. Use linguagem acessível, evite termos técnicos.
3. Baseie-se nas respostas anteriores para decidir a próxima pergunta.
4. Após 4 a 6 perguntas relevantes, encerre a anamnese.
5. Responda SEMPRE em JSON válido com este formato exato:
   {"pergunta": "texto da pergunta ao paciente", "encerrado": false, "urgencia": "normal"}

   Quando encerrar:
   {"pergunta": "", "encerrado": true, "urgencia": "normal", "resumo": "resumo breve dos sintomas"}

6. O campo "urgencia" deve ser: "normal", "urgente" ou "emergencia"
   - emergencia: dor no peito, falta de ar grave, desmaio, AVC, trauma severo
   - urgente: febre alta, dor intensa, vômitos persistentes, sintomas moderados
   - normal: sintomas leves, consulta de rotina

7. Primeira pergunta sempre: "Qual é o principal problema que te trouxe aqui hoje?"
8. Perguntas relevantes de follow-up baseadas no sintoma relatado:
   - Intensidade (escala 1-10 ou leve/moderada/intensa)
   - Duração (há quanto tempo)
   - Outros sintomas associados
   - Se tem febre, náusea, etc.
   - Uso de medicamentos
9. Seja empático e acolhedor."""

DEFAULT_CLASSIFIER_PROMPT_FILE = str(Path(__file__).parent.parent / "backend" / "app" / "prompts" / "classifier.txt")

DEFAULT_COVERAGE_PROMPT_FILE = str(Path(__file__).parent / "prompt_coverage_esi_v5.txt")


# ---------------------------------------------------------------------------
# Model loading via default.yaml
# ---------------------------------------------------------------------------
def _resolve_env(value):
    """Substitui ${env:VAR} pelo valor da variável de ambiente."""
    if isinstance(value, str):
        def replace(m):
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                raise ValueError(f"Variável de ambiente não definida: {var}")
            return val
        return re.sub(r'\$\{env:([^}]+)\}', replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


def load_model(model_key: str, config_path: str = DEFAULT_CONFIG, overrides: dict | None = None):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lms = config.get("lms", {})
    if model_key not in lms:
        available = ", ".join(lms.keys())
        raise ValueError(f"Modelo '{model_key}' não encontrado em {config_path}.\nDisponíveis: {available}")

    cfg = lms[model_key]
    class_path = cfg["class"]
    args = _resolve_env(cfg.get("args", {}))
    if overrides:
        args = {**args, **overrides}

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**args)


# ---------------------------------------------------------------------------
# Contabilização de tokens (por papel) — preenchida durante invoke()
# ---------------------------------------------------------------------------
TOKEN_USAGE: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})


def reset_token_usage() -> None:
    TOKEN_USAGE.clear()


def _record_usage(usage_key: str | None, response) -> None:
    if not usage_key:
        return
    um = getattr(response, "usage_metadata", None)
    if not um:
        return
    TOKEN_USAGE[usage_key]["input"] += int(um.get("input_tokens", 0) or 0)
    TOKEN_USAGE[usage_key]["output"] += int(um.get("output_tokens", 0) or 0)
    TOKEN_USAGE[usage_key]["calls"] += 1


# Preço em USD por 1 milhão de tokens (input, output). VALORES APROXIMADOS —
# conferir antes de publicar a tabela de custo do paper (B8). Chaves = model_key
# do default.yaml; fallback para 0 quando o modelo não está tabelado.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude_haiku4_5": (1.0, 5.0),
    "claude_sonnet4_5": (3.0, 15.0),
    "qwen3-next": (0.5, 1.5),
    "deepseekv32": (0.3, 1.2),
    "kimik25": (0.6, 2.5),
    "minimax21": (0.5, 2.0),
    "sabia_4": (2.0, 8.0),
    "sabiazinho_4": (0.5, 2.0),
    "sabia_4_thinking": (2.0, 8.0),
    "gpt_5_4": (2.0, 10.0),
    "gpt_5_4_mini": (0.5, 2.0),
    "gpt_5_5": (3.0, 15.0),
    "gpt_nano": (0.1, 0.4),
}


def compute_token_cost(role_to_model: dict[str, str]) -> tuple[float, dict[str, float]]:
    """Retorna (custo_total_usd, custo_por_papel) a partir de TOKEN_USAGE."""
    per_role: dict[str, float] = {}
    total = 0.0
    for role, usage in TOKEN_USAGE.items():
        model_key = role_to_model.get(role, "?")
        pin, pout = PRICING_USD_PER_MTOK.get(model_key, (0.0, 0.0))
        cost = usage["input"] / 1e6 * pin + usage["output"] / 1e6 * pout
        per_role[role] = cost
        total += cost
    return total, per_role


def summarize_token_cost(role_to_model: dict[str, str], n_cases: int | None = None) -> str:
    """Monta um resumo legível de tokens e custo a partir de TOKEN_USAGE.

    role_to_model mapeia o papel (triage/patient/classifier/coverage) para a
    model_key correspondente, para aplicar o preço certo por papel.
    """
    lines = ["", "Tokens e custo estimado (USD):",
             f"  {'papel':<12}{'modelo':<18}{'in':>10}{'out':>10}{'chamadas':>10}{'custo$':>12}"]
    total_cost = 0.0
    for role, usage in TOKEN_USAGE.items():
        model_key = role_to_model.get(role, "?")
        pin, pout = PRICING_USD_PER_MTOK.get(model_key, (0.0, 0.0))
        cost = usage["input"] / 1e6 * pin + usage["output"] / 1e6 * pout
        total_cost += cost
        lines.append(f"  {role:<12}{model_key:<18}{usage['input']:>10}{usage['output']:>10}"
                     f"{usage['calls']:>10}{cost:>12.4f}")
    lines.append(f"  {'TOTAL':<12}{'':<18}{'':>10}{'':>10}{'':>10}{total_cost:>12.4f}")
    if n_cases:
        lines.append(f"  Custo médio por caso: ${total_cost / n_cases:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Invocação LangChain (com retry)
# ---------------------------------------------------------------------------
def invoke(model, system_prompt: str, user_content: str, usage_key: str | None = None) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    for attempt in range(3):
        try:
            response = model.invoke(messages)
            _record_usage(usage_key, response)
            content = response.content
            # Some models (e.g. DeepSeek R1) return a list of content blocks
            if isinstance(content, list):
                parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                return "\n".join(parts)
            return content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Classificação de urgência (modelo fixo)
# ---------------------------------------------------------------------------
def classify_urgency(classifier_model, history: list[dict], classifier_system: str, repeat_classifier: bool = False) -> tuple[str, str]:
    """Classifica urgência a partir do histórico usando o modelo classificador fixo."""
    lines = []
    for msg in history:
        role = "Paciente" if msg["role"] == "paciente" else "Triagem"
        lines.append(f"{role}: {msg['text']}")
    conversation = "Conversa de triagem:\n" + "\n".join(lines)
    content = f"{classifier_system}\n\n{conversation}" if repeat_classifier else conversation
    raw = invoke(classifier_model, classifier_system, content, usage_key="classifier")
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
            return data.get("urgencia", "normal"), data.get("resumo", "")
    except Exception:
        pass
    return "normal", ""


# ---------------------------------------------------------------------------
# Coverage score ESI v5 (LLM-judge de cobertura das perguntas)
# ---------------------------------------------------------------------------
def compute_coverage(classifier_model, history: list[dict], coverage_system: str) -> dict:
    """LLM-judge avalia quais critérios ESI v5 foram investigados pelas perguntas."""
    perguntas = [msg["text"] for msg in history if msg["role"] == "sistema"]
    if not perguntas:
        return {
            "passo_a_investigado": False, "passo_b_investigado": False,
            "criterios_investigados": [], "criterios_ausentes": [],
            "criterios_relevantes_total": 0, "coverage_score": 0.0,
        }
    content = "Perguntas feitas durante a triagem:\n" + "\n".join(f"- {p}" for p in perguntas)
    raw = invoke(classifier_model, coverage_system, content, usage_key="coverage")
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {
        "passo_a_investigado": False, "passo_b_investigado": False,
        "criterios_investigados": [], "criterios_ausentes": [],
        "criterios_relevantes_total": 0, "coverage_score": 0.0,
    }


# ---------------------------------------------------------------------------
# Prompt do paciente simulado
# ---------------------------------------------------------------------------
def build_patient_system_prompt(case: dict, severity_hint: bool = True) -> str:
    sintomas = json.loads(case.get("sintomas_nlp") or "[]")
    sinais = json.loads(case.get("sinais_nlp") or "[]")

    sintomas_str = ", ".join(sintomas) if sintomas else "não informados"
    sinais_str = ", ".join(sinais) if sinais else "não informados"

    gestante_ctx = ""
    if case.get("gestante") and str(case["gestante"]).strip():
        gestante_ctx = f"Você está grávida ({case['gestante']})."

    # A dica de severidade é derivada do nível ESI verdadeiro e pode vazar o
    # rótulo para a conversa. Com severity_hint=False (flag --no-severity-hint),
    # a intensidade emerge apenas dos sintomas/sinais registrados.
    severity_hint_str = ""
    if severity_hint:
        esi = int(case.get("esi_level", 4))
        if esi <= 1:
            severity_hint_str = "Você está se sentindo muito mal, com sintomas graves."
        elif esi == 2:
            severity_hint_str = "Você está se sentindo bastante mal, com desconforto intenso."
        elif esi == 3:
            severity_hint_str = "Você está se sentindo mal, precisando de ajuda, mas estável."

    return f"""Você é um paciente chegando ao pronto-socorro. Responda às perguntas do sistema de triagem como um paciente real, em linguagem simples e coloquial. Não use jargão médico nem mencione diagnósticos pelo nome técnico.

Queixa principal: {case.get('chief_complaint', 'desconforto geral')}
Sintomas que você sente: {sintomas_str}
Sinais registrados (para seu contexto): {sinais_str}
Você tem {case.get('idade', '?')} anos, sexo {case.get('sexo', 'não informado')}.
{gestante_ctx}
{severity_hint_str}

Regras:
- Responda APENAS à pergunta feita, de forma natural e breve (1 a 3 frases).
- Não mencione diagnósticos médicos pelo nome técnico.
- Se não souber algo, diga que não sabe.
- Seja consistente com os sintomas descritos acima."""


# ---------------------------------------------------------------------------
# Conversa simulada
# ---------------------------------------------------------------------------
def build_triage_content(history: list[dict], repeat_system: str | None = None) -> str:
    prefix = f"{repeat_system}\n\n" if repeat_system else ""
    if not history:
        return prefix + "Inicie a anamnese com a primeira pergunta."
    lines = []
    for msg in history:
        role = "Paciente" if msg["role"] == "paciente" else "Sistema"
        lines.append(f"{role}: {msg['text']}")
    return prefix + "Histórico da conversa:\n" + "\n".join(lines) + "\n\nFaça a próxima pergunta ou encerre se já tiver informações suficientes."


def parse_triage_response(raw: str) -> dict:
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"pergunta": raw.strip(), "encerrado": False, "urgencia": "normal"}


def run_conversation(
    triage_model,
    patient_model,
    triage_system: str,
    patient_system: str,
    max_turns: int,
    classifier_model,
    classifier_system: str,
    repeat_prompt: bool = False,
    repeat_classifier: bool = False,
) -> tuple[list[dict], str, str]:
    """Simula a conversa de triagem.

    O triage_model gera as perguntas e decide quando encerrar. A classificação
    final de urgência é feita pelo classifier_model (fixo), que lê o histórico
    completo — isolando a avaliação da qualidade das perguntas geradas.
    """
    history: list[dict] = []

    repeat_system = triage_system if repeat_prompt else None
    for _ in range(max_turns):
        triage_input = build_triage_content(history, repeat_system=repeat_system)
        triage_raw = invoke(triage_model, triage_system, triage_input, usage_key="triage")
        triage_resp = parse_triage_response(triage_raw)

        pergunta = triage_resp.get("pergunta", "").strip()
        encerrado = triage_resp.get("encerrado", False)

        if encerrado or not pergunta:
            urgencia, resumo = classify_urgency(classifier_model, history, classifier_system, repeat_classifier=repeat_classifier)
            return history, urgencia, resumo

        history.append({"role": "sistema", "text": pergunta})

        patient_raw = invoke(patient_model, patient_system, pergunta, usage_key="patient")
        history.append({"role": "paciente", "text": patient_raw.strip()})

    # Força encerramento ao atingir limite de turnos — classifica com modelo fixo
    urgencia, resumo = classify_urgency(classifier_model, history, classifier_system)
    return history, urgencia, resumo


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_dataset(
    path: str,
    esi_levels: list[int] | None,
    nr_filter: set[str] | None,
    n: int,
    seed: int = 42,
    label_column: str | None = None,
) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            level = int(row["esi_level"]) if row.get("esi_level") else None
            # Quando há label_column, exige o rótulo preenchido; sem ele, exige
            # esi_level válido (para o mapeamento ESI_TO_URGENCIA).
            if label_column:
                if not row.get(label_column, "").strip():
                    continue
            elif level is None:
                continue
            # O filtro por nível ESI vale SEMPRE que esi_levels for informado,
            # inclusive junto com label_column (necessário para o split 50+50).
            if esi_levels and level not in esi_levels:
                continue
            if nr_filter and str(row.get("nratendimento", "")) not in nr_filter:
                continue
            rows.append(row)

    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Avalia o sistema de triagem com vinhetas do dataset ESI")
    parser.add_argument("--dataset", default="esi-dataset/dataset.csv")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Arquivo YAML com configuração dos modelos")
    parser.add_argument("--n", type=int, default=50, help="Número de casos (0 = todos)")
    parser.add_argument("--esi-level", type=int, nargs="*", help="Filtrar por nível ESI (ex: --esi-level 2 4)")
    parser.add_argument("--nr", nargs="*", help="Filtrar por nratendimento específico")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chave do modelo que gera as perguntas de anamnese (default: {DEFAULT_MODEL})")
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL, help=f"Chave do modelo fixo para classificação de urgência (default: {DEFAULT_CLASSIFIER_MODEL})")
    parser.add_argument("--patient-model", default=None, help="Chave do modelo de paciente simulado (default: igual a --model)")
    parser.add_argument("--prompt-file", default=None, help="Arquivo .txt com system prompt alternativo para triagem")
    parser.add_argument("--classifier-prompt-file", default=DEFAULT_CLASSIFIER_PROMPT_FILE, help=f"Arquivo .txt com system prompt do classificador (default: {DEFAULT_CLASSIFIER_PROMPT_FILE})")
    parser.add_argument("--coverage", action="store_true", help="Calcular coverage score ESI v5 por conversa (LLM-judge extra por caso)")
    parser.add_argument("--coverage-prompt-file", default=DEFAULT_COVERAGE_PROMPT_FILE, help=f"Arquivo .txt com system prompt do avaliador de coverage (default: {DEFAULT_COVERAGE_PROMPT_FILE})")
    parser.add_argument("--out", default="esi-dataset/results.csv")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column", default=None, help="Coluna do dataset a usar como ground truth (ex: urgencia_esi_v5_opus47)")
    parser.add_argument("--repeat-prompt", action="store_true", help="Repetir o system prompt no início de cada mensagem de usuário (melhora cobertura em modelos sem reasoning)")
    parser.add_argument("--repeat-classifier", action="store_true", help="Repetir o prompt do classificador no início do conteúdo enviado ao classificador")
    parser.add_argument("--no-severity-hint", action="store_true", help="Não injetar a dica de severidade (derivada do nível ESI) no prompt do paciente — evita leakage do rótulo")
    parser.add_argument("--temperature", type=float, default=None, help="Temperatura aplicada a todos os modelos (default: padrão do provider). Use 0 para rodada determinística")
    args = parser.parse_args()

    patient_model_key = args.patient_model or args.model
    overrides = {"temperature": args.temperature} if args.temperature is not None else None
    use_severity_hint = not args.no_severity_hint

    # System prompts
    if args.prompt_file:
        triage_system = Path(args.prompt_file).read_text(encoding="utf-8").strip()
        print(f"  Prompt triagem: {args.prompt_file}")
    else:
        triage_system = TRIAGE_SYSTEM_PROMPT

    classifier_system = Path(args.classifier_prompt_file).read_text(encoding="utf-8").strip()
    print(f"  Prompt classificador: {args.classifier_prompt_file}")

    coverage_system = None
    if args.coverage:
        coverage_system = Path(args.coverage_prompt_file).read_text(encoding="utf-8").strip()
        print(f"  Prompt coverage: {args.coverage_prompt_file}")

    # Carrega modelos do YAML
    print(f"Carregando modelos via {args.config}...")
    triage_model = load_model(args.model, args.config, overrides=overrides)
    classifier_model = load_model(args.classifier_model, args.config, overrides=overrides) if args.classifier_model != args.model else triage_model
    patient_model = load_model(patient_model_key, args.config, overrides=overrides) if patient_model_key != args.model else triage_model
    print(f"  anamnese={args.model}  classificador={args.classifier_model}  paciente={patient_model_key}")
    if args.no_severity_hint:
        print("  severity hint: DESLIGADO (--no-severity-hint)")
    if args.temperature is not None:
        print(f"  temperature: {args.temperature}")

    # Dataset
    cases = load_dataset(
        args.dataset,
        esi_levels=args.esi_level if args.esi_level else None,
        nr_filter=set(args.nr) if args.nr else None,
        n=args.n,
        seed=args.seed,
        label_column=args.label_column,
    )
    if not cases:
        print("Nenhum caso encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"Avaliando {len(cases)} casos  |  max_turns={args.max_turns}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "nratendimento", "esi_level", "esi_color",
        "expected_urgencia", "predicted_urgencia", "correct",
        "n_turns", "chief_complaint", "resumo_final", "conversation_json",
    ]
    if args.coverage:
        fieldnames += ["passo_a_investigado", "passo_b_investigado", "coverage_score", "criterios_ausentes"]

    results = []
    errors = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, case in enumerate(cases, 1):
            nr = case.get("nratendimento", "?")
            esi = int(case["esi_level"]) if case.get("esi_level") else None
            expected = case[args.label_column] if args.label_column else ESI_TO_URGENCIA.get(esi, "normal")

            print(f"  [{i:4d}/{len(cases)}] nr={nr}  ESI={esi}  esperado={expected} ...", end=" ", flush=True)

            patient_system = build_patient_system_prompt(case, severity_hint=use_severity_hint)

            try:
                history, predicted, resumo = run_conversation(
                    triage_model=triage_model,
                    patient_model=patient_model,
                    triage_system=triage_system,
                    patient_system=patient_system,
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

            coverage = {}
            if args.coverage:
                try:
                    coverage = compute_coverage(classifier_model, history, coverage_system)
                except Exception as e:
                    print(f"(coverage ERRO: {e})", end=" ")
                    coverage = {}

            cov_score = coverage.get("coverage_score", None)
            cov_str = f"  cov={cov_score:.0%}" if cov_score is not None else ""
            print(f"predito={predicted}  {'OK' if correct else 'ERRADO'}  ({n_turns} turnos){cov_str}")

            row = {
                "nratendimento": nr,
                "esi_level": esi,
                "esi_color": case.get("esi_color", ""),
                "expected_urgencia": expected,
                "predicted_urgencia": predicted,
                "correct": correct,
                "n_turns": n_turns,
                "chief_complaint": case.get("chief_complaint", ""),
                "resumo_final": resumo,
                "conversation_json": json.dumps(history, ensure_ascii=False),
            }
            if args.coverage:
                row["passo_a_investigado"] = coverage.get("passo_a_investigado", False)
                row["passo_b_investigado"] = coverage.get("passo_b_investigado", False)
                row["coverage_score"] = coverage.get("coverage_score", 0.0)
                row["criterios_ausentes"] = json.dumps(coverage.get("criterios_ausentes", []), ensure_ascii=False)
            writer.writerow(row)
            f.flush()
            results.append(row)

    if not results:
        print("\nNenhum resultado registrado.")
        return

    total = len(results)
    acertos = sum(1 for r in results if r["correct"])
    print(f"\n{'='*60}")
    print(f"Resultado salvo em: {out_path}")
    print(f"Total de casos: {total}  |  Erros: {errors}")
    print(f"Accuracy geral: {acertos}/{total} = {acertos/total*100:.1f}%")
    print(f"Média de turnos: {sum(r['n_turns'] for r in results)/total:.1f}")

    if args.coverage:
        cov_values = [r["coverage_score"] for r in results if "coverage_score" in r]
        if cov_values:
            print(f"Coverage score médio: {sum(cov_values)/len(cov_values):.1%}")
            passo_a = sum(1 for r in results if r.get("passo_a_investigado"))
            passo_b = sum(1 for r in results if r.get("passo_b_investigado"))
            print(f"  Passo A investigado: {passo_a}/{total} ({passo_a/total:.0%})")
            print(f"  Passo B investigado: {passo_b}/{total} ({passo_b/total:.0%})")

    print("\nPor nível ESI:")
    by_esi = defaultdict(list)
    for r in results:
        by_esi[r["esi_level"]].append(r)
    for level in sorted(by_esi):
        group = by_esi[level]
        n_ok = sum(1 for r in group if r["correct"])
        print(f"  ESI {level}: {n_ok}/{len(group)} corretos ({n_ok/len(group)*100:.1f}%)"
              f"  esperado={ESI_TO_URGENCIA.get(level,'?')}")

    print("\nMatriz de confusão (linhas=esperado, colunas=predito):")
    labels = ["normal", "urgente", "emergencia"]
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["expected_urgencia"]][r["predicted_urgencia"]] += 1
    header = f"{'':12s}" + "".join(f"{l:12s}" for l in labels)
    print(f"  {header}")
    for exp in labels:
        row_str = f"  {exp:12s}" + "".join(f"{matrix[exp][pred]:12d}" for pred in labels)
        print(row_str)

    role_to_model = {
        "triage": args.model,
        "patient": patient_model_key,
        "classifier": args.classifier_model,
        "coverage": args.classifier_model,
    }
    print(summarize_token_cost(role_to_model, n_cases=total))


if __name__ == "__main__":
    main()
