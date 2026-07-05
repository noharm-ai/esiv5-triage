"""
Avaliação de cobertura ESI v5 pelo Opus (primary metric).

Para cada conversa gerada (perguntas do M_T + respostas do paciente), o Opus 4.7
avalia QUANTOS requisitos do ESI v5 as perguntas do modelo cobriram. O escore de
urgência (Sonnet/M_C) é irrelevante aqui — mede-se a capacidade do modelo variável
(M_T) de conduzir a anamnese. Vencedor = maior cobertura com menos perguntas.

Entrada: CSV do compare_models.py contendo as colunas 'model', 'nr' e
'conversation_json' (a conversa completa como lista de {role, text}).

Escrita incremental + --resume (robusto a interrupção). Rastreia custo Opus.

Uso:
    python esi-dataset/opus_coverage.py --conversations esi-dataset/compare_b1h_all.csv \
        --out esi-dataset/opus_coverage.csv --resume
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

csv.field_size_limit(10_000_000)  # conversation_json pode ser grande

env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path)

DEFAULT_PROMPT = str(Path(__file__).parent / "prompt_coverage_opus_esi_v5.txt")
OPUS_MODEL_ID = "us.anthropic.claude-opus-4-7"
OPUS_PRICE_IN_OUT = (15.0, 75.0)  # USD por 1M tokens (aprox. Opus 4.x)
_USAGE = {"input": 0, "output": 0, "calls": 0}


def load_opus(model_id=OPUS_MODEL_ID):
    from langchain_aws import ChatBedrockConverse
    return ChatBedrockConverse(region_name="us-east-1", model_id=model_id)


def invoke(model, system_prompt, user_content):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    for attempt in range(3):
        try:
            resp = model.invoke(messages)
            um = getattr(resp, "usage_metadata", None)
            if um:
                _USAGE["input"] += int(um.get("input_tokens", 0) or 0)
                _USAGE["output"] += int(um.get("output_tokens", 0) or 0)
                _USAGE["calls"] += 1
            c = resp.content
            if isinstance(c, list):
                c = "\n".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text")
            return c
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def conversation_to_text(history: list[dict]) -> tuple[str, int]:
    """Formata a conversa e conta as perguntas do assistente (role 'sistema')."""
    lines = []
    n_perguntas = 0
    for msg in history:
        if msg.get("role") == "sistema":
            n_perguntas += 1
            lines.append(f"Assistente: {msg.get('text','')}")
        else:
            lines.append(f"Paciente: {msg.get('text','')}")
    return "Conversa de triagem:\n" + "\n".join(lines), n_perguntas


def evaluate(model, system, history):
    content, n_perg = conversation_to_text(history)
    raw = invoke(model, system, content)
    data = {}
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            data = json.loads(m.group())
    except Exception:
        pass
    return data, n_perg


def main():
    p = argparse.ArgumentParser(description="Cobertura ESI v5 por Opus sobre conversas geradas")
    p.add_argument("--conversations", required=True, help="CSV com colunas model, nr, conversation_json")
    p.add_argument("--prompt-file", default=DEFAULT_PROMPT)
    p.add_argument("--model-id", default=OPUS_MODEL_ID)
    p.add_argument("--out", default="esi-dataset/opus_coverage.csv")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    system = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    print(f"Prompt: {args.prompt_file}\nCarregando Opus {args.model_id}...")
    model = load_opus(args.model_id)

    with open(args.conversations, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Conversas a avaliar: {len(rows)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "nr", "coverage_score", "n_perguntas",
              "criterios_relevantes_total", "n_investigados", "criterios_ausentes"]

    done = set()
    write_header = True
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, newline="", encoding="utf-8") as rf:
            for r in csv.DictReader(rf):
                done.add((r["model"], str(r["nr"])))
        write_header = False
        print(f"Resume: {len(done)} já avaliados serão pulados")

    fout = open(out_path, "w" if write_header else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        w.writeheader(); fout.flush()

    errors = 0
    for i, row in enumerate(rows, 1):
        mk, nr = row.get("model", ""), str(row.get("nr", ""))
        if (mk, nr) in done:
            continue
        try:
            history = json.loads(row.get("conversation_json") or "[]")
        except Exception:
            history = []
        try:
            data, n_perg = evaluate(model, system, history)
            cov = data.get("coverage_score", "")
            w.writerow({
                "model": mk, "nr": nr,
                "coverage_score": cov,
                "n_perguntas": n_perg,
                "criterios_relevantes_total": data.get("criterios_relevantes_total", ""),
                "n_investigados": len(data.get("criterios_investigados", []) or []),
                "criterios_ausentes": json.dumps(data.get("criterios_ausentes", []), ensure_ascii=False),
            })
            fout.flush()
            cs = f"{cov:.0%}" if isinstance(cov, (int, float)) else "?"
            print(f"  [{i}/{len(rows)}] {mk} nr={nr} → cobertura={cs} ({n_perg} perguntas)")
        except Exception as e:
            print(f"  [{i}/{len(rows)}] {mk} nr={nr} ERRO: {e}")
            errors += 1
    fout.close()

    if _USAGE["calls"]:
        pin, pout = OPUS_PRICE_IN_OUT
        cost = _USAGE["input"]/1e6*pin + _USAGE["output"]/1e6*pout
        print(f"\nCusto Opus: ${cost:.2f}  ({_USAGE['calls']} chamadas)  | erros: {errors}")

    # ── Agregação por modelo ──────────────────────────────────────────────────
    cov = defaultdict(list); perg = defaultdict(list)
    with open(out_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                cov[r["model"]].append(float(r["coverage_score"]))
                perg[r["model"]].append(int(r["n_perguntas"]))
            except Exception:
                pass
    print(f"\n{'='*66}\nCOBERTURA ESI v5 (Opus) — vencedor = maior cobertura, menos perguntas\n{'='*66}")
    print(f"{'modelo':<20}{'cobertura':>12}{'perguntas':>12}{'cob/pergunta':>14}")
    print("-"*58)
    agg = []
    for m in cov:
        c = sum(cov[m])/len(cov[m]) if cov[m] else 0
        t = sum(perg[m])/len(perg[m]) if perg[m] else 0
        agg.append((m, c, t, c/t if t else 0))
    for m, c, t, e in sorted(agg, key=lambda x: (-x[1], x[2])):
        print(f"{m:<20}{c:>11.1%}{t:>12.1f}{e:>14.3f}")


if __name__ == "__main__":
    main()
