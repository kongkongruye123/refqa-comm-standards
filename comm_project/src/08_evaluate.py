# -*- coding: utf-8 -*-
"""comm_project/src/08_evaluate.py

Evaluate Base vs SFT vs DPO on a fixed-evidence evaluation set.

Inputs:
- comm_project/data/eval/eval_set.jsonl
- comm_project/data/corpus/evidence_corpus.jsonl

Models:
- base: qwen/Qwen2.5-1.5B-Instruct
- sft : base + LoRA adapter at comm_project/outputs/sft_adapter
- dpo : base + LoRA adapter at comm_project/outputs/dpo_adapter

Outputs:
- reports/eval_<model>.json  (per-sample results)

Metrics (from docs/comm_llm_plan.md 11.2):
- json_valid_rate
- schema_pass_rate
- citation_exact_match_rate
- grounded_answer_rate (simple n-gram coverage proxy)
- refusal_correct_rate

Notes:
- This script does NOT do retrieval. Evidence is provided by eval_set.evidence_chunk_ids.
- It enforces the unified prompt format and expects strict JSON outputs.
- IMPORTANT: For instruct models like Qwen, we use tokenizer.apply_chat_template
  to ensure system/user roles are respected. If not available, we fall back to
  plain concatenation.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Local imports (make it work when run from repo root)
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.prompts import SYSTEM_PROMPT, format_user_prompt, validate_json_output  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
REPORTS_DIR = PROJECT_ROOT.parent / "reports"

DEFAULT_BASE_MODEL = "qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft_adapter"
DEFAULT_DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo_adapter"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_evidence_map(corpus_path: Path):
    evidence = {}
    for row in read_jsonl(corpus_path):
        evidence[row["chunk_id"]] = row
    return evidence


def build_evidence_text(evidence_chunks):
    blocks = []
    for ch in evidence_chunks:
        blocks.append(
            f"[chunk_id={ch['chunk_id']} doc_id={ch['doc_id']} section={ch['section']}]\n{ch['text']}"
        )
    return "\n\n".join(blocks)


def extract_ngrams(text: str, n: int = 3):
    toks = re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def grounded_coverage(answer: str, evidence_text: str, n: int = 3):
    a = extract_ngrams(answer, n=n)
    if not a:
        return 0.0
    e = extract_ngrams(evidence_text, n=n)
    if not e:
        return 0.0
    hit = sum(1 for g in a if g in e)
    return hit / max(1, len(a))


@dataclass
class ModelBundle:
    model: object
    tokenizer: object


def load_model_bundle(model_name: str, adapter_path, use_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    kwargs["torch_dtype"] = dtype
    if use_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return ModelBundle(model=model, tokenizer=tok)


def build_model_inputs(bundle: ModelBundle, system_text: str, user_text: str):
    """Build model inputs using chat template when available."""
    tok = bundle.tokenizer

    messages = [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]

    if hasattr(tok, "apply_chat_template"):
        try:
            input_ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return {"input_ids": input_ids}
        except TypeError:
            # Older signature
            input_ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            return {"input_ids": input_ids}
        except Exception:
            # fall back below
            pass

    # Fallback: plain concatenation
    full_prompt = system_text.strip() + "\n" + user_text.strip() + "\n"
    return tok(full_prompt, return_tensors="pt")


def generate_json(bundle: ModelBundle, system_text: str, user_text: str, max_new_tokens: int = 512):
    tok = bundle.tokenizer
    model = bundle.model

    inputs = build_model_inputs(bundle, system_text, user_text)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    # Heuristic: take substring from first '{' to last '}'
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    return text.strip()


def evaluate_one(sample: dict, evidence_map: dict, bundle: ModelBundle, max_new_tokens: int):
    ev_ids = sample["evidence_chunk_ids"]
    chunks = []
    missing = []
    for cid in ev_ids:
        if cid in evidence_map:
            chunks.append(evidence_map[cid])
        else:
            missing.append(cid)

    evidence_text = build_evidence_text(chunks)
    user_prompt = format_user_prompt(sample["question"], evidence_text=evidence_text)

    raw = generate_json(bundle, SYSTEM_PROMPT, user_prompt, max_new_tokens=max_new_tokens)

    result = {
        "id": sample["id"],
        "type": sample["type"],
        "question": sample["question"],
        "evidence_chunk_ids": ev_ids,
        "missing_evidence": missing,
        "raw_output": raw,
    }

    # json_valid
    try:
        obj = json.loads(raw)
        result["json_valid"] = True
        result["parsed"] = obj
    except Exception as e:
        result["json_valid"] = False
        result["parse_error"] = str(e)
        return result

    # schema
    ok, msg = validate_json_output(obj)
    result["schema_pass"] = bool(ok)
    result["schema_msg"] = msg

    # citation exact match
    cit_ok = 0
    cit_total = 0
    if isinstance(obj.get("citations"), list):
        for c in obj["citations"]:
            if not isinstance(c, dict):
                continue
            cit_total += 1
            q = c.get("quote")
            cid = c.get("chunk_id")
            if isinstance(q, str) and isinstance(cid, str) and cid in evidence_map:
                if q and q in evidence_map[cid].get("text", ""):
                    cit_ok += 1
    result["citation_total"] = cit_total
    result["citation_exact_match"] = cit_ok

    # grounded coverage (proxy)
    ans = obj.get("answer") if isinstance(obj.get("answer"), str) else ""
    cov = grounded_coverage(ans, evidence_text, n=3)
    result["grounded_coverage"] = cov

    # refusal correctness
    if sample["type"] == "unanswerable":
        reason = obj.get("cannot_answer_reason")
        citations = obj.get("citations")
        refusal_ok = bool(isinstance(reason, str) and reason.strip()) and (not citations)
        result["refusal_correct"] = refusal_ok
    else:
        result["refusal_correct"] = None

    return result


def aggregate(results: list):
    total = len(results)
    json_valid = sum(1 for r in results if r.get("json_valid"))
    schema_pass = sum(1 for r in results if r.get("schema_pass"))

    cit_total = sum(r.get("citation_total", 0) for r in results if r.get("json_valid"))
    cit_ok = sum(r.get("citation_exact_match", 0) for r in results if r.get("json_valid"))

    grounded = sum(1 for r in results if r.get("json_valid") and r.get("grounded_coverage", 0) >= 0.2)

    unans = [r for r in results if r.get("type") == "unanswerable" and r.get("json_valid")]
    refusal_ok = sum(1 for r in unans if r.get("refusal_correct"))

    return {
        "total": total,
        "json_valid_rate": json_valid / total if total else 0,
        "schema_pass_rate": schema_pass / total if total else 0,
        "citation_exact_match_rate": cit_ok / cit_total if cit_total else 0,
        "grounded_answer_rate": grounded / total if total else 0,
        "refusal_correct_rate": refusal_ok / len(unans) if unans else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "sft", "dpo"], required=True)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter_path", default=None, help="Override adapter path")
    ap.add_argument("--use_4bit", action="store_true", help="Load model in 4bit (if supported)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of eval samples (0=all)")
    args = ap.parse_args()

    if not EVAL_PATH.exists():
        raise RuntimeError(f"Missing eval set: {EVAL_PATH}. Run 03_make_eval.py first.")
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Missing evidence corpus: {CORPUS_PATH}. Run 02_build_corpus.py first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = None
    if args.model == "sft":
        adapter = Path(args.adapter_path) if args.adapter_path else DEFAULT_SFT_ADAPTER
    elif args.model == "dpo":
        adapter = Path(args.adapter_path) if args.adapter_path else DEFAULT_DPO_ADAPTER

    if adapter is not None and not adapter.exists():
        raise RuntimeError(f"Adapter path not found: {adapter}")

    evidence_map = load_evidence_map(CORPUS_PATH)
    eval_samples = list(read_jsonl(EVAL_PATH))
    if args.limit and args.limit > 0:
        eval_samples = eval_samples[: args.limit]

    print(f"[INFO] eval_samples={len(eval_samples)}")
    print(f"[INFO] model={args.model} base_model={args.base_model} adapter={adapter}")

    bundle = load_model_bundle(args.base_model, adapter, use_4bit=args.use_4bit)

    results = []
    for s in eval_samples:
        r = evaluate_one(s, evidence_map, bundle, max_new_tokens=args.max_new_tokens)
        results.append(r)
        if len(results) % 10 == 0:
            print(f"  done {len(results)}/{len(eval_samples)}")

    summary = aggregate(results)
    out_path = REPORTS_DIR / f"eval_{args.model}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote report: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
