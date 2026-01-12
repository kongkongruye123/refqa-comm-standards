# -*- coding: utf-8 -*-
"""comm_project/src/07_train_dpo.py

Day7 Step 8: Train DPO adapter starting from base model + SFT adapter.

This implementation is compatible with:
- trl==0.26.x (uses DPOConfig + DPOTrainer(processing_class=...))

Inputs:
- comm_project/data/dpo/train.jsonl
- comm_project/data/dpo/eval.jsonl

Outputs:
- comm_project/outputs/dpo_adapter*/

Notes
- Uses QLoRA (4bit) when available.
- Loads SFT adapter and makes it trainable (is_trainable=True) so DPO updates LoRA weights.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default="qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft_adapter", default=str(PROJECT_ROOT / "outputs" / "sft_adapter"))

    ap.add_argument("--train_file", default=str(PROJECT_ROOT / "data" / "dpo" / "train.jsonl"))
    ap.add_argument("--eval_file", default=str(PROJECT_ROOT / "data" / "dpo" / "eval.jsonl"))

    ap.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "dpo_adapter"))
    ap.add_argument("--log_file", default=str(PROJECT_ROOT / "outputs" / "logs" / "dpo_train.log"))

    ap.add_argument("--tb_logdir", default=str(PROJECT_ROOT / "outputs" / "logs" / "tensorboard" / "dpo"))
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=700)

    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-5)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--no_4bit", action="store_true")

    ap.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing to avoid 'inputs have requires_grad=False' warnings.",
    )

    return ap.parse_args()


def detect_use_4bit(args) -> bool:
    if args.no_4bit:
        return False
    if args.use_4bit:
        return True
    try:
        import bitsandbytes as _bnb  # noqa: F401

        return True
    except Exception:
        return False


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    # TRL 0.26.x
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    use_4bit = detect_use_4bit(args)
    dtype = get_dtype()

    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_short = args.model_name.split("/")[-1]
        run_name = f"{ts}-{model_short}-DPO" + ("-4bit" if use_4bit else "")

    tb_logdir = Path(args.tb_logdir) / run_name
    tb_logdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model_name={args.model_name}")
    print(f"[INFO] sft_adapter={args.sft_adapter}")
    print(f"[INFO] train_file={args.train_file}")
    print(f"[INFO] eval_file={args.eval_file}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] use_4bit={use_4bit} dtype={dtype}")
    print(f"[INFO] tensorboard_logdir={tb_logdir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if use_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Ensure input embeddings can require grads.
    # This avoids torch.utils.checkpoint warning: "None of the inputs have requires_grad=True".
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    if not args.disable_gradient_checkpointing:
        # Reduce VRAM usage
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Load SFT adapter and make it trainable
    sft_adapter = args.sft_adapter
    if sft_adapter and str(sft_adapter).lower() != "none":
        from peft import PeftModel

        try:
            model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        except TypeError:
            model = PeftModel.from_pretrained(model, sft_adapter)
            # Force trainable LoRA params
            for n, p in model.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad_(True)

    # Sanity check
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] trainable_params={trainable} total_params={total}")

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})

    dpo_cfg_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        report_to=["tensorboard"],
        logging_dir=str(tb_logdir),
        run_name=run_name,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    try:
        dpo_args = DPOConfig(**dpo_cfg_kwargs, evaluation_strategy="steps")
    except TypeError:
        dpo_args = DPOConfig(**dpo_cfg_kwargs, eval_strategy="steps")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    metrics = trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Append a simple log line
    try:
        with Path(args.log_file).open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"trained_at={datetime.utcnow().isoformat()}Z\n")
            f.write(f"run_name={run_name}\n")
            f.write(f"tensorboard_logdir={tb_logdir}\n")
            f.write(json.dumps({"train_metrics": getattr(metrics, "metrics", {})}, ensure_ascii=False) + "\n")
    except Exception:
        pass

    print(f"[OK] saved dpo adapter to: {out_dir}")


if __name__ == "__main__":
    main()

