from __future__ import annotations

import argparse
from pathlib import Path

from k3lab.train.grpo_env import MiniRepoGRPOEnv, build_grpo_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA GRPO with environment-owned verifiable MiniRepo reward."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-tool-iterations", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            'Training dependencies are missing. Install with: pip install -e ".[train]"'
        ) from exc

    rows = build_grpo_rows(args.tasks)
    if not rows:
        raise SystemExit("No repo_patch tasks found.")
    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_iterations,
        use_vllm=args.use_vllm,
        bf16=args.bf16,
        logging_steps=1,
        save_steps=max(1, args.steps // 5),
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        environment_factory=MiniRepoGRPOEnv,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
