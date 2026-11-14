"""QLoRA fine-tune on the RTX 3090 with Unsloth.

Output is a PEFT LoRA adapter that vLLM serves on top of the AWQ base (`--lora-modules charlie=...`).
"""

from __future__ import annotations

from pathlib import Path

from charliegpt.config import Settings


def train_cuda(settings: Settings, data_dir: Path, out_dir: Path, merge: bool = False) -> Path:
    # Unsloth must be imported before transformers/trl to patch them.
    from unsloth import FastLanguageModel  # isort: skip
    from unsloth.chat_templates import train_on_responses_only

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    m, t = settings.model, settings.train
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=m.train_base, max_seq_length=m.max_seq_length, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=t.lora_r,
        lora_alpha=t.lora_alpha,
        lora_dropout=t.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=settings.dataset.seed,
    )

    def to_text(batch):
        # enable_thinking=False renders the same empty <think></think> block the server emits at inference.
        return {"text": [tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False) for msgs in batch["messages"]]}

    files = {"train": str(data_dir / "train.jsonl")}
    if (data_dir / "valid.jsonl").stat().st_size > 0:
        files["validation"] = str(data_dir / "valid.jsonl")
    ds = load_dataset("json", data_files=files).map(to_text, batched=True, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=SFTConfig(
            output_dir=str(out_dir / "checkpoints"),
            dataset_text_field="text",
            per_device_train_batch_size=t.batch_size,
            gradient_accumulation_steps=t.grad_accum,
            num_train_epochs=t.epochs,
            learning_rate=t.learning_rate,
            warmup_ratio=t.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            bf16=True,
            logging_steps=10,
            eval_strategy="steps" if "validation" in ds else "no",
            eval_steps=t.eval_steps,
            save_steps=t.eval_steps,
            save_total_limit=3,
            seed=settings.dataset.seed,
            report_to="none",
        ),
    )
    # Loss only on your replies, never on other people's messages or the system prompt.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()

    adapter_dir = out_dir / m.adapter_name
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if merge:
        # Full 16-bit merge, e.g. to convert for MLX on the Mac (`mlx_lm.convert --hf-path ... -q`).
        model.save_pretrained_merged(str(out_dir / f"{m.adapter_name}-merged"), tokenizer, save_method="merged_16bit")
    return adapter_dir
