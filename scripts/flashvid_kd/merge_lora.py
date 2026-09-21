#!/usr/bin/env python3
"""
Merge LoRA adapter (saved by run_llava_video_flashvid_lora_kd.sh training) with the base model
and save a single full model for inference.
"""
import argparse
import os
import torch
import transformers
from llava.model import LlavaQwenForCausalLM


def normalize_non_lora_keys(state_dict):
    """Match keys to LlavaQwenForCausalLM.state_dict() (same as llava/model/builder.py)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("base_model."):
            k = k[11:]
        # Some checkpoints contain mixed key styles:
        # - model.layers.*
        # - model.model.mm_projector.*
        # Only strip one "model." when the key actually starts with "model.model."
        if k.startswith("model.model."):
            k = k[6:]
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base LLaVA-Qwen model.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path (e.g. lmms-lab/LLaVA-Video-7B-Qwen2)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter (output_dir of training)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation")
    parser.add_argument("--model_max_length", type=int, default=32768, help="Tokenizer/model effective max length for merged checkpoint")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Loading base model from {args.base_model} ...")
    model = LlavaQwenForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    adapter_path = args.adapter_path.rstrip("/")
    non_lora_path = os.path.join(adapter_path, "non_lora_trainables.bin")
    if os.path.isfile(non_lora_path):
        print(f"Loading non-LoRA trainables onto base (before LoRA merge) from {non_lora_path} ...")
        try:
            non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=True)
        except TypeError:
            non_lora = torch.load(non_lora_path, map_location="cpu")
        non_lora = normalize_non_lora_keys(non_lora)
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        loaded_count = len(non_lora) - len(unexpected)
        print(f"  (non_lora) loaded keys: {loaded_count}/{len(non_lora)}")

        # IMPORTANT: `missing` means keys expected by the full model but not provided by `non_lora`.
        # This is normal because `non_lora_trainables.bin` only stores trainable non-LoRA params.
        if missing:
            print(f"  (non_lora) model-only keys not provided by non_lora: {len(missing)} (usually expected)")

        # `unexpected` indicates real key mismatch on the non_lora side.
        if unexpected:
            unexpected_sorted = sorted(unexpected)
            print(f"  (non_lora) keys from non_lora not found in model: {len(unexpected_sorted)}")
            for k in unexpected_sorted[:50]:
                print("   -", k)
            for kw in ("mm_projector", "vision_resampler", "image_newline", "mm_newline"):
                hits = [k for k in unexpected_sorted if kw in k]
                if hits:
                    print(f"  (non_lora) unexpected contains '{kw}': {len(hits)} (showing first 10)")
                    for k in hits[:10]:
                        print("   *", k)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter path is not a directory: {adapter_path}")

    from peft import PeftModel
    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()
    model.config.tokenizer_model_max_length = args.model_max_length

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    model.config.save_pretrained(args.output_dir)

    print("Loading tokenizer from base model and saving ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.save_pretrained(args.output_dir)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(args.output_dir)

    print("Done. Merged model saved at:", args.output_dir)


if __name__ == "__main__":
    main()
