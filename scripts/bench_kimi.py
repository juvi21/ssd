#!/usr/bin/env python3
import argparse
import random
import time

from ssd import LLM, SamplingParams
from ssd.utils.misc import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Kimi Linear AR/spec/SSD.")
    parser.add_argument("--model", required=True, help="Target Kimi snapshot path.")
    parser.add_argument("--draft", default=None, help="Draft snapshot path. Defaults to target model.")
    parser.add_argument("--mode", choices=["ar", "sync", "async"], default="ar")
    parser.add_argument("--gpus", type=int, default=1, help="Total GPU count.")
    parser.add_argument("--k", type=int, default=4, help="Speculation depth.")
    parser.add_argument("--f", type=int, default=3, help="Async fan out.")
    parser.add_argument("--batch-size", type=int, default=1, help="Max concurrent sequences.")
    parser.add_argument("--num-prompts", type=int, default=8, help="Prompt count.")
    parser.add_argument("--input-len", type=int, default=128, help="Prompt length in tokens.")
    parser.add_argument("--output-len", type=int, default=128, help="Generated tokens per prompt.")
    parser.add_argument("--temp", type=float, default=0.0, help="Target temperature.")
    parser.add_argument("--draft-temp", type=float, default=None, help="Draft temperature override.")
    parser.add_argument("--block-size", type=int, default=256, help="KV block size.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length.")
    parser.add_argument("--gpu-mem-util", type=float, default=0.7, help="Fraction of free GPU memory reserved for KV cache.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eager", action="store_true", help="Disable CUDA graphs.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def make_random_prompts(model_path: str, num_prompts: int, input_len: int, seed: int):
    tokenizer = load_tokenizer(model_path)
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size
    prompts = []
    for _ in range(num_prompts):
        prompts.append([rng.randrange(vocab_size) for _ in range(input_len)])
    return prompts


def main():
    args = parse_args()
    prompts = make_random_prompts(args.model, args.num_prompts, args.input_len, args.seed)
    sampling_params = [
        SamplingParams(
            temperature=args.temp,
            draft_temperature=args.draft_temp,
            ignore_eos=True,
            max_new_tokens=args.output_len,
        )
        for _ in range(len(prompts))
    ]

    llm_kwargs = dict(
        num_gpus=args.gpus,
        max_num_seqs=args.batch_size,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.block_size,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.eager,
        verbose=args.verbose,
        max_steps=args.max_steps,
    )
    if args.mode != "ar":
        llm_kwargs.update(
            speculate=True,
            speculate_k=args.k,
            draft_async=args.mode == "async",
            async_fan_out=args.f,
            draft=args.draft or args.model,
        )

    llm = LLM(args.model, **llm_kwargs)
    start = time.time()
    _, metrics = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - start
    total_generated_tokens = len(prompts) * args.output_len
    print(
        {
            "mode": args.mode,
            "elapsed_s": round(elapsed, 3),
            "generated_tokens": total_generated_tokens,
            "throughput_tok_s": round(total_generated_tokens / elapsed, 2),
            "metrics": metrics,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
