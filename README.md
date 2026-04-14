# GPU Mode Paris Hackathon 2026 — 3rd Place Solution

**Track 1: LLM Pre-training** | **Team: ricy** | **Score: 4.08 val_loss**

Our team placed **3rd** at the [GPU Mode Paris Hackathon 2026](https://github.com/gpu-mode/paris-hackathon-2026-training), organized by [Sesterce](https://www.sesterce.com/) on NVIDIA B300 GPUs.

## The Challenge

Train a GPT-2 scale language model from scratch on 32× NVIDIA B300 GPUs in **10 minutes**. The only metric: **validation loss** (lower is better).

Constraints:
- Fixed dataset (~49B tokens, tokenized with a 32K vocabulary)
- Fixed time budget: 10 minutes of GPU time
- Must expose a standard `get_model()` / `forward()` contract
- Organizers run your `train.py` with their own arguments

## Our Approach

Starting from a basic GPT-2 baseline, we iterated through **16 experiments** over the course of the day, improving validation loss from **~9.5 to 3.92** (2 GPUs) and **3.62** (32 GPUs test run).

### Architecture (119M params)

| Technique | Why |
|-----------|-----|
| **RMSNorm** | Faster than LayerNorm, no mean subtraction |
| **RoPE** | Better position encoding, saves parameters vs learned embeddings |
| **ReLU²** | 2 matmuls instead of 3 (vs SwiGLU), same param count, ~7% faster |
| **QK Norm** | Stabilizes attention when using aggressive optimizers like Muon |
| **Logit soft-capping** | `30 * tanh(logits/30)` prevents logit explosion (Gemma-style) |
| **Post-embedding norm** | RMSNorm after token embedding for stability |
| **Value Embeddings** | Learned positional bias on V (ResFormer) |
| **x0 Residual Connection** | Skip from initial embedding to every layer (learnable lambda, init=0) |
| **Per-layer Lambdas** | Learnable scaling for attn and mlp per layer |
| **U-Net Skip Connections** | Symmetric layer connections (0↔11, 1↔10, ...) with learned scalars |
| **Zero-init output projections** | Near-identity at init via residual (muP-like) |
| **Weight tying** | Shared embedding and output projection weights |

### Training Optimizations

| Technique | Impact |
|-----------|--------|
| **Muon optimizer** | Newton-Schulz orthogonalization on 2D weights, AdamW on rest. Faster convergence. |
| **WSD schedule** | Warmup → Stable → Decay. Keeps max LR longer than cosine. |
| **torch.compile** | ~20-40% throughput gain via kernel fusion |
| **gc.disable()** | Eliminates unpredictable ~100ms GC pauses |
| **Data prefetch** | Background CPU thread loads next batch while GPU computes |
| **Muon 3 iters** | 3 Newton-Schulz iterations (not 5) — sufficient convergence, ~5% faster |
| **Batch 32 + accum 2** | Same tokens/step as batch 16 + accum 4, fewer micro-steps = faster |

### Key Results (2 GPUs)

| Version | val_loss | ms/step | Steps in 10 min | Key change |
|---------|----------|---------|-----------------|------------|
| v1 (baseline + modern arch) | 4.15 | 300 | 1832 | SwiGLU, RMSNorm, RoPE, Muon |
| v2 (+ quick wins) | 4.07 | 250 | 1872 | QK Norm, logit capping, GC off, WSD |
| v3 (350M model) | 4.28 | 450 | 1177 | Too big → too slow → regression |
| v5 (throughput) | 4.09 | 237 | 1964 | Batch 32, Muon 3 iters, prefetch |
| v6 (+ ReLU², U-Net) | 4.01 | 220 | 2000 | ReLU², U-Net skips, zero-init |
| **v7 (final)** | **3.92** | **230** | **2500** | Optimal max_steps for WSD decay |

## Key Lessons

1. **Throughput > model size** on a fixed time budget. A 119M model at 230ms/step beats a 350M model at 450ms/step because it does 2× more weight updates.

2. **The WSD learning rate schedule is critical.** If `max_steps` is set too high, the decay phase never activates and you lose ~0.2 val_loss. Always ensure the decay starts within your time budget.

3. **FP8 hurts on small models.** We tried `torchao` FP8 training — it was 60% slower (350ms vs 230ms/step). The overhead of scaling factor computation outweighs matmul gains on 768×3072 matrices. Only worth it for 500M+ params.

4. **Test before you submit.** Our best test run hit 3.62 on 32 GPUs, but the final submission scored 4.08 — likely because the WSD decay didn't fully activate under the organizers' arguments.

## Files

```
model.py   — GPT architecture (119M params)
train.py   — Training loop with Muon + AdamW, WSD schedule, DDP, prefetch
```

## Usage

```bash
# Single GPU
python train.py --data_dir /path/to/data --max_steps 2500

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 train.py --data_dir /path/to/data --max_steps 2500

# 32 GPUs (4 nodes × 8 GPUs)
# See the hackathon repo for SLURM submit scripts
```

## References

- [GPU Mode Paris Hackathon 2026](https://github.com/gpu-mode/paris-hackathon-2026-training) — challenge repo
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — key inspiration for WSD schedule and scaling rules
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — distributed training reference

## Team

- **Edouard Foussier** — [GitHub](https://github.com/edouardfoussier) · [LinkedIn](https://linkedin.com/in/edouardf)
- **Boris** (rm-wu) — [GitHub](https://github.com/rm-wu)
- + 2 teammates
