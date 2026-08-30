# VSPS — Verified Structured Proposition Synthesis for Vietnamese Image Captioning

Code and data accompanying the paper *"A Method for Reducing Hallucinations in Detailed
Vietnamese Image Captioning Using Supervision from Verified Propositions"* (FAIR 2026).

VSPS reduces object hallucination by verifying typed visual propositions **during
training-data construction**: propositions are generated from each training image,
verified into three states (SUPPORTED / UNCERTAIN / REJECTED), neutralized or excluded
when under-evidenced, and rendered into supervision for QLoRA SFT. The deployed model
generates captions in a **single pass** — no verifier, retrieval, or post-editing at
inference time.

The whole pipeline is reproducible **step by step**: each step below is one standalone
shell command whose printed numbers can be checked against the tables in the paper.

## 0. Hardware requirements

| Step | GPU | Notes |
|---|---|---|
| Smoke test, scoring, figures | none | CPU-only |
| Inference (Phase 2), latency | 1× 24 GB (RTX 3090/4090) | 4-bit NF4 |
| Fine-tuning (Phase 1b) | 1× 24 GB | QLoRA r=16 |
| Regenerating Phase-1a records | 1× 24 GB, ~30 h | **shipped** in `data/stage1_records/` — can be skipped |

## 1. Environment

Requires **Python 3.10–3.12** (with the `venv` module), a JDK (for VnCoreNLP), and
`git`. `setup.sh` refuses newer Pythons: the pinned torch-cu121 wheel line stops at
Python 3.12.

```bash
bash setup.sh          # venv + torch cu121 + pinned transformers==4.51.3 + VnCoreNLP
source .venv/bin/activate && export NCS_DATA=$PWD/ncs-data
bash get_data.sh       # KTVIC images + annotations (see notes inside the script)
python scripts/smoke_test.py   # PASS 4/4 = environment is correct (no GPU needed)
```

Two pins are **mandatory**: `transformers==4.51.3` and `trl==0.19.1`. Newer releases
change how Qwen2.5-VL handles `cache_position`/M-RoPE and change the trainer signature;
the adapter still loads but inference goes wrong *silently*.
`scripts/verify_environment.py` checks everything before any long run.

## 2. Layout

```
rescap/            core library
  pipeline/generate.py   proposition generation (typed, atomic) from an image
  pipeline/verify.py     dual-query verification, K=5 sampling, 3 states + lower-only ceilings
  pipeline/select.py     budgeted redundancy-aware selection (default budget 9)
  pipeline/realize.py    rendering: template path / constrained-LM path
  chair.py               CHAIR-vi (object matching over VnCoreNLP word segmentation)
scripts/           every experiment in the paper — one file per experiment
data/
  stage1_records/  3,700 verified Phase-1a records, one per training image (27 MB packed)
  supervision/     pre-built supervision store (sft.jsonl) — ready to train on
  coco_probe/      COCO manifests + the frozen Vietnamese→COCO object dictionary
  results/         raw numbers behind every table in the paper
  screening/, stress50/, vram/   auxiliary experiment data
```

## 3. Reproduction, phase by phase

### Phase 1a — generate + verify propositions (optional, ~30 GPU-hours; results shipped)

```bash
tar -xzf data/stage1_records/stage1_records.tar.gz -C $NCS_DATA   # use shipped records
# OR regenerate from scratch:
python scripts/generate_stage1.py --split train --out $NCS_DATA/stage1
python scripts/backoff_nguoi.py --in $NCS_DATA/stage1 --out $NCS_DATA/stage1_nguoi
python scripts/uncap_nguoi.py  --in $NCS_DATA/stage1_nguoi --out $NCS_DATA/stage1_final
```

`backoff_nguoi.py` re-verifies person entities with the neutral noun (neutralization);
`uncap_nguoi.py` performs the offline un-capping from stored evidence (τ = 0.9,
recovers 11,769 propositions on 2,351 images — printed at the end of the run).

### Phase 1b — build supervision + fine-tune (QLoRA, ~2 h per run on an RTX 4090)

```bash
# use the shipped store data/supervision/sft.jsonl, or rebuild it from stage-1 records
python scripts/train_stage2.py --data data/supervision --out $NCS_DATA/runs/sft --seed 42
# controls: --no-4bit (bf16 LoRA, no quantization) or --use-8bit
```

The main system of the paper is **SFT-only** (7,177 examples: 3,617 short + 3,560
detailed, two modes sharing one adapter; seeds 42/43/44).

### Phase 2 — single-pass inference + scoring (558-image KTVIC test split)

```bash
python scripts/evaluate.py --adapter $NCS_DATA/runs/sft --split test \
    --out $NCS_DATA/results/eval.json
python scripts/score_ci.py $NCS_DATA/results/eval.json     # 10k-sample bootstrap CI
```

Expected numbers (main system, seed 42 — the paper's main tables): hallucinated
objects/caption **1.57** · CHAIR_s **79.2** · CHAIR_i **50.1** · objects mentioned/caption
**3.13** · short-mode CIDEr **17.3** · unsupported gender attribution **4.1%**.
Across seeds 42/43/44: 1.55 ± 0.04 · CHAIR_s 78.3 ± 0.9 · CHAIR_i 49.7 ± 0.4.
Zero-shot on the same backbone: 4.26 hallucinated objects/caption, CHAIR_s 98.0%.
Template-only variant (VSPS-Base): 1.15 hallucinated objects · CHAIR_s 70.4%.

### Satellite experiments

```bash
python scripts/coco_probe.py --manifest data/coco_probe/manifest_full5000.json  # out-of-domain COCO
python scripts/score_coco_probe.py     # expected: 0.45 → 0.17 halluc./caption (−63%), CHAIR_s 34.4% → 15.4%
python scripts/doichung.py --method vcd            # VCD baseline (CVPR'24), same backbone
python scripts/doichung.py --method selfcorrect    # Self-Correction baseline
python scripts/run_ablations.py                    # budget × policy grid
python scripts/measure_latency.py                  # median 0.81 s (short) / 1.88 s (detailed) on RTX 4090
python scripts/stress50_analysis.py \
    --manifest data/stress50/stress50_manifest.json \
    --preds "zeroshot=data/results/zeroshot-detailed.preds.json" \
            "vsps=data/results/vsps-detailed.preds.json" \
    --out stress50_report.json                     # 50-image challenge set (gender/count/colour)
python scripts/agreement.py                        # Cohen's kappa (needs the annotation files, available on request)
```

### Optional: preference-tuning exploration (negative result)

`scripts/build_dpo_data.py` and the `--dpo` flag of `train_stage2.py` reproduce the
preference-tuning experiments reported as a controlled negative result: on the 4-bit
backbone, every tested dose degraded CIDEr and leaked non-Vietnamese tokens. They are
not part of the main system.

### Figures

```bash
python scripts/plot_results.py && python scripts/plot_training.py && \
python scripts/plot_qualitative.py && python scripts/plot_pha.py
```

## 4. Number-to-file traceability

Every number in the paper traces back to a file in `data/results/`. Two independent
checks, both CPU-only and both runnable straight from this repository:

```bash
# Re-score every shipped prediction file from scratch and compare with the shipped
# summary — verified to reproduce data/results/abs_halluc_summary.json bit-for-bit:
python scripts/abs_halluc.py --results data/results --out abs_halluc_check.json

# Replay the verification rule engine over the shipped Phase-1a records
# (default --limit 50; pass --limit 3700 for the full set, ~10 min on CPU):
python scripts/validate_replay.py --in $NCS_DATA/stage1 --limit 3700
```

Known property of the replay harness: on the full 3,700 records the stored probe
samples reproduce 100.0% (381,182/381,182), while verdict reproduction is 98.3%;
the discrepancies concentrate on UNCERTAIN-adjacent, near-boundary cases
(REJECTED→SUPPORTED flips: 1 in 192,216). The harness therefore refuses (99% gate)
to let replay-derived numbers be quoted. All numbers in the paper come from the
stored verdicts and real model runs, not from the replay harness.

## 5. License & contact

Code: MIT (see `LICENSE`). KTVIC and COCO images are distributed under their own
original licenses and are not included here. Contact: see the author addresses in the
paper.
