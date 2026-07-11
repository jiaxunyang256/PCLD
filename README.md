# PCLD

Official implementation for **Text-Centric Multimodal Context Modeling for
Chinese Patronizing and Condescending Language Detection**.

PCLD treats the ASR transcript as the primary decision signal. Top-level user
comments provide noisy post-publication social feedback, while visual,
acoustic, facial and Qwen-SKG representations are auxiliary evidence projected
into a shared textual space. On PCLMMPLUS, the final Full+Qwen-SKG model obtains
**0.8497 Macro-F1** and **0.9296 AUC** (five-seed mean).

## Highlights

- PCLMMPLUS contains 831 videos: 313 CPCL and 518 non-CPCL samples.
- The fixed video-level split is 600/66/165 for train/dev/test.
- All comments and feature streams stay with their source video to prevent
  cross-split leakage.
- Training uses label-free, offline Qwen3-Embedding-0.6B semantic caches.
- Raw videos, identifiers, comments, features and checkpoints are excluded from
  Git by default.

## Repository layout

```text
PCLD/
├── README.md
├── requirements.txt
├── configs/pclmmplus.yaml
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── supporting model modules
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── build_qwen_cache.py
│   └── preprocess/
├── splits/{train,dev,test}.txt
├── results/reported_metrics.json
├── tests/test_core.py
└── data/README.md
```

## Installation

Python 3.10 or newer is recommended. FFmpeg is required for audio extraction.

```bash
# Download or clone the anonymous repository first
cd PCLD
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data and features

Follow [data/README.md](data/README.md) to construct `data/index.csv`. Each row
is one video and contains its label, split and paths to ASR, comments and
pre-extracted evidence. Empty modality paths are supported and masked by the
data loader.

The included preprocessing commands are independent so their outputs can be
cached and audited:

```bash
python scripts/preprocess/extract_audio.py --video-dir data/raw --output-dir data/wav
python scripts/preprocess/transcribe_whisper.py --audio-dir data/wav --output-dir data/TXT
python scripts/preprocess/extract_asr_features.py --text-dir data/TXT --output-dir features/TEXT_features
python scripts/preprocess/extract_mfcc.py --audio-dir data/wav --output-dir features/AUDIO_features
python scripts/preprocess/extract_vit.py --video-dir data/raw --output-dir features/VIT_features
python scripts/preprocess/extract_face.py --video-dir data/raw --output-dir features/FACE_features --model checkpoints/fer_vt.ts
```

Build a label-free Qwen semantic cache after the ASR and comment paths have
been added to the index:

```bash
python scripts/build_qwen_cache.py \
  --index_csv data/index.csv \
  --output_dir features/QWEN_SKG \
  --output_index_csv data/index_qwen.csv \
  --text_mode asr_comments_skg \
  --target_column skg_cache_path
```

Set `data.index_csv` in the YAML file to the generated index. The full Qwen-SKG
setting appends up to 16 substring-matched SKG triples to the cached text. Cache
construction does not use labels or model predictions.

## Training and evaluation

Train one seed:

```bash
python scripts/train.py --config configs/pclmmplus.yaml --seed 42
```

Reproduce the five-seed protocol:

```bash
for seed in 13 21 42 87 100; do
  python scripts/train.py --config configs/pclmmplus.yaml --seed "$seed"
done
```

The threshold is selected using development Macro-F1, saved in the checkpoint
and then fixed for test evaluation.

```bash
python scripts/evaluate.py \
  --config configs/pclmmplus.yaml \
  --checkpoint outputs/pclmmplus_full_qwen_skg/42/best.pt \
  --split test
```

## Reported result

| Model | Accuracy | Macro-F1 | Recall | Precision | AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Full+Qwen-SKG | 0.8618 | **0.8497** | 0.7869 | 0.8353 | **0.9296** | 0.8922 |

Values are means over seeds 13, 21, 42, 87 and 100 on the same fixed test
split. Standard deviations and machine-readable values are available in
[`results/reported_metrics.json`](results/reported_metrics.json). Qwen-SKG is
best interpreted as sample-level semantic enrichment rather than proof of
isolated symbolic graph reasoning.

## Tests

```bash
python -m pytest -q
```

## Responsible use

This system is intended for research and human-in-the-loop moderation, not
autonomous surveillance, profiling or punitive decisions. CPCL examples concern
vulnerable groups, and both false positives and false negatives can cause harm.
Do not publish raw comments, user profiles, direct video identifiers or raw
videos unless platform policy and informed data-governance procedures permit
it. Prefer anonymized text and processed features.

## Citation

The submission is currently anonymous. Replace the placeholder below after the
camera-ready paper is public.

```bibtex
@inproceedings{anonymous2026pcld,
  title     = {Text-Centric Multimodal Context Modeling for Chinese Patronizing
               and Condescending Language Detection},
  author    = {Anonymous},
  year      = {2026}
}
```
