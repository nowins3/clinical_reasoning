# Clinical Reasoning LLM Challenge

A reproducible notebook for comparing clinical text generation models on the Zindi Clinical Reasoning LLM Challenge. Given a clinical scenario and its questions, the goal is to produce a response resembling the human clinician's answer. The notebook is a research and competition workflow, **not** a clinical decision support system.

## What is included

- `clinical_reasoning.ipynb`: data checks, model setup, supervised fine tuning, validation, prediction, and an optional teacher candidate workflow.
- Generative model configurations for Med42-v2, BioMedLM, ChatDoctor, BioGPT, Phi4-MedQA, MediTron, Clinical T5, and Gemma 3 1B.
- A separate GatorTron encoder baseline for clinical-panel classification and nearest-answer retrieval. Its retrieved answer comes from another case; it is not newly generated.
- Notes explaining why Google Med-PaLM cannot be trained locally from publicly downloadable weights.

The original notebook's exploratory model notes are retained and distinguished from implemented experiments. No training results or leaderboard scores are claimed.

## Setup

Use Python 3.10 or newer, a PyTorch installation compatible with your hardware, and the following dependencies:

```bash
python -m pip install "transformers>=4.51,<5" "peft>=0.13,<1" \
  "datasets>=3,<5" "accelerate>=1,<2" "sentencepiece>=0.2,<1" \
  "sacremoses>=0.1,<1" pandas numpy torch
```

Large checkpoints can require substantial GPU memory. Some model repositories are gated, and the Phi4-MedQA adapter declares a quantized base that may need a compatible CUDA and bitsandbytes installation. Review each repository's access conditions before downloading. For a reproducible run, pin the resolved package versions in your own environment.

## Data

Download the competition data separately and place these files in one directory:

| File | Purpose |
| --- | --- |
| `train.csv` | Required; includes `Master_Index`, `Prompt`, and `Clinician`. |
| `test.csv` | Required; includes `Master_Index` and `Prompt`. |
| `train_raw.csv`, `test_raw.csv` | Optional inspection copies; excluded from training. |

The notebook trains on `Prompt` → `Clinician`. Existing GPT4.0, LLAMA, and GEMINI answers are not used as target labels. Data is not distributed with this project. Confirm the competition rules and data permissions before sharing outputs.

Set the data and output paths before opening the notebook, or edit its configuration cell:

```bash
export CLINICAL_DATA_DIR=/absolute/path/to/competition-csvs
export CLINICAL_OUTPUT_DIR=/absolute/path/to/model-outputs
```

For a gated Hugging Face checkpoint, provide `HF_TOKEN` through an environment variable or a private notebook secret. Never place it in source code or commit it. A token appeared in an earlier notebook version and should be revoked.

## Run a model

1. Open `clinical_reasoning.ipynb` and run the environment, configuration, and data cells in order. Data validation checks required columns, IDs, and blank inputs. The validation partition groups repeated prompts so identical cases cannot cross the split.
2. Set `MODEL_KEY` to a key in the notebook's `MODELS` registry; the default is `biogpt`. Review batch size, token limits, epochs, and hardware settings.
3. Uncomment `tokenizer, model, trainer = train_generator(MODEL_KEY)` for a generative model. The run writes its checkpoint under `OUTPUT_DIR / MODEL_KEY / final`.
4. Run `evaluate_generator(MODEL_KEY, tokenizer, model)` to write validation predictions. Run `write_submission(MODEL_KEY, tokenizer, model)` to produce `submission.csv` with `Master_Index` and `Clinician` columns. Confirm the current competition submission format before uploading.

For a later session, use `load_finetuned_generator(MODEL_KEY)` before validation or prediction. For GatorTron, follow the dedicated `train_gatortron`, `evaluate_gatortron`, and `predict_gatortron` examples. Its retrieval index uses only the training partition.

Med42 can optionally create teacher candidates for **training rows only** with `make_teacher_candidates()`. Those candidates are not treated as verified clinical answers. The notebook does not implement Med-PaLM fine tuning or a custom multi-head latent attention architecture.

## Evaluation and limitations

Validation reports exact match and token F1 against held-out clinician responses. These scores describe text overlap, not medical accuracy or safety. Assess factual quality and clinically significant errors separately. Training, downloads, and inference have not been executed as part of this packaged notebook; dependency compatibility and model access must be checked in the intended runtime.
