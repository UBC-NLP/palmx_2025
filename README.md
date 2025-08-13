# PalmX 2025 Shared Task MCQ Evaluation

This repository contains the evaluation code and data for the [PalmX 2025 Shared Task](https://palmx.dlnlp.ai/index.html) on Benchmarking LLMs for Arabic and Islamic Culture.

A lightweight CLI to evaluate Hugging Face causal LMs on the **PalmX 2025** subtask datasets (**culture** and **islamic**) using next-token log-likelihood over letter choices (A/B/C/D).

> Datasets on HF Hub:
> - `UBC-NLP/palmx_2025_subtask1_culture`
> - `UBC-NLP/palmx_2025_subtask2_islamic`

## Install

```bash
# (optional) create & activate a virtualenv first
pip install -r requirements.txt
```

> If your model is gated (e.g., some Llama variants or a private model), login first:
>
> ```bash
> huggingface-cli login
> ```

## Usage

```
python run_evaluation.py \
  --model_name UBC-NLP/NileChat-3B \
  --subtask culture \
  --phase dev \
  --batch_size 8 \
  --predictions_file predictions.csv \
  --log_outputs
```

### Arguments

- `--model_name`: HF model id or a local path to a directory containing weights & tokenizer.
- `--subtask`: `culture` or `islamic`.
- `--phase`: dataset split, `dev` or `test`.
- `--batch_size`: batch size used for scoring the choices (default: `8`).
- `--predictions_file`: path to save predictions CSV (default: `predictions.csv`).
- `--log_outputs`: if provided, writes a detailed per-item CSV to `outputs_log.csv` (customizable via `--log_file`).
- `--scores_file`: path where the final accuracy is written as `accuracy=<float>` (default: `scores.txt`).

### Outputs

- `predictions.csv` — two columns: `id`, `prediction` (predicted letter label, e.g., `A`, `B`, ...).
- `scores.txt` — one line with the final accuracy in `key=value` format, e.g.:
  ```
  accuracy=0.873500
  ```
- `outputs_log.csv` (when `--log_outputs` is set) — per-item details including question/choices, per-choice scores and probabilities, ground-truth, and correctness.

## How it works

For each MCQ item, the script formats a prompt like:

```
{question}

A. ...
B. ...
C. ...
D. ...
الجواب:
```

Then it scores the log-likelihood of the next token being ` A`, ` B`, ` C`, or ` D`. It uses a numerically-stable softmax over these log-likelihoods to produce per-choice probabilities and picks the argmax as the predicted label.

## Tips

- GPU is auto-detected. If you run out of memory, reduce `--batch_size` or try a smaller model.
- If your tokenizer has no pad token, we set it to EOS to allow batching with padding.
- Some models may need `--trust_remote_code` or different precision; customize `palmx_eval/processor.py` if needed.

## Citation
If you use this dataset or code in your research, please cite:

```bibtex


```

And the [Original Palm Dataset paper](https://aclanthology.org/2025.acl-long.1579/):
```bibtex
@inproceedings{alwajih-etal-2025-palm,
    title = "Palm: A Culturally Inclusive and Linguistically Diverse Dataset for {A}rabic {LLM}s",
    author = "Alwajih, Fakhraddin  and
      El Mekki, Abdellah  and
      Magdy, Samar Mohamed  and
      Elmadany, AbdelRahim A.  and
      Nacar, Omer  and
      Nagoudi, El Moatez Billah  and
      Abdel-Salam, Reem  and
      Atwany, Hanin  and
      Nafea, Youssef  and
      others",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1579/",
    doi = "10.18653/v1/2025.acl-long.1579",
    pages = "32871--32894",
    ISBN = "979-8-89176-251-0"
}

```

## License
This project is licensed under the CC-BY-NC-ND-4.0 License.

## Contact
For questions or feedback, please open an issue on this repository.
