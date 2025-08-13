#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from palmx_eval.processor import MCQProcessor, DEFAULT_CHOICE_PREFIXES, format_batch

DATASETS = {
    "culture": "UBC-NLP/palmx_2025_subtask1_culture",
    "islamic": "UBC-NLP/palmx_2025_subtask2_islamic",
}

def write_scores_file(accuracy: float, path: str = "scores.txt") -> None:
    # Write as a simple key=value pair for easy scraping by CI
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"accuracy={accuracy:.6f}\n")

def maybe_print_first_batch_details(batch_docs, probs_grouped, scores_grouped, preds, truths, choice_prefixes):
    print("\n--- Detailed Results for First Batch ---")
    for doc_idx, doc in enumerate(batch_docs):
        print(f"\nDocument ID: {doc['id']}")
        print(f"  Question: {doc['question']}")
        for j, choice_text in enumerate(doc['choices']):
            score = scores_grouped[doc_idx][j] if j < len(scores_grouped[doc_idx]) else float('nan')
            prob = probs_grouped[doc_idx][j] if j < len(probs_grouped[doc_idx]) else float('nan')
            label_char = choice_prefixes[j][0] if j < len(choice_prefixes) else str(j+1)
            print(f"    {label_char}. {choice_text} -> Score: {score:.4f}, Prob: {prob:.4f}")
        pred = preds[doc_idx]
        truth = truths[doc_idx]
        print(f"  Predicted Label: '{pred}'")
        print(f"  Ground Truth Label: '{truth}'")
        print(f"  Result: {'CORRECT ✅' if pred == truth else 'INCORRECT ❌'}")
    print("\n--- (End of first batch details) ---\n")

def log_outputs_csv(
    path: str,
    docs: List[Dict[str, Any]],
    preds: List[str],
    probs_grouped: List[List[float]],
    scores_grouped: List[List[float]],
):
    rows = []
    for i, doc in enumerate(docs):
        row = {
            "id": doc["id"],
            "question": doc["question"],
            "A": doc["choices"][0] if len(doc["choices"])>0 else "",
            "B": doc["choices"][1] if len(doc["choices"])>1 else "",
            "C": doc["choices"][2] if len(doc["choices"])>2 else "",
            "D": doc["choices"][3] if len(doc["choices"])>3 else "",
            "answer_label": doc["answer_label"],
            "pred_label": preds[i],
            "correct": preds[i] == doc["answer_label"],
        }
        # add per-choice scores and probs if available
        for j, label in enumerate(["A", "B", "C", "D"]):
            s = scores_grouped[i][j] if (i < len(scores_grouped) and j < len(scores_grouped[i])) else np.nan
            p = probs_grouped[i][j] if (i < len(probs_grouped) and j < len(probs_grouped[i])) else np.nan
            row[f"score_{label}"] = s
            row[f"prob_{label}"] = p
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Causal LM on PalmX 2025 (culture / islamic) MCQ tasks."
    )
    parser.add_argument("--model_name", required=True, help="HF model id or local path")
    parser.add_argument("--subtask", required=True, choices=["culture", "islamic"], help="Which subtask to evaluate")
    parser.add_argument("--phase", required=True, choices=["dev", "test"], help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring choices")
    parser.add_argument("--predictions_file", default="predictions.csv", help="Where to write predictions CSV")
    parser.add_argument("--scores_file", default="scores.txt", help="Where to write final accuracy as key=value")
    parser.add_argument("--log_outputs", action="store_true", help="If set, write a detailed per-item log CSV")
    parser.add_argument("--log_file", default="outputs_log.csv", help="Path for detailed per-item log when --log_outputs is set")
    return parser.parse_args()

def main():
    args = parse_args()

    dataset_name = DATASETS[args.subtask]
    dataset_split = args.phase

    try:
        processor = MCQProcessor(model_name=args.model_name)
    except Exception as e:
        print(f"Failed to initialize MCQProcessor: {e}")
        sys.exit(1)

    print(f"\nLoading dataset '{dataset_name}' split '{dataset_split}'...")
    try:
        data = load_dataset(dataset_name, split=dataset_split)
        print(f"Dataset loaded successfully with {len(data)} examples.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        sys.exit(2)

    all_predictions: List[str] = []
    all_ground_truths: List[str] = []
    all_ids: List[str] = []

    num_batches = (len(data) + args.batch_size - 1) // args.batch_size
    print(f"\n--- Starting evaluation on {len(data)} questions in {num_batches} batches ---")

    try:
        for i in range(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]
            batch_docs = format_batch(batch)

            batch_ids = [doc["id"] for doc in batch_docs]
            preds, probs_grouped, scores_grouped = processor.process_batch(batch_docs)
            truths = [doc["answer_label"] for doc in batch_docs]

            all_ids.extend(batch_ids)
            all_predictions.extend(preds)
            all_ground_truths.extend(truths)

            if i == 0:
                maybe_print_first_batch_details(
                    batch_docs, probs_grouped, scores_grouped, preds, truths, processor.choice_prefixes
                )

            if args.log_outputs:
                # append to per-item log CSV incrementally
                if not os.path.exists(args.log_file):
                    log_outputs_csv(args.log_file, batch_docs, preds, probs_grouped, scores_grouped)
                else:
                    # append mode
                    existing = pd.read_csv(args.log_file)
                    new_df = pd.DataFrame()
                    rows = []
                    for j in range(len(batch_docs)):
                        doc = batch_docs[j]
                        row = {
                            "id": doc["id"],
                            "question": doc["question"],
                            "A": doc["choices"][0] if len(doc["choices"])>0 else "",
                            "B": doc["choices"][1] if len(doc["choices"])>1 else "",
                            "C": doc["choices"][2] if len(doc["choices"])>2 else "",
                            "D": doc["choices"][3] if len(doc["choices"])>3 else "",
                            "answer_label": doc["answer_label"],
                            "pred_label": preds[j],
                            "correct": preds[j] == doc["answer_label"],
                        }
                        for k, label in enumerate(["A", "B", "C", "D"]):
                            s = scores_grouped[j][k] if (j < len(scores_grouped) and k < len(scores_grouped[j])) else np.nan
                            p = probs_grouped[j][k] if (j < len(probs_grouped) and k < len(probs_grouped[j])) else np.nan
                            row[f"score_{label}"] = s
                            row[f"prob_{label}"] = p
                        rows.append(row)
                    new_df = pd.DataFrame(rows)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_csv(args.log_file, index=False)

        # Accuracy
        correct = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t)
        total = len(all_ground_truths)
        accuracy = correct / total if total > 0 else 0.0

        print("\n--- Overall Evaluation Complete ---")
        print(f"Total Questions Evaluated: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Overall Accuracy: {accuracy:.2%}")

        # Save predictions
        pd.DataFrame({"id": all_ids, "prediction": all_predictions}).to_csv(args.predictions_file, index=False)
        print(f"Predictions saved to '{args.predictions_file}'.")

        # Save scores
        write_scores_file(accuracy, args.scores_file)
        print(f"Final accuracy written to '{args.scores_file}'.")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
