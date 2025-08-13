from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT_TEMPLATE = "{}\n\n{}\nالجواب:"
DEFAULT_CHOICE_PREFIXES = ["A.", "B.", "C.", "D."]

class MCQProcessor:
    """
    Handles processing and evaluation of Multiple Choice Questions (MCQs)
    using a Causal Language Model.
    """
    def __init__(self, model_name: str, device: str = None,
                 prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
                 choice_prefixes: List[str] = None):

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            try:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.prompt_template = prompt_template
        self.choice_prefixes = choice_prefixes if choice_prefixes is not None else DEFAULT_CHOICE_PREFIXES

        print(f"Loading model and tokenizer: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            # Avoid automatic device_map; keep simple and explicit for repeatability
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model/tokenizer for {model_name}: {e}")
            print("If using a gated model, ensure you have access and are logged in via `huggingface-cli login`.")
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def _format_doc_to_prompt_text(self, doc: Dict[str, Any]) -> str:
        """
        Formats a document (MCQ item) into a full prompt string.
        """
        question_text = doc["question"]
        options = []
        for i, opt_text in enumerate(doc["choices"]):
            if i < len(self.choice_prefixes):
                options.append(f"{self.choice_prefixes[i]} {opt_text}")
            else:
                options.append(f"{i+1}. {opt_text}")
        return self.prompt_template.format(question_text, "\n".join(options))

    def _get_choice_labels(self, doc: Dict[str, Any]) -> List[str]:
        """
        Returns the single letter labels for choices (e.g., ['A', 'B', 'C', 'D']).
        """
        num_choices = len(doc["choices"])
        return [self.choice_prefixes[i][0] for i in range(num_choices) if i < len(self.choice_prefixes)]

    def process_batch(self, batch_docs: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[List[float]]]:
        """
        Processes a batch of MCQ documents and returns:
          predicted_labels, probabilities_grouped, scores_grouped
        where groupings align with input docs and choices within each doc.
        """
        flat_full_sequences = []
        flat_prompt_tokens_ns_map = []
        flat_continuation_tokens_ns_map = []
        doc_choice_counts = []

        for doc in batch_docs:
            prompt_text = self._format_doc_to_prompt_text(doc)
            choice_labels = self._get_choice_labels(doc)
            doc_choice_counts.append(len(choice_labels))

            prompt_tokens_ns = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            for choice_label in choice_labels:
                continuation_text = " " + choice_label
                full_sequence_text = prompt_text + continuation_text
                flat_full_sequences.append(full_sequence_text)
                flat_prompt_tokens_ns_map.append(prompt_tokens_ns)
                continuation_tokens_ns = self.tokenizer.encode(continuation_text, add_special_tokens=False)
                flat_continuation_tokens_ns_map.append(continuation_tokens_ns)

        if not flat_full_sequences:
            return [], [], []

        max_len = self.tokenizer.model_max_length if self.tokenizer.model_max_length is not None else getattr(self.model.config, 'max_position_embeddings', 512)
        tokenized_batch = self.tokenizer(
            flat_full_sequences, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len, add_special_tokens=True
        )
        input_ids_batch = tokenized_batch.input_ids.to(self.device)
        attention_mask_batch = tokenized_batch.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            logits_batch = outputs.logits

        flat_choice_log_likelihoods = []
        for b_idx in range(len(flat_full_sequences)):
            current_input_ids_slice = input_ids_batch[b_idx]
            current_logits_slice = logits_batch[b_idx]
            current_prompt_tokens_ns = flat_prompt_tokens_ns_map[b_idx]
            current_continuation_tokens_ns = flat_continuation_tokens_ns_map[b_idx]

            if not current_continuation_tokens_ns:
                flat_choice_log_likelihoods.append(-float('inf'))
                continue

            idx_offset = 1 if (self.tokenizer.bos_token_id is not None and current_input_ids_slice.shape[0] > 0 and current_input_ids_slice[0] == self.tokenizer.bos_token_id) else 0
            start_of_continuation_in_ids = idx_offset + len(current_prompt_tokens_ns)
            actual_sequence_len = attention_mask_batch[b_idx].sum().item()

            if start_of_continuation_in_ids + len(current_continuation_tokens_ns) > actual_sequence_len:
                num_tokens_to_score = actual_sequence_len - start_of_continuation_in_ids
                if num_tokens_to_score <= 0:
                    flat_choice_log_likelihoods.append(-float('inf'))
                    continue
            else:
                num_tokens_to_score = len(current_continuation_tokens_ns)

            log_probs_full_sequence = torch.nn.functional.log_softmax(current_logits_slice, dim=-1)
            current_choice_log_likelihood = 0.0
            valid_tokens_scored = 0

            for i in range(num_tokens_to_score):
                token_id_being_predicted = current_continuation_tokens_ns[i]
                idx_of_token_in_input_ids = start_of_continuation_in_ids + i

                if idx_of_token_in_input_ids == 0 or idx_of_token_in_input_ids >= actual_sequence_len:
                    if i == 0: 
                        current_choice_log_likelihood = -float('inf')
                    break

                log_prob_dist_for_token = log_probs_full_sequence[idx_of_token_in_input_ids - 1, :]
                current_choice_log_likelihood += log_prob_dist_for_token[token_id_being_predicted].item()
                valid_tokens_scored += 1

            if valid_tokens_scored > 0:
                flat_choice_log_likelihoods.append(current_choice_log_likelihood)
            else:
                flat_choice_log_likelihoods.append(-float('inf'))

        all_predicted_labels, all_probabilities, all_scores_grouped = [], [], []
        current_flat_idx = 0
        for doc_idx, num_choices in enumerate(doc_choice_counts):
            if num_choices == 0:
                all_predicted_labels.append("N/A"); all_probabilities.append([]); all_scores_grouped.append([])
                continue

            doc_scores = flat_choice_log_likelihoods[current_flat_idx : current_flat_idx + num_choices]
            all_scores_grouped.append(doc_scores)
            scores_array = np.asarray(doc_scores)

            # Softmax calculation, robust to -inf
            exp_scores = np.exp(scores_array - np.max(scores_array, initial=-np.inf))
            exp_scores[scores_array == -float('inf')] = 0
            sum_exp_scores = np.sum(exp_scores)
            probabilities = exp_scores / sum_exp_scores if sum_exp_scores > 0 else np.full(num_choices, 1.0 / num_choices)
            most_probable_choice_idx = int(np.argmax(probabilities)) if probabilities.size > 0 else -1

            all_probabilities.append(probabilities.tolist())
            original_doc_choice_labels = self._get_choice_labels(batch_docs[doc_idx])
            pred_label = original_doc_choice_labels[most_probable_choice_idx] if 0 <= most_probable_choice_idx < len(original_doc_choice_labels) else "Error"
            all_predicted_labels.append(pred_label)
            current_flat_idx += num_choices

        return all_predicted_labels, all_probabilities, all_scores_grouped

    def calculate_accuracy(self, predicted_labels, ground_truth_labels) -> float:
        if len(predicted_labels) != len(ground_truth_labels):
            raise ValueError("Predicted and ground truth lists must have the same length.")
        if not predicted_labels: 
            return 0.0
        correct = sum(1 for pred, truth in zip(predicted_labels, ground_truth_labels) if pred == truth)
        return correct / len(predicted_labels)


def format_batch(batch) -> list:
    """
    Transforms a batch from the Hugging Face dataset format to the
    list of dictionaries format expected by MCQProcessor.
    """
    formatted_docs = []
    num_items = len(batch['id'])
    for i in range(num_items):
        doc = {
            "id": batch['id'][i],
            "question": batch['question'][i],
            "choices": [
                batch['A'][i],
                batch['B'][i],
                batch['C'][i],
                batch['D'][i]
            ],
            "answer_label": batch['answer'][i]
        }
        formatted_docs.append(doc)
    return formatted_docs
