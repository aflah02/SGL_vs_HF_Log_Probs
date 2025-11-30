import os

os.environ["HF_HOME"] = "/NS/llm-artifacts/nobackup/HF_HOME"  # set HF cache dir if desired

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import torch
from datasets import load_dataset
import json

# Load Data

data_path = 'test_data.jsonl'

ds = load_dataset("json", data_files={"data": data_path}, split="data")

# Load Model

model_path = "meta-llama/Llama-3.2-1B"
device = 'cuda:0'

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def _compute_logliks_for_query(model, device, query: str, choices: list):
    """
    Given a query (string) and a list of choice strings, return a list of
    log-likelihoods (floats) where each value is log p(choice_tokens | query_tokens).
    If a choice cannot be evaluated (e.g., empty), returns a large negative number.
    """
    # prepare combined inputs: query + choice (no special tokens added)
    # We intentionally avoid add_special_tokens=False so we get consistent token positions

    inputs = [query + " " + c for c in choices]  # adding a space to separate can help tokenization
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)  # (B, L)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # compute query token length once (use tokenizer without special tokens)
    q_enc = tokenizer(query, add_special_tokens=False)
    qlen = len(q_enc["input_ids"])
    B, L = input_ids.shape

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        # shift logits/labels: logits[:-1] predict tokens 1..L-1
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, L-1)

        # compute log softmax once
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)

        result_lls = []
        result_token_wise = []
        for b in range(B):
            # actual unpadded length
            if attention_mask is None:
                # if no mask was returned, assume entire length is used
                seq_len = L
            else:
                seq_len = int(attention_mask[b].sum().item())
            # query tokens occupy indices 0..qlen-1
            # choice tokens occupy indices qlen .. seq_len-1 (inclusive)
            if qlen >= seq_len:
                # no choice tokens present (or query uses whole sequence) => undefined conditional prob
                result_lls.append(float("-1e9"))
                continue

            # in shifted labels coordinates, prediction for token at original index t is at index (t-1)
            start_label_idx = qlen - 1  # corresponds to predicting token at index qlen
            end_label_idx = seq_len - 2  # corresponds to predicting token at index seq_len-1
            if start_label_idx > end_label_idx:
                # no tokens to sum
                result_lls.append(float("-1e9"))
                continue

            # gather the log probs for the true labels at those positions
            label_positions = torch.arange(start_label_idx, end_label_idx + 1, device=device)
            # shift_labels[b] length is L-1; index into it at label_positions
            token_ids = shift_labels[b].index_select(0, label_positions)  # (num_choice_tokens,)
            token_logps = log_probs[b].index_select(0, label_positions).gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
            # sum log probs
            token_logps_list = token_logps.tolist()
            ll = float(token_logps.sum().item())
            result_lls.append(ll)

            merge_token_ids_and_log_probs = []

            for token_id, token_logps in zip(token_ids, token_logps_list):
                token_id = token_id.item()
                merge_token_ids_and_log_probs.append([token_id, tokenizer.decode(token_id), token_logps])

            result_token_wise.append(merge_token_ids_and_log_probs)
    return result_lls, result_token_wise

os.makedirs("HF_Outputs/", exist_ok=True)

with open("HF_Outputs/hf_logprobs.jsonl" ,"w", encoding="utf-8") as fout:
    for row_idx, row in enumerate(ds):
        query = row.get('query')
        choices = row.get("choices")
        all_lls, result_token_wise = _compute_logliks_for_query(model, device, query, choices)
        out_row = {'row_idx': row_idx}
        out_row["choice_logliks"] = all_lls
        out_row["result_token_wise"] = result_token_wise
        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
