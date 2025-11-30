import os

os.environ["HF_HOME"] = "/NS/llm-artifacts/nobackup/HF_HOME"  # set HF cache dir if desired
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.utils import launch_server_cmd
import sglang as sgl
import json
import pandas as pd
import argparse

p = argparse.ArgumentParser()
p.add_argument("--attention_backend", type=str, required=True)
p.add_argument(
    "--enable_deterministic_inference",
    action="store_true",
    help="Enable deterministic inference (True if flag is set)"
)

args = p.parse_args()

@sgl.function
def multiple_choice_eval(s, query, choices):
    """
    Temp 0 reference: https://docs.sglang.ai/references/frontend/frontend_tutorial.html#Constrained-Decoding
    Choices Method reference: https://docs.sglang.io/references/frontend/choices_methods.html#token-length-normalized (This should be default yet still added to be extra safe)
    Code based on: https://github.com/sgl-project/sglang/blob/main/examples/frontend_language/usage/choices_logprob.py
    """
    s += query + " "
    s += sgl.gen("multiple_choice_response", choices=choices, temperature=0, choices_method=sgl.token_length_normalized)
    return s

# Load Data

data_path = 'test_data.jsonl'

dataset = pd.read_json(data_path, lines=True)

model_path = "meta-llama/Llama-3.2-1B"

# Load Model

RUN_COMMAND = f"""python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --attention-backend {args.attention_backend}"""

if args.enable_deterministic_inference:
    RUN_COMMAND += " --enable-deterministic-inference"

SERVER_PROCESS, PORT = launch_server_cmd(RUN_COMMAND)

endpoint = f"http://localhost:{PORT}"
wait_for_server(endpoint)

sgl.set_default_backend(sgl.RuntimeEndpoint(endpoint))

os.makedirs("SGL_Outputs/", exist_ok=True)

save_file_name = f"SGL_Outputs/sgl_logprobs_attn_backend_{args.attention_backend}_"

if args.enable_deterministic_inference:
    save_file_name += "deterministic_true.jsonl"
else:
    save_file_name += "deterministic_false.jsonl"

with open(save_file_name ,"w", encoding="utf-8") as fout:
    for row_idx, row in dataset.iterrows():
        query = row["query"]
        choices = row["choices"]
        state = multiple_choice_eval.run(query=query, choices=choices)
        response = state.get_meta_info("multiple_choice_response")
        normalized_prompt_logprobs = response['normalized_prompt_logprobs']
        out_row = {'row_idx': row_idx}
        out_row["choice_logliks"] = normalized_prompt_logprobs
        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

terminate_process(SERVER_PROCESS)