import os
import re
import random
from typing import Any, Dict, List

import vllm
import torch

import pandas as pd

from transformers import LogitsProcessor

from .base_model import ShopBenchBaseModel

#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.95 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class IgnoreLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, texts):
        allowed_ids = []
        for text in texts:
            allowed_ids.extend(tokenizer.encode(text))
        self.allowed_ids = sorted(set(allowed_ids))
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] -= 100
        return scores


class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_number=5):
        pattern = ",".join([str(i) for i in range(max_number + 1)])
        self.allowed_ids = sorted(set(tokenizer.encode(pattern)))
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores


class CiteFromPromptLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, boost_factor=1.0):
        self.boost_factor = boost_factor
        self.eos_token_id = tokenizer.eos_token_id
        self.comma_token_id = tokenizer.convert_tokens_to_ids(",")
        
    def __call__(self, prompt_tokens_ids: List[int], past_token_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        tokens = list(set(prompt_tokens_ids + [self.eos_token_id, self.comma_token_id]))
        scores[tokens] += self.boost_factor
        return scores


def extract_options(text):
    question = []
    numbers = []
    options = []
    text = text.strip()
    for part in reversed(text.split("\n\n")):
        if not options:
            patterns = re.findall(r"\n(\d+\..+)", "\n" + part)
            if len(patterns) > 2:
                offset = part.find(patterns[0])
                part = part[:offset].strip()
                for t in patterns:
                    m = re.match(r"^(\d+)\.(.+)", t.strip())
                    if m:
                        option = m.group(2).strip()
                        if len(option) > 0:
                            numbers.append(int(m.group(1)))
                            options.append(option)
        if options:
            question.insert(0, part)
    if question:
        text = "\n\n".join(question).strip()
    return text, numbers, options


RETRIEVAL_PATTERNS = [
    "three comma-separated numbers",
    "three numbers, separated by comma",
    "three numbers, separated with comma",
    "Select three from the list",
]

RANKING_PATTERNS = [
    " respond with the ranking ",
]

NER_PATTERNS = [
    " extract ONLY ONE keyphrase ",
    " extract the keyphrase ",
    " output the entity ",
    " to the entity type ",
]


def classify_task(prompt: str, is_multiple_choice: bool):
    if is_multiple_choice:
        return "multiple-choice question answering"
    _, _, options = extract_options(prompt)
    if len(options) > 5 or any(pattern in prompt for pattern in RETRIEVAL_PATTERNS):
        return "retrieval"
    elif len(options) == 5 or any(pattern in prompt for pattern in RANKING_PATTERNS):
        return "ranking"
    elif any(pattern in prompt for pattern in NER_PATTERNS):
        return "named entity recognition"
    return "generation"


class VLLMModel(ShopBenchBaseModel):
    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        
        self.model_name = "./models/awq_average-150-192-theta-0.25"

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.llm = vllm.LLM(
            self.model_name,
            quantization="awq",
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
            max_model_len=3072,
            distributed_executor_backend="ray",
        )
        self.tokenizer = self.llm.get_tokenizer()

    def get_batch_size(self) -> int:
        return 16

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:

        data = []
        for i, prompt in enumerate(batch["prompt"]):
            task_type = classify_task(prompt, is_multiple_choice)
            prompt = re.sub(r" +\n", "\n", prompt)
            prompt = prompt.strip()
            data.append({
                "pid": i,
                "prompt": prompt,
                "task_type": task_type,
            })

        df = pd.DataFrame(data).set_index("pid")

        batch_size = 16

        for task_type, group in df.groupby("task_type"):
            batches = [group[i:i+batch_size] for i in range(0, len(group), batch_size)]
            for batch_df in batches:
                prompts = []
                for prompt in batch_df["prompt"].tolist():
                    prompts.append(f"<|im_start|>system\nYou are a helpful online shopping assistant. Your task is {task_type}.<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n")
                responses = self.generate_responses(task_type, prompts)
                df.loc[batch_df.index, "response"] = responses

        return df["response"].tolist()

    def generate_responses(self, task_type: str, prompts: List[str]) -> List[str]:

        logits_processors = [IgnoreLogitsProcessor(self.tokenizer, ["None", "none"])]

        if task_type == "named entity recognition":
            logits_processors += [CiteFromPromptLogitsProcessor(self.tokenizer, boost_factor=1.0)]
            max_new_tokens = 15
        elif task_type == "retrieval":
            logits_processors += [DigitLogitsProcessor(self.tokenizer, max_number=20)]
            max_new_tokens = 8
        elif task_type == "ranking":
            logits_processors += [DigitLogitsProcessor(self.tokenizer, max_number=5)]
            max_new_tokens = 9
        elif task_type == "multiple-choice question answering":
            logits_processors += [DigitLogitsProcessor(self.tokenizer, max_number=5)]
            max_new_tokens = 1
        else:
            max_new_tokens = 50
        
        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
                logits_processors=logits_processors,
            ),
            use_tqdm = False
        )
        
        batch_response = []
        for prompt, response in zip(prompts, responses):
            text = response.outputs[0].text
            if task_type == "named entity recognition":
                # TODO: check if the extracted words presented in the prompt
                pass
            elif task_type == "generation":
                sentences = text.strip().split(". ")
                if len(sentences) > 1:
                    result = ""
                    for sentence in sentences:
                        result += sentence.strip() + ". "
                        if len(result.split()) > 15:
                            break
                    text = result
            batch_response.append(text.strip())

        return batch_response
