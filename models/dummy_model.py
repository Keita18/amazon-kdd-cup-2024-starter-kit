from typing import List, Union
import random
import os

from .base_model import ShopBenchBaseModel

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class DummyModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        """
        Generates a prediction based on the input prompt and task type.

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Args:
            prompt (str): The input prompt for the model.
            is_multiple_choice (bool): Indicates whether the task is a multiple choice question.

        Returns:
            str: The prediction as a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        possible_responses = [1, 2, 3, 4]

        if is_multiple_choice:
            # Randomly select one of the possible responses for multiple choice tasks
            return str(random.choice(possible_responses))
        else:
            # For other tasks, shuffle the possible responses and return as a string
            random.shuffle(possible_responses)
            return str(possible_responses)
            # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
            # For generation tasks, this should ideally return an unconstrained string.

class Vicuna2ZeroShot(ShopBenchBaseModel):
    """
    A baseline solution that uses Vicuna-7B to generate answers with zero-shot prompting. 
    """
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)
        ### model_path = 'lmsys/vicuna-7b-v1.5'
        ### Before submitting, please put necessary files to run Vicuna-7B at the corresponding path, and submit them with `git lfs`. 
        self.tokenizer = AutoTokenizer.from_pretrained('./models/vicuna-7b-v1.5/', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('./models/vicuna-7b-v1.5/', device_map='auto', torch_dtype='auto', trust_remote_code=True, do_sample=True)
        self.system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"


    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        prompt = self.system_prompt + prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs.input_ids = inputs.input_ids.cuda()
        if is_multiple_choice:
            # only one token for multiple choice questions. 
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=1, temperature=0)
        else:
            # 100 tokens for non-multiple choice questions. 
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=100, temperature=0)
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt):]
        return generation
