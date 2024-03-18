# Guide to Writing Your Own Models

## Model Code Organization
For a streamlined experience, we suggest placing the code for all your models within the `models` directory. This is a recommendation for organizational purposes, but it's not a strict requirement.

## Model Base Class
Your models should inherit from the `ShopBenchBaseModel` class found in [base_model.py](base_model.py). We provide an example model, `dummy_model.py`, to illustrate how you might structure your own model. Crucially, your model class must implement the `predict` method.

## Configuring Your Model
To ensure your model is recognized and utilized correctly, please specify your model class name in the [`user_config.py`](user_config.py) file, by following the instructions in the inline comments.

## Model Inputs and Outputs

### Inputs
Your model will receive two pieces of information for every task:
- `prompt` (`str`): This is the specific task's input prompt.
- `is_multiple_choice` (`bool`): This indicates whether the task is a multiple choice question.

### Outputs
The output from your model's `predict` function should always be a string. Depending on the task, this could be:
- A single integer (in the range [0, 3]) for multiple choice tasks.
- A comma-separated list of integers for ranking tasks.
- A comma-separated list of named entities for Named Entity Recognition (NER) tasks.

For more information on how these responses are processed, please see [parsers.py](../parsers.py).

### Task Type
Note that the type of task will not be explicitly provided to your model. However, you can infer the task type from the prompt provided.

## Internet Access
Your model will not have access to the internet during evaluation. As such, you'll need to include any necessary model weights directly in your repository before submission. Ensure that your Model class is self-contained and fully operational without internet access.

