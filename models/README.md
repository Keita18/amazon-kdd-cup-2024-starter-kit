# Guide to Writing Your Own Models

## Model Code Organization
For a streamlined experience, we suggest placing the code for all your models within the `models` directory. This is a recommendation for organizational purposes, but it's not a strict requirement.

## Model Base Class
Your models should inherit from the `ShopBenchBaseModel` class found in [base_model.py](base_model.py). We provide an example model, `dummy_model.py`, to illustrate how you might structure your own model. Crucially, your model class must implement the `batch_predict` method.

## Configuring Your Model
To ensure your model is recognized and utilized correctly, please specify your model class name in the [`user_config.py`](user_config.py) file, by following the instructions in the inline comments.

## Model Inputs and Outputs

### Inputs
- `batch` (`Dict[str, Any]`): A batch of inputs as a dictionary, where the dictionary has the following key:
    - `prompt` (`List[str]`): `A list if prompts representing the tasks in a batch`
- `is_multiple_choice` (`bool`): This indicates whether the task is a multiple choice question.

### Outputs

The output from your model's `batch_predict` function should be a list of string responses for all the prompts in the input batch.
Depending on the task, each response could be:
- A single integer (in the range [0, 3]) for multiple choice tasks.
- A comma-separated list of integers for ranking tasks.
- A comma-separated list of named entities for Named Entity Recognition (NER) tasks.
- (unconstrained) generated response for the generation tasks

For more information on how these responses are processed, please see [parsers.py](../parsers.py).


**Note** that the `task_type` will not be explicitly provided to your model. However, the information about the `task_type` is implicitly available in the prompt provided.

## Internet Access
Your model will not have access to the internet during evaluation. As such, you'll need to include any necessary model weights directly in your repository before submission. Ensure that your Model class is self-contained and fully operational without internet access.

