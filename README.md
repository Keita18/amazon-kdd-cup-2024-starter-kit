![AMAZON KDD CUP 2024: MULTI-TASK ONLINE SHOPPING CHALLENGE FOR LLMS](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1139/566667103918dae81381.jpg)
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/yWurtB2huX)

# [Amazon KDD CUP 2024: Multi-Task Online Shopping Challenge for LLMs](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms) Starter Kit


This repository is the Amazon KDD Cup 2024 **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

#  Competition Overview

Online shopping is complex, involving various tasks from browsing to purchasing, all requiring insights into customer behavior and intentions. This necessitates multi-task learning models that can leverage shared knowledge across tasks. Yet, many current models are task-specific, increasing development costs and limiting effectiveness. Large language models (LLMs) have the potential to change this by handling multiple tasks through a single model with minor prompt adjustments. Furthermore, LLMs can also improve customer experiences by providing interactive and timely recommendations. However, online shopping, as a highly specified domain, features a wide range of domain-specific concepts (e.g. brands, product lines) and knowledge (e.g. which brand produces which products), making it challenging to adapt existing powerful LLMs from general domains to online shopping.

Motivated by the potentials and challenges of LLMs, we present **ShopBench**, a massive challenge for online shopping, with `57 tasks` and `~20000 questions`, derived from real-world Amazon shopping data. All questions in this challenge are re-formulated to a unified text-to-text generation format to accommodate the exploration of LLM-based solutions. ShopBench focuses on four main key shopping skills (which will serve as **Tracks 1-4**): 
- shopping concept understanding
- shopping knowledge reasoning
- user behavior alignment
- multi-lingual abilities

In addition, we set up **Track 5: All-around** to encourage even more versatile and all-around solutions. Track 5 requires participants to solve all questions in Tracks 1-4 with **a single solution**, which is expected to be more principled and unified than track-specific solutions to Tracks 1-4. We will correspondingly assign larger awards to Track 5. 

# 📊 Dataset

ShopBench used in this challenge is an anonymized, multi-task dataset sampled from real-world Amazon shopping data. Statistics of ShopBench is given in the following Table. 

| 	# Tasks	  | # Questions	| # Products	| # Product Category	| # Attributes	| # Reviews	| # Queries|
| ----------  | ----------- | --------    | -----------------   | ------------- | --------- | ---------|
|	57          |	20598	      |   ~13300    |	400	                | 1032          |	~11200	  |~4500     |

ShopBench is split into a few-shot development set and a test set to better mimic real-world applications --- where you never know the customer's questions beforehand. With this setting, we encourage participants to use any resource that is publicly available (e.g. pre-trained models, text datasets) to construct their solutions, instead of overfitting the given development data (e.g. generating pseudo data samples with GPT). 

The development datasets will be given in json format with the following fields. 

- `input_field`: This field contains the instructions and the question that should be answered by the model. 
- `output_field`: This field contains the ground truth answer to the question. 
- `task_type`: This field contains the type of the task (Details in the next Section, "Tasks")
- `metric`: This field contains the metric used to evaluate the question (Details in Section "Evaluation Metrics"). 

However, the test dataset (which will be hidden from participants) will have a different format with only two fields: 
- `input_field`, which is the same as above. 
- `is_multiple_choice`: This field contains a `True` or `False` that indicates whether the question is a multiple choice or not. The detailed 'task_type' will not be given to participants. 

# 👨‍💻👩‍💻 Tasks
ShopBench is constructed to evaluate four important shopping skills, which correspond to Tracks 1-4 of the challenge. 

- **Shopping Concept Understanding**: There are many domain-specific concepts in online shopping, such as brands, product lines, etc. Moreover, these concepts often exist in short texts, such as queries, making it even more challenging for models to understand them without adequate contexts. This skill emphasizes the ability of LLMs to understand and answer questions related to these concepts. 
- **Shopping Knowledge Reasoning**: Complex reasoning with implicit knowledge is involved when people make shopping decisions, such as numeric reasoning (e.g. calculating the total amount of a product pack), multi-step reasoning (e.g. identifying whether two products are compatible with each other). This skill focuses on evaluating the model's reasoning ability on products or product attributes with domain-specific implicit knowledge. 
- **User Behavior Alignment**:  User behavior modeling is of paramount importance in online shopping. However, user behaviors are highly diverse, including browsing, purchasing, query-then-clicking, etc. Moreover, most of them are implicit and not expressed in texts. Therefore, aligning with heterogeneous and implicit shopping behaviors is a unique challenge for language models in online shopping, which is the primary aim of this track.  
- **Multi-lingual Abilities**: Multi-lingual models are especially desired in online shopping as they can be deployed in multiple marketplaces without re-training. Therefore, we include a separate multi-lingual track, including multi-lingual concept understanding and user behavior alignment, to evaluate how a single model performs in different shopping locales without re-training. 

In addition, we setup Track 5: All-around, requiring participants to solve all questions in Tracks 1-4 with a unified solution to further emphasize the generalizability and the versatility of the solutions. 

ShopBench involves a total of 5 types of tasks, all of which are re-formulated to text-to-text generation to accommodate LLM-based solutions. 

- **Multiple Choice**: Each question is associated with several choices, and the model is required to output a single correct choice.
- **Retrieval**: Each question is associated with a requirement and a list of candidate items, and the model is required to retrieve all items that satisfy the requirement. 
- **Ranking**: Each question is associated with a requirement and a list of candidate items, and the model is required to re-rank all items according to how each item satisfies the requirement. 
- **Named Entity Recognition**: Each question is associated with a piece of text and an entity type. The model is required to extract all phrases from the text that fall in the entity type. 
- **Generation**: Each question is associated with an instruction and a question, and the model is required to generate text pieces following the instruction to answer the question. There are multiple types of generation questions, including extractive generation, translation, elaboration, etc.    

To test the generalization ability of the solutions, the development set will only cover a part of all 57 tasks, resulting to tasks that are unseen throughout the challenge. However, all 5 task types will be covered in the development set to help participants understand the prompts and output formats.   


## Evaluation Metrics
ShopBench includes multiple types of tasks, each requiring specific metrics for evaluation. The metrics selected are as follows:
- **Multiple Choice:** Accuracy is used to measure the performance for multiple choice questions.
- **Ranking:** Normalized Discounted Cumulative Gain (NDCG) is used to evaluate ranking tasks.
- **Named Entity Recognition (NER):** Micro-F1 score is used to assess NER tasks.
- **Retrieval:** Hit@3 is used to assess retrieval tasks. The number of positive samples not exceeding 3 across ShopBench.
- **Generation:** Metrics vary based on the task type:
  - Extraction tasks (e.g., keyphrase extraction) uses ROUGE-L.
  - Translation tasks uses BLEU score.
  - For other generation tasks, we employ [Sentence Transformer](https://huggingface.co/sentence-transformers) to calculate sentence embeddings of the generated text $x_{gen}$ and the ground truth text $x_{gt}$. We then compute the cosine similarity between $x_{gen}$ and $x_{gt}$ (clipped to [0, 1]) as the metric. This approach focuses on evaluations on text semantics rather than just token-level accuracy.

As all tasks are converted into text generation tasks, rule-based parsers will parse the answers from participants' solutions. Answers that parsers cannot process will be scored as 0. The parsers will be available to participants.

Since all these metrics range from [0, 1], we calculate the average metric for all tasks within each track (macro-averaged) to determine the overall score for a track and identify track winners. The overall score of Track 5 will be calculated by averaging scores in Tracks 1-4. 

# 🗃️ Submission

The challenge would be evaluated as a code competition. Participants must submit their code and essential resources, such as fine-tuned model weights and indices for Retrieval-Augmented Generation (RAG), which will be run on our servers to generate results and then for evaluation.

## Hardware and System Configuration
We apply a limit on the hardware available to each participant to run their solutions. Specifically, 

- All solutions will be run on [AWS g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances equipped with [NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/). 
- Solutions for Phase 1 will have access to `2` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
- Solutions for Phase 2 will have access to `4` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
Please note that NVIDIA T4 uses somewhat outdated architectures and is thus not compatible with certain acceleration toolkits (e.g. Flash Attention), so please be careful about compatibility.

Besides, the following restrictions will also be imposed. 

- Network connection will be disabled (except for HuggingFace to download open-source checkpoints). 
- Each submission will be assigned a certain amount of time to run. Submissions that exceed the time limits will be killed and will not be evaluated. The tentative time limit is set as follows **[TO BE TESTED WITH AICROWD SUBMISSION SYSTEM]**. 

| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | Track 5 |
| ------ | ------- | ------- | ------- | ------- | ------- |
| **Phase 1**| 140 minutes | 40 minutes | 60 minutes | 60 minutes | 5 hours |

- Each team will be able to make up to **4 submissions per week**, with a maximum of **2 Track 5 all-around submissions** **[TO BE TESTED WITH AICROWD SUBMISSION SYSTEM]**. 

Based on the hardware and system configuration, we recommend participants to begin with 7B models. According to our experiments, 7B models like Vicuna-7B and Mistral can perform inference smoothly on 2 NVIDIA T4 GPUs, while 13B models will result in OOM. 

#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# How to write your own model?

Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models for this competition.

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```bash
    git@gitlab.aicrowd.com:aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit.git
    cd amazon-kdd-cup-2024-starter-kit
    ```

3. **Install** competition specific dependencies!
    ```bash
    cd amazon-kdd-cup-2024-starter-kit
    pip install -r requirements.txt
    # an to run local_evaluation.py
    pip install -r requirements_eval.txt
    ```

4. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

5. Test your model locally using `python local_evaluation.py`.

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

An example Dockerfile is provided in [utilities/_Dockerfile](utilities/_Dockerfile) which you can use as a starting point.

You can check detailed information about setting up runtime dependencies in the 👉 [docs/runtime.md](docs/runtime.md) file.

## What should my code structure be like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── README.md                       # Project documentation and setup instructions
├── aicrowd.json                    # Submission meta information - like your username, track name
├── data
│   └── development.json            # Development dataset local testing
├── docs
│   └── runtime.md                  # Documentation on the runtime environment setup, dependency confifgs
├── local_evaluation.py             # Use this to check your model evaluation flow locally
├── metrics.py                      # Scripts to calculate evaluation metrics for your model's performance
├── models
│   ├── README.md                   # Documentation specific to the implementation of model interfaces
│   ├── base_model.py               # Base model class 
│   ├── dummy_model.py              # A simple or placeholder model for demonstration or testing
│   └── user_config.py              # IMPORTANT: Configuration file to specify your model 
├── parsers.py                      # Model output parser
├── requirements.txt                # Python packages to be installed for model development
├── requirements_eval.txt           # Additional Python packages to be installed for local evaluation
└── utilities
    └── _Dockerfile                 # Example Dockerfile for specifying runtime via Docker
```

Finally, **you must specify your submission specific metadata JSON in `aicrowd.json`**

The `aicrowd.json` of each submission should contain the following content:

```json
{
    "challenge_id": "amazon-kdd-cup-24-understanding-shopping-concepts",
    "authors": [
      "aicrowd-bot"
    ],
    "gpu": false,
    "description": "(optional) description about your awesome agent"
}
```
**IMPORTANT: To use GPUs** - Set the GPU flag to `true`. 

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above. You can modify the `authors` and `description` keys. Please DO NOT add any additional keys to `aicrowd.json` unless otherwise communicated during the course of the challenge.

## How do I submit to different tracks ?

You can submit to different tracks by specifying task specific the `challenge_id` in [aicrowd.json](aicrowd.json). Here are the challenge ids for the different tracks: 

Sure, let's create a markdown table based on the information provided:

| Track Name                                | Challenge ID                          |
|-------------------------------------------|-----------------------------------------|
| Understanding Shopping Concepts           | amazon-kdd-cup-24-understanding-shopping-concepts |
| Shopping Knowledge Reasoning              | amazon-kdd-cup-24-shopping-knowledge-reasoning    |
| User Behavior Alignment                   | amazon-kdd-cup-24-user-behavior-alignment         |
| Multi-Lingual Abilities                   | amazon-kdd-cup-24-multi-lingual-abilities         |
| All-Around                                | amazon-kdd-cup-24-all-around                      |



## How to make a submission?

Please follow the instructions in [docs/submission.md](ocs/submission.md) to make your first submission. 

**Note**: **Remember to accept the Challenge Rules** on the challenge page, and task page before making your first submission.


## Evaluation Metrics & Local Evaluation
Please refer to [local_evaluation.py](local_evaluation.py) for more details on how we will evaluate your submissions.


## Hardware and System Configuration
We apply a limit on the hardware available to each participant to run their solutions. Specifically, 

- All solutions will be run on [AWS g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances equipped with [NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/). 
- Solutions for Phase 1 will have access to `2` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
- Solutions for Phase 2 will have access to `4` x [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/s). 
Please note that NVIDIA T4 uses somewhat outdated architectures and is thus not compatible with certain acceleration toolkits (e.g. Flash Attention), so please be careful about compatibility.

Besides, the following restrictions will also be imposed. 

- Network connection will be disabled (except for HuggingFace to download open-source checkpoints). 
- Each submission will be assigned a certain amount of time to run. Submissions that exceed the time limits will be killed and will not be evaluated. The tentative time limit is set as follows **[TO BE TESTED WITH AICROWD SUBMISSION SYSTEM]**. 

| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | Track 5 |
| ------ | ------- | ------- | ------- | ------- | ------- |
| **Phase 1**| 140 minutes | 40 minutes | 60 minutes | 60 minutes | 5 hours |

- Each team will be able to make up to **4 submissions per week**, with a maximum of **2 Track 5 all-around submissions** **[TO BE TESTED WITH AICROWD SUBMISSION SYSTEM]**. 

Based on the hardware and system configuration, we recommend participants to begin with 7B models. According to our experiments, 7B models like Vicuna-7B and Mistral can perform inference smoothly on 2 NVIDIA T4 GPUs, while 13B models will result in OOM. 


**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms

- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/discussion

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/leaderboards
