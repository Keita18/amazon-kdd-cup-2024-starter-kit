![AMAZON KDD CUP 2024: MULTI-TASK ONLINE SHOPPING CHALLENGE FOR LLMS](https://aicrowd-production.s3.eu-central-1.amazonaws.com/challenge_images/amazon-kdd-cup-2024/amazon-kdd-cup-24-banner.jpg)
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/yWurtB2huX)

# 🛒 [Amazon KDD CUP 2024: Multi-Task Online Shopping Challenge for LLMs](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms) Starter Kit


This repository is the Amazon KDD Cup 2024 **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Tasks](#-tasks)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Getting Started](#-getting-started)
   - [How to write your own model?](#️-how-to-write-your-own-model)
   - [How to start participating?](#-how-to-start-participating)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
      - [How are my model responses parsed by the evaluators?](#-how-are-my-model-responses-parsed-by-the-evaluators-)
6. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Important Links](#-important-links)


# 📖 Competition Overview

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
- `task_name`: This field contains the name of the task. However, the exact task names are redacted, and we only provide participants with hashed task names (e.g. `task1`, `task2`). 
- `metric`: This field contains the metric used to evaluate the question (Details in Section "Evaluation Metrics"). 
- `track`: This field specifies the track the question comes from. 

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


## 📏 Evaluation Metrics
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

Please refer to [local_evaluation.py](local_evaluation.py) for more details on how we will evaluate your submissions.

# 🏁 Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#-how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# ✍️ How to write your own model?

Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/forks/new) to create a fork.

3.  **Clone the repository**

    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USER-NAME>/amazon-kdd-cup-2024-starter-kit.git
    cd amazon-kdd-cup-2024-starter-kit
    ```

4. **Install** competition specific dependencies!
    ```bash
    cd amazon-kdd-cup-2024-starter-kit
    pip install -r requirements.txt
    # an to run local_evaluation.py
    pip install -r requirements_eval.txt
    ```

5. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

6. Test your model locally using `python local_evaluation.py`.

7. Accept the Challenge Rules on the main [challenge page](https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms) by clicking on the **Participate** button. Also accept the Challenge Rules on the Task specific page (link on the challenge page) that you want to submit to.

8. Make a submission as described in [How to make a submission](#-how-to-make-a-submission) section.


## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission. 
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tracks](docs/submission.md#submitting-to-different-tracks).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.

## 💻 What hardware does my code run on ?
You can find more details about the hardware and system configuration in [docs/hardware-and-system-config.md](docs/hardware-and-system-config.md).
In summary, we provide you `4` x [[NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/)] in Phase 2.

Your solution will be given a certain amount of time for inference, after which it would be immediately killed and no results would be available. The time limit is set at 
| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | Track 5 |
| ------ | ------- | ------- | ------- | ------- | ------- |
| **Phase 2**| 70 minutes | 20 minutes | 30 minutes | 20 minutes | 140 minutes |

For reference, the baseline solution with zero-shot LLaMA3-8B-instruct consumes the following amount of time. 

| Phase  | Track 1 | Track 2 | Track 3 | Track 4 | 
| ------ | ------- | ------- | ------- | ------- | 
| **Phase 2**| 1490s | 397s | 576s | 359s | 

We limit the prediction time of each sample to at most **10 seconds**. This limit applies at a batch level. For example, for a batch of 8 samples, you should return the prediction after at most 80 seconds. Otherwise, your submission will be killed. 

Your maximum repo size is 200GB. 

## 🧩 How are my model responses parsed by the evaluators ?
Please refer to [parsers.py](parsers.py) for more details on how we parse your model responses.


# ❓ Frequently Asked Questions 
## Which track is this starter kit for ?
This starter kit can be used to submit to any of the tracks. You can find more information in [docs/submission.md#submitting-to-different-tracks](docs/submission.md#submitting-to-different-tracks).

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms
- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/amazon-kdd-cup-2024-multi-task-online-shopping-challenge-for-llms/leaderboards
