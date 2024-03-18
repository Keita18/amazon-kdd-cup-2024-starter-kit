# Making submission

This file will help you in making your first submission.

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



## Submission Entrypoint

The evaluator will create an instance of a Model specified in from `models/user_config.py` to run the evaluation. 

## Setting up SSH keys

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


### IMPORTANT: Checking in your Models before submission!

Before you submit make sure that you have saved your models, which are needed by your inference code.
Lnowing your model weights will be significantly large files, you can use `git-lfs` to upload them. More details [here](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304). 

**Note**: If you check in your models directly into your git repo **without** using `git-lfs`, you may see errors like: 
- `fatal: the remote end hung up unexpectedly`
- `remote: fatal: pack exceeds maximum allowed size`

Sometimes, the reason could also be a large file checked in directly into git, and even if not available in current working directory, but present in git history.
If that happens, please ensure the model is also removed from git history, and then check in the model again using `git-lfs`. 

## How to submit your code?

You can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**

```bash
cd amazon-kdd-cup-2024-starter-kit

# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/amazon-kdd-cup-2024-starter-kit.git 
git push aicrowd master
```

```bash

# Commit All your changes
git commit -am "My commit message"

# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at:
`https://gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/amazon-kdd-cup-2024-starter-kit/issues`

**NOTE**: Please remember to update your username instead of `<YOUR_AICROWD_USER_NAME>` in the above link