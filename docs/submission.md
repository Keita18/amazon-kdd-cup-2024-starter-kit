# Making submission

This file will help you in making your first submission.


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