![Airborne Banner](https://i.imgur.com/MxW7ySd.jpg)

# Airborne Object Tracking Challenge Starter Kit

👉 [Challenge page](https://www.aicrowd.com/challenges/airborne-object-tracking-challenge?utm_source=starter-kit&utm_medium=click&utm_campaign=prime-air)

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/hAuevqx9Tj)


This repository is the main Airborne Object Tracking challenge **Submission template and Starter kit**! 

Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your agent to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!
*  **SiamMOT**: Siamese Multi-Object Tracking baseline


# Table of Contents

1. [Competition Procedure](#competition-procedure)
2. [How to access and use dataset](#how-to-access-and-use-dataset)
3. [How to start participating](#how-to-start-participating)
4. [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies-)
5. [What should my code structure be like ?](#what-should-my-code-structure-be-like-)
6. [How to make submission](#how-to-make-submission)
7. [:star: SiamMOT baseline](#submit-siammot-baseline)
8. [Other concepts and FAQs](#other-concepts)
9. [Important links](#-important-links)


<p style="text-align:center"><img style="text-align:center" src="https://images.aicrowd.com/dataset_files/challenge_753/493d98aa-b7e5-45f8-aed1-640e4768f647_video.gif"  width="1024"></p>


#  Competition Procedure

The main task of the competition is to detect a collision threat reliably. In this challenge, you will train your agents locally and then upload them to AIcrowd (via git) to be evaluated. 

**The following is a high level description of how this round works**

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website].(https://www.aicrowd.com/challenges/airborne-object-tracking-challenge)
2. **Clone** this repo and start developing your solution.
3. **Train** your models to detect objects and write inference code in `test.py`.
4. [**Submit**](#how-to-submit-a-model) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-submit-a-model). The automated evaluation setup will evaluate the submissions against the test dataset to compute and report the metrics on the leaderboard of the competition.

# How to access and use dataset

The starter kit contains dataset exploration notebooks and helper functions to access the dataset.
You can check the instructions for the same here: 👉 [DATASET.md](/docs/DATASET.md).

# How to start participating

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:amazon-prime-air/airborne-detection-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd airborne-detection-starter-kit
    pip3 install -r requirements.txt
    ```

4. **Run local exploration notebook** present in `data/dataset-playground.ipynb` using `jupyter notebook` command locally.


5. Try out random prediction codebase present in `test.py`.


## How do I specify my software runtime / dependencies ?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `environment.yml` (conda environment), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [RUNTIME.md](/docs/RUNTIME.md) file.

## What should my code structure be like ?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Packages to be installed inside docker image
├── data                   # Your local dataset copy - you don't need to upload it (read DATASET.md)
├── requirements.txt       # Python packages to be installed
├── test.py                # IMPORTANT: Your testing/inference phase code, must be derived from AirbornePredictor (example in test.py)
└── utility                # The utility scripts to provide smoother experience to you.
    ├── docker_build.sh
    ├── docker_run.sh
    ├── environ.sh
    └── verify_or_download_data.sh
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "evaluations-api-airborne",
  "grader_id": "evaluations-api-airborne",
  "authors": ["aicrowd-bot"],
  "tags": "change-me",
  "description": "Random prediction model for Airborne challenge",
  "gpu": false
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

Please specify if your code will use a GPU or not for the evaluation of your model. If you specify `true` for the GPU, GPU will be provided and used for the evaluation.

## How to make submission

👉 [SUBMISSION.md](/docs/SUBMISSION.md)

**Best of Luck** :tada: :tada:

# SiamMOT baseline

[SiamMOT](https://github.com/amazon-research/siam-mot) is a region-based Siamese Multi-Object Tracking network that detects and associates object instances simultaneously.

This repository contains [SiamMOT](https://github.com/amazon-research/siam-mot) baseline interface which you can submit and improve upon.

> :warning: Please note that identical SiamMOT models (with delta <= 1.5% in EDR or AFDR) might be disqualified from winning the prize.
>
> An identical model is a model that uses the exact same code and config file provided with the baseline.

## Additional Steps

1. Change your entrypoint i.e. `run.sh` from `python test.py` to `python siam_mot_test.py`.
2. Copy the Dockerfile present in `siam-mot/Dockerfile` to repository root.
3. Set `gpu: true` in your `aicrowd.yaml`.
4. Follow common steps shared in [SUBMISSION.md](/docs/SUBMISSION.md)

```
#> cp siam-mot/Dockerfile Dockerfile
```

# Other Concepts

## Time constraints

You need to make sure that your model can predict airborne objects for each flight within 1000 seconds, otherwise the submission will be marked as failed. (_need revision on timeouts based on budget_)

## Local evaluation

You can also test end to end evaluation on your own systems. The scripts are available in `core/metrics` folder.
A working example is also available as [Colab Notebook here](https://colab.research.google.com/drive/1hobQBEfIxdPtc0jeMBtQKce8flrCKBq1?usp=sharing).

## Hardware used for evaluations

We use p3.2xlarge to run your evaluations i.e. 8 vCPU, 61 GB RAM, V100 GPU.

*(please enable GPU by putting "gpu": true in your aicrowd.json file)*

## Frequently Asked Questions

We have curated frequently asked questions and common mistakes on Discourse, you can read them here: [FAQ and Common mistakes](https://discourse.aicrowd.com/t/faqs-and-common-mistakes-while-making-a-submission/5781)


# 📎 Important links


💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/airborne-object-tracking-challenge

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/airborne-object-tracking-challenge/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/airborne-object-tracking-challenge/leaderboards
