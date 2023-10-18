# Large Language Models are Better Reasoners with Self-Verification
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/WENGSYX/Self-Verification.svg?color=blue&style=flat-square">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/WENGSYX/Self-Verification">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/WENGSYX/Self-Verification">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/WENGSYX/Self-Verification">
</p>

This is the official implementation of `Large Language Models are Better Reasoners with Self-Verification`.
(**EMNLP 2023 Findings**)

## Demo

[https://github-production-user-asset-6210df.s3.amazonaws.com/64484703/250267125-281968bf-a8a2-438f-838b-d189c2501d40.mp4
https://user-images.githubusercontent.com/70048414/232352935-55c6bf7c-3958-406e-8610-0913475a0b05.mp4](https://github.com/WENGSYX/Self-Verification/assets/64484703/281968bf-a8a2-438f-838b-d189c2501d40)

## Installation
Make sure you have Python>=3.8 installed on your machine.
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install tqdm transformers sklearn pandas numpy sentencepiece openai
```

## Set your OpenAI API key
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

## Set arguments.
```
model=CODEX # {"gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "CODEX", "CODEX-001"}. "codex" is the smallest model.
dataset=multiarith # We can use other datasets. See help for the details.
api_time_interval=4.0 # Caution. The API allows users request API up to 20 times in a minutes, otherwise errors happen.
```

## Quick Start

### Demo
```
python demo.py
```

### Self-Verification (our proposal)
```
python main.py --method=verifier_cot --model=${model} --dataset=${dataset}
```

### CoT
```
# MultiArith and GSM8K are currently available.
python main.py --method=few_shot_cot --model=${model} --dataset=${dataset}
```



## Method

![main](./img/method.png)

1. Forward Reasoning, the LLM generates candidate thought chains and conclusions for a given problem text; 
2. Backward Verification, we use the LLM to verify whether the conditions meet the candidate conclusions and rank the candidate conclusions based on a verification score.



## Cite

> ```
> @misc{weng2023large,
>       title={Large Language Models are Better Reasoners with Self-Verification}, 
>       author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},
>       year={2023},
>       eprint={2212.09561},
>       archivePrefix={arXiv},
>       primaryClass={cs.AI}
> }
> ```
