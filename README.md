# Argumentative LLMs
This is the oficial code repository for the paper "[Argumentative Large Language Models for Explainable and Contestable Claim Verification](https://arxiv.org/abs/2405.02079)". Argumentative LLMs (ArgLLMs) augment large language models with a formal reasoning layer based on computational argumentation. This approach enables ArgLLMs to generate structured and faithful explanations of their reasoning, while also allowing users to challenge and correct any identified issues.

## Getting Started
To run the main experiments, please follow these steps:
1. Install the required dependencies in requirements.txt. Note that, depending on the versions of HuggingFace models you use, you may need to update the `transformers` library to a more recent version.
1. Run experiments using the `python3 main.py <OPTIONS>` command. For the list of available options, please run `python3 main.py -h`

## Reproducibility Information
The experiments in our paper were run using the package versions in `requirements.txt` on a locally customized distribution of `Ubuntu 22.04.2`. The used machine was equipped with two RTX 4090 24GB GPUs and an Intel(R) Xeon(R) w5-2455X processor.

## Acknowledgements
We thank Nico Potyka and the other contributors to the [Uncertainpy](https://github.com/nicopotyka/Uncertainpy) package, which we adapted for use in our code.
