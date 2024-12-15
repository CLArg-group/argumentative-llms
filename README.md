# Argumentative LLM
A project augmenting large language models with argumentative reasoning.

## Getting Started
To run the experiments, please follow these steps:
1. Install the required dependencies in requirements.txt. Note that, depending on the versions of HuggingFace models you use, you may need to update the `transformers` library to a more recent version.
1. Download the datasets using the link in the paper.
1. Run experiments using the `python3 main.py <OPTIONS>` command. For the list of available options, please run `python3 main.py -h`

## Reproducibility Information
The experiments in our paper were run using the package versions in `requirements.txt` on a locally customized distribution of `Ubuntu 22.04.2`. The used machine was equipped with two RTX 4090 24GB GPUs and an Intel(R) Xeon(R) w5-2455X processor.