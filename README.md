# Introduction
Welcome! This repository contains the code for the submission "Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models". We build upon the publicly available code from the paper "Argumentative Large Language Models for Explainable and Contestable Claim Verification". 

## Running the Experiments
To run the experiments, please follow these steps:
1. Install the dependencies in requirements.txt. You may need to update the `transformers` library to a more recent version.
2. Run experiments using the `python3 main.py <OPTIONS>` command. Note that if using a non-default uncertainty quantification method (Semantic Entropy, Eccentricity, or LUQ), the experiment must first be run with run-phase="first" to generate the raw uncertainty outputs which are stored in pickle files. Then, the same experiment with run-phase="second" must be run to perform the binned normalization and performance evaluation for valid results. For the full list of options, run `python3 main.py -h`

## Acknowledgements
The Uncertainpy package is a third-party public package which we adapt for use in our experiments (https://github.com/nicopotyka/Uncertainpy).