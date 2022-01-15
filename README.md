# GINC small-scale in-context learning dataset

GINC (Generative In-Context learning Dataset) is a small-scale synthetic dataset for studying in-context learning.
The pretraining data is generated by a mixture of HMMs and the in-context learning prompt examples are also generated from HMMs (either from the mixture or not).
The prompt examples are out-of-distribution with respect to the pretraining data since every example is independent, concatenated, and separated by delimiters.
We provide code to generate GINC-style datasets of varying vocabulary sizes, number of HMMs, and other parameters.


## Quickstart
Please create a conda environment or virtualenv using the information in `conda-env.yml`, then install `transformers` by going into the `transformers/` directory and running `pip install -e .`.
Modify `consts.sh` to change the default output locations and insert code to activate the environment of choice.
Run `scripts/runner.sh` to run all the experiments on `sbatch`.

## Explore the data
The default dataset has vocab size 50 and the pretraining data is generated as a mixture of 5 HMMs.
The pretraining dataset is in `data/GINC_trans0.1_start10.0_nsymbols50_nvalues10_nslots10_vic0.9_nhmms10/train.json`
while in-context prompts are in `data/GINC_trans0.1_start10.0_nsymbols50_nvalues10_nslots10_vic0.9_nhmms10/id_prompts_randomsample_*.json`.
Note that "values" corresponds to "entities" and "slots" corresponds to "properties", using terminology from the paper (below).

## What does the data look like?
An example dataset is provided in the `data` directory, where an [example pretraining dataset](https://raw.githubusercontent.com/p-lambda/incontext-learning/main/data/GINC_trans0.1_start10.0_nsymbols50_nvalues10_nslots10_vic0.9_nhmms10/train.json) and an [example set of in-context prompts](https://raw.githubusercontent.com/p-lambda/incontext-learning/main/data/GINC_trans0.1_sta[…]0_nslots10_vic0.9_nhmms10/id_prompts_randomsample_3.json) can be found.

- Pretraining dataset file: Each line in the pretraining file contains one "document", which is a sequence sampled from a random HMM in the family.
- In-context prompt file: In the prompt file, every input in this dataset is 2 tokens long and each output is 1 token long, such that each in-context example is length 3 (the number in the file name). Each line in the file contains an in-context prompt and its label (and other metadata). The number of prompt examples in the file starts from 0 training examples (just 1 test example) to more training examples later in the file.

This repo contains the experiments for the paper [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080). If you found this repo useful, please cite
```
@article{xie2021incontext,
  author = {Sang Michael Xie and Aditi Raghunathan and Percy Liang and Tengyu Ma},
  journal = {arXiv preprint arXiv:2111.02080},
  title = {An Explanation of In-context Learning as Implicit Bayesian Inference},
  year = {2021},
}
```

