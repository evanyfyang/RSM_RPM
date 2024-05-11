# Enhancing Generative Aspect-Based Sentiment Analysis with Relation-Level Supervision and Prompt

This repository releases the code of the following paper:

- Title: [Enhancing Generative Aspect-Based Sentiment Analysis with Relation-Level Supervision and Prompt](https://ieeexplore.ieee.org/abstract/document/10448322/)
- Authors: Yifan Yang∗, Yice Zhang∗, and Ruifeng Xu
- Conference: ICASSP 2024

This code employs our RSM and RPM on [GAS](https://aclanthology.org/2021.acl-short.64/).

## Requirements

- transformers==4.26.1
- pytorch==1.10.1
- einops=0.4.0
- torchmetrics==0.7.0
- tntorch==1.0.1
- pytorch-lightning==1.9.3

## Datasets
We have pre-processed data in `data`

If you want to process them by yourself:

Get datasets from:
- [ASTE-V2](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020)
- [ASQP](https://github.com/ZubinGou/multi-view-prompting/tree/main/data/asqp)
- [ACOS](https://github.com/ZubinGou/multi-view-prompting/tree/main/data/acos)

Put them into `data/raw` 

Then
- execute `python data_preprocess_quad.py --raw_data_dir data/raw/asqp --output_data_dir data/asqp_t5 --dataset rest15` for ASQP and ACOS datasets
- execute `python data_preprocess_triplet.py --raw_data_dir data/raw/aste --output_data_dir data/aste_t5 --dataset 14lap` for ASTE datasets
- You should change dataset names to run for each dataset

## Run our code!
Enter the code's dir and
- execute `chmod +x bash/*`,
- execute `bash/run_all.sh -c (Your_gpu_id)`.

You can also run single task by
- execute `chmod +x bash/*`
- execute `bash bash/train_extractor_aste.sh -a 5 -d 14lap -l 30 -c 0 -s 40 -t 1 -u 1` to run aste task
- execute `bash bash/train_extractor_quad.sh -a 1 -d rest15 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1` to run asqp task
- execute `bash bash/train_extractor_acos.sh -a 1 -d rest16 -l 20 -c ${CUDA_IDS} -s 40 -t 1 -u 1` to run acos task

### Parameters
- `-a` control the ratio of MLE loss to RSM loss
- `-d` dataset name, could be one of (14lap, 14res, 15res, 16res) in ASTE, (rest15, rest16) in ASQP, (rest16, laptop16) in ACOS
- `-l` loss = this parameter * 1e-5
- `-c` Your gpu id
- `-s` seed
- `-t` use RPM
- `-u` use RSM

Note that the performance posted in the paper is the average results of 5 run with 5 different random seeds, which has some differences from a single run.

## If you have any questions, please raise an `issue` or contact me

- email: `evanyfyang@163.com`
