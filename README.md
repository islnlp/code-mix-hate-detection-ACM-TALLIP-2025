# Code-Mixed Hate Detection [ACM TALLIP 2025] 

<p align="center">
  <img src="hate_logo.png" alt="Project Logo" width="150"/>
</p>

<div align="center">
  <a href="https://dl.acm.org/doi/abs/10.1145/3726866">
    <img src="https://img.shields.io/badge/ACM-TALLIP2025-blue" alt="ACM TALLIP 2025" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10.12-blue" alt="Python Version" />
  </a>
  <a href="https://huggingface.co/">
    <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-orange" alt="Hugging Face" />
  </a>
</div>


## 📌 Table of Contents
- [Prerequisites](#-prerequisites)
- [Datasets](#-datasets)
- [Getting Started](#-getting-started)
- [Experiments](#-experiments)
  - [Experiment 1](#experiment-1)
  - [Experiment 2](#experiment-2)
  - [Experiment 3](#experiment-3)
- [Contribution](#-contribution)
- [Citation](#-citation)

---

### 📋 Prerequisites

1. Install requirements.txt file using
```bash
pip install -r requirements.txt
```
### 📂 Datasets

We have used publicly available datasets. The link of the datasets are:

Hate_Codemix : https://github.com/deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text

Hate_English : https://hasocfire.github.io/hasoc/2019/dataset.html

Hate_Hindi : https://github.com/hate-alert/HateCheckHIn


### 🚀 Getting Started

To get results of models on Hatespeech_Codemix data directly run the file like this:

```
python3 Main_results/{model}.py
```
Select the model from the following: 
{0SVM_Codemix.py, 0RF_Codemix.py, 0NB_Codemix.py, 0indicBERT_Trans_Codemix.py, 0indicBERT_without_Trans_Codemix.py, 0mBERT_Trans_Codemix.py, 0mBERT_without_Trans_Codemix.py, 0muril_Trans_Codemix.py, 0muril_without_Trans_Codemix.py, 0XLMR_Trans_Codemix.py, 0XLMR_without_Trans_Codemix.py, 0XLM_Trans_Codemix.py, 0XLM_without_Trans_Codemix.py}

### Experiment 1

1. Hate and Non-hate samples are 1416 each of Hatespeech_English(new) and
Hatespeech_Hindi which is then combined with Hatespeech_Codemix.


To get results directly run the file like this:

``` 
python3 Main_results/{model}.py
```
Select the model from the following: {0SVM_Combined_1416, 0RF_Combined_1416, 0NB_Combined_1416, 0indicBERT_Trans_Combined_1416.py, 0indicBERT_without_Trans_Combined_1416.py, 0mBERT_Trans_Combined_1416, 0mBERT_without_Trans_Combined_1416, 0muril_Trans_Combined_1416.py, 0muril_without_Trans_Combined_1416.py, 0XLMR_Trans_Combined_1416, 0XLMR_without_Trans_Combined_1416, 0XLM_Trans_Combined_1416, 0XLM_without_Trans_Combined_1416}

2. Hate and Non-hate samples in Hatespeech_English(new) and Hatespeech_Hindi are
in the same ratio(Maximum possible samples) as Hatespeech_Codemix.

To get results directly run the file like this:

```
python3 Main_results/{model}.py
```
Select the model from the following: 
{0SVM_Combined_cmratio.py, 0RF_Combined_cmratio, 0NB_Combined_cmratio.py, 0indicBERT_Trans_Combined_cmratio.py, 0indicBERT_without_Trans_Combined_cmratio.py, 0mBERT_Trans_Combined_cmratio.py, 0mBERT_without_Trans_Combined_cmratio.py, 0muril_Trans_Combined_cmratio.py, 0muril_without_Trans_Combined_cmratio.py, 0XLMR_Trans_Combined_cmratio.py, 0XLMR_without_Trans_Combined_cmratio.py, 0XLM_Trans_Combined_cmratio.py, 0XLM_without_Trans_Combined_cmratio.py}


### Experiment 2


1. Incremental Mixing: A batch of 200 samples of Hate and Non-hate class each from
English and Hindi datasets were mixed along with Codemix data. We have then tried a
sample size of 400,600,800,1000,1200,1400.

To get results directly run the file like this:

```
python3 Experiment2/{model}.py
```
Select the model from the following: 
{0indicBERT_Trans_Combined_exp2_200.py, 0indicBERT_without_Trans_Combined_exp2_200.py, 0mBERT_Trans_Combined_exp2_200.py, 0mBERT_without_Trans_Combined_exp2_200.py, 0muril_Trans_Combined_exp2_200.py, 0muril_without_Trans_Combined_exp2_200.py, 0XLMR_Trans_Combined_exp2_200.py, 0XLMR_without_Trans_Combined_exp2_200.py, 0XLM_Trans_Combined_exp2_200.py, 0XLM_without_Trans_Combined_exp2_200.py}

These codes are default for 200 samples. For different sample sizes required lines are
commented in the code.



2. Label Ratio Mixing: A batch of 200 samples each from English and Hindi datasets in
which Hate and Non-hate class is present in the same ratio as in Codemix data were
mixed along with Codemix data.We have then tried a sample size of
400,600,800,1000,1200,1400.
   
To get results directly run the file like this:

```
python3 Experiment2/{model}.py
```
{0indicBERT_Trans_Combined_exp2_cmratio.py, 0indicBERT_without_Trans_Combined_exp2_cmratio.py, 0mBERT_Trans_Combined_exp2_cmratio.py, 0mBERT_without_Trans_Combined_exp2_cmratio.py, 0muril_Trans_Combined_exp2_cmratio.py, 0muril_without_Trans_Combined_exp2_cmratio.py, 0XLMR_Trans_Combined_exp2_cmratio.py, 0XLMR_without_Trans_Combined_exp2_cmratio.py, 0XLM_Trans_Combined_exp2_cmratio.py, 0XLM_without_Trans_Combined_exp2_cmratio.py}

These codes are default for 200 samples of Hindi and English dataset. For different
sample sizes required lines are commented in the code.




### Experiment 3
Here the training sets consisted solely of Hindi and English combined samples, only
English samples(only code-mix validation set) and only Hindi samples(only code-mix
validation set). Samples were taken in an incremental way like part 1 of Experiment 2.
Here codes for the highest possible sample size is given. For different sample sizes
required lines are commented in the code.



1. Mixed Validation Set: Comprises Hindi and English samples similar to training set.
   
To get results directly run the file like this:

```
python3 Experiment3/{model}.py
```
Select the model from the following: 
{0indicBERT_Trans_Combined_exp3_vcom.py, 0indicBERT_without_Trans_Combined_exp3_vcom.py, 0mBERT_Trans_Combined_exp3_vcom.py, 0mBERT_without_Trans_Combined_exp3_vcom.py, 0muril_Trans_Combined_exp3_vcom.py, 0muril_without_Trans_Combined_exp3_vcom.py, 0XLMR_Trans_Combined_exp3_vcom.py, 0XLMR_without_Trans_Combined_exp3_vcom.py, 0XLM_Trans_Combined_exp3_vcom.py, 0XLM_without_Trans_Combined_exp3_vcom.py}


2. Codemix Validation Set: Using the original Codemix validation set.
   
To get results directly run the file like this:

```
python3 Experiment3/{model}.py
```
Select the model from the following: 
{0indicBERT_Trans_Combined_exp3_vcod.py, 0indicBERT_without_Trans_Combined_exp3_vcod.py, 0mBERT_Trans_Combined_exp3_vcod.py, 0mBERT_without_Trans_Combined_exp3_vcod.py, 0muril_Trans_Combined_exp3_vcod.py, 0muril_without_Trans_Combined_exp3_vcod.py, 0XLMR_Trans_Combined_exp3_vcod.py, 0XLMR_without_Trans_Combined_exp3_vcod.py, 0XLM_Trans_Combined_exp3_vcod.py, 0XLM_without_Trans_Combined_exp3_vcod.py}


### 📜 Citation
```
@article{10.1145/3726866,
author = {Mazumder, Debajyoti and Kumar, Aakash and Patro, Jasabanta},
title = {Improving Code-Mixed Hate Detection by Native Sample Mixing: A Case Study for Hindi-English Code-Mixed Scenario},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2375-4699},
url = {https://doi.org/10.1145/3726866},
doi = {10.1145/3726866},
abstract = {Hate detection has long been a challenging task for the NLP community. The task becomes complex in a code-mixed environment because the models must understand the context and the hate expressed through language alteration. Compared to the monolingual setup, we see much less work on code-mixed hate as large-scale annotated hate corpora are unavailable for the study. To overcome this bottleneck, we propose using native language hate samples (native language samples/ native samples hereafter). We hypothesise that in the era of multilingual language models (MLMs), hate in code-mixed settings can be detected by majorly relying on the native language samples. Even though the NLP literature reports the effectiveness of MLMs on hate detection in many cross-lingual settings, their extensive evaluation in a code-mixed scenario is yet to be done. This paper attempts to fill this gap through rigorous empirical experiments. We considered the Hindi-English code-mixed setup as a case study as we have the linguistic expertise for the same. Some of the interesting observations we got are: (i) adding native hate samples in the code-mixed training set, even in small quantity, improved the performance of MLMs for code-mixed hate detection, (ii) MLMs trained with native samples alone observed to be detecting code-mixed hate to a large extent, (iii) the visualisation of attention scores revealed that, when native samples were included in training, MLMs could better focus on the hate emitting words in the code-mixed context, and (iv) finally, when hate is subjective or sarcastic, naively mixing native samples doesn’t help much to detect code-mixed hate. We have released the data and code repository to reproduce the reported results.},
note = {Just Accepted},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = mar,
keywords = {Code-mixed hate detection, Cross-lingual learning, Native sample mixing.}
}
```
