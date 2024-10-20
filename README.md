# Code-Mixed Hate Detection

### Prerequisites

1. Install requirements.txt file using
```bash
pip install -r requirements.txt
```
### Datasets

We have used publicly available datasets. The link of the datasets are:

Hate_Codemix : https://github.com/deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text

Hate_English : https://hasocfire.github.io/hasoc/2019/dataset.html

Hate_Hindi : https://github.com/hate-alert/HateCheckHIn


### Getting Started

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
