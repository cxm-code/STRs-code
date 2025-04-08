# Requirements
You need to configure the following environment before running the ProSTR model. 
It should be noted that this project is carried out in the Windows system, if you are using Linux system, We hope you can install the corresponding environment version yourself.
* Windows system
* NVIDIA GeForce RTX 3060
* PyCharm 2020
* 
# Prepare data
Reference:Johns N I, Gomes A L, Yim S S, et al.Metagenomic mining of regulatory elements enables programmable species-selective gene expression[J].Nature methods,2018, 15 (5): 323-329.

ecoli_generation.csv   The data used by the generator
ecoli_prediction.csv   The data used by the predictor

# Design Promoter Sequence
We take design promoters in E.coli as an example,to illustrate how to train the ProSTR model and design the promoter sequences
## 1.Training the generator
run \Generator\step1_cGAN.py
run \Generator\str.py
run \Generator\loss.py
run \Generator\step2_cGAN.py
## 2. Training the predictor
run \prediction\GRU.py
