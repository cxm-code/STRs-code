# Requirements
You need to configure the following environment before running the ProSTR model. 
It should be noted that this project is carried out in the Windows system, if you are using Linux system, We hope you can install the corresponding environment version yourself.
* Windows system
* NVIDIA GeForce RTX 3060
* PyCharm 2020
* Python 3.6.3
* torch==1.0.0
* scipy==1.1.0
* scikit-image==0.14.0
* scikit-learn==0.19.1
* pandas==0.20.3
* numpy==1.14.3

# Prepare data
Reference:Johns N I, Gomes A L, Yim S S, et al.Metagenomic mining of regulatory elements enables programmable species-selective gene expression.Nature methods,2018, 15 (5): 323-329.

* ecoli_generation.csv    The data used by the generator <br>
* ecoli_prediction.csv   The data used by the predictor <br>

# Design Promoter Sequence
We take design promoters in E.coli as an example,to illustrate how to train the ProSTR model and design the promoter sequences
## 1.Training the generator
* run \Generator\step1_cGAN.py    <br>

>>Train the first layer generation mechanism<br>

* run \Generator\str.py    <br>

>>Introduce STR in the generated sequence<br>

* run \Generator\loss.py   <br>

>>Calculate the similarity between the generated sequence and the natural sequence<br>

* run \Generator\step2_cGAN.py <br>

>>Train the second layer generation mechanism<br>
## 2. Training the predictor

run \prediction\GRU.py <br>
## 3.Evaluate the performance of ProSTR model generated sequences

* DNAshape:run valid\dnashape.py  <br>

* GC:run valid\GCviolin.py   <br>

* Diversity:run valid\edit1.py  run valid\geditdistence.py   <br>

* K-mer:run valid\kmer.py    <br>

* BLAST search:run valid\blast.py   <br>
