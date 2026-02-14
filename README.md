# ProSTR
ProSTR is a two-step conditional generation framework for designing high-activity promoters by incorporating STR priors. <br>
In Step 1, it constructs and selects high-scoring core scaffolds that jointly contain TFBS and STR elements. In Step 2, it fills the upstream and downstream flanking regions under scaffold constraints to generate full-length promoter sequences. Finally, an independently trained expression-activity predictor is used to score and rank the completed sequences, yielding high-potential candidates. This staged strategy improves structural controllability, screening efficiency, and overall stability.

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

* ecoli_generation.csv      The data used by the generator <br>
* ecoli_prediction.csv      The data used by the predictor <br>

# Design Promoter Sequence
We take design promoters in E.coli as an example,to illustrate how to train the ProSTR model and design the promoter sequences.
## 1.Training the generator
* run \Generator\step1_cGAN.py    <br>

>>Train the first-step conditional generator to learn the baseline sequence distribution and produce initial candidates.<br>

* run \Generator\str.py    <br>

>>Introduce and fill STR patterns in generated candidates to build an STR-enriched candidate set.<br>

* run \Generator\loss.py   <br>

>>Compute similarity loss between generated and natural sequences for quality-aware filtering.<br>

* run \Generator\step2_cGAN.py <br>

>>Train the second-step generator on the filtered set to further improve sequence quality and functional characteristics.<br>
## 2. Training the predictor (can be done in advance)

run \prediction\GRU.py <br>
>>This step trains the activity predictor.<br>
A pretrained predictor is needed before generation and reused as the scoring module in both stages. It scores candidates after STR insertion and flanking completion, selects high-activity sequences for stage-two refinement, and filters final outputs by predicted activity, yielding a consistent and stable selection process.<br>
## 3.Evaluate the performance of ProSTR model generated sequences

* DNAshape:run valid\dnashape.py  <br>

* GC:run valid\GCviolin.py   <br>

* Diversity:run valid\edit1.py  run valid\geditdistence.py   <br>

* K-mer:run valid\kmer.py    <br>

* BLAST search:run valid\blast.py   <br>

