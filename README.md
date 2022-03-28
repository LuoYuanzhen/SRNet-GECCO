# 1. Introduction
A Symbolic Regression (SR) method called SRNet to mine hidden semantics of each network layer in Neural Network 
(Multi-Layer Perceptron). SRNet is an evolutionary computing algorithm, leveraging Multi-Chromosomes Cartesian Genetic 
Programming (MCCGP) to find mathmatical formulas $f_i(x)*w_i+b_i$ for each network layer, white-boxing the black box.

This paper has been accepted by GECCO-22, see our paper for more.
# 2. Code
## Projects
Note that for the simplicity of experimental analysis, we divide the SRNet into 2 projects, namely 
**srnet-clas** and **srnet-reg**, for classification task and regression task respectively. It is easy
to combine both projects into one single project since the code of SRNet (package at `srnet-clas/CGPNet` 
or `srnet-reg/CGPNet`) is easy to implement for both classification task and regression task.

## Requirements
Make sure you have installed the following pacakges before start running our code:

- pytorch 1.8.1
- sympy 1.8
- numpy 1.21.0
- joblib 1.0.1

Our experiments were running in Ubuntu 18.04 with Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz and RTX3090. 
The python version is 3.9

# 3. Experiments
For both regression and classification task, see `srnet-clas/README.md` and `srnet-reg/README.md` for
more details about how to reproduce our experimental results.

Here we show our experimental figures in our paper:
## Convergence 
The convergence curves for all dataset:
![Regression convergence curve of fitness](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/trend_result.png)
![Classification convergence curve of fitness](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/trend_result.png)

## Semantics (Mathematical Expressions)
![Hidden Semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/hidden_semantics.png)

Combining these expressions, we can obtain the overall expressions for all NNs:
![Semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/semantics.png)

## Comparison
We compare SRNet to LIME and MAPLE on both regression and classification tasks.

### Regression
![Regression Comparison](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/IMG/reg_comparison.png)

### Classifiation
Decision boundary:
![DB of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/local_db.png)

Accuracy:
![Acc of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/accs.png)

# 4. Cite
Please **cite** our paper if you use the code.
