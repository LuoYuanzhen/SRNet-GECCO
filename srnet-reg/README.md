# SRNet for Regression Task
This is a project for SRNet to explore hidden semantics for neural networks (MLPs) classifier. 

Note again that it is easy to combine both classification task and regression task in a single project. However,
for the simplicity of experimental analysis, we divide them into two projects (i.e. namely srnet-clas and srnet-reg).

## How to Run
### Training SRNet 
Simply run cgp_net_experiments.py to run 12 benchmarks in 'dataset/' 30 times. The selected elites in all independent
runs would be stored at 'cgpnet_result/' as json file; the convergence curve of fitness would be saved at
'cgpnet_result/' as pdf file.

You may get the convergence curve of fitness of SRNet that run 30 times on each MLP:
![The convergence curve of fitness](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/trend_result.png)
### Get Experimental Figures
After running the cgp_net_experiments.py, we can get all the experimental figures. 
## Interpolation/Extrapolation Curves/Points and Heat Map
Simply run 'analyse/best_draw.py' to get the experiment figures, including the fitting curves (or distribution points)
of SRNet in interpoaltion/extrapolation domain, the heat map of comparition of outputs of SRNet layer vs the NN layer.

For example, the fitting curves (or distribution points) of SRNet in interpolation/extrapolation domain of K0:
![The fitting curves of SRNet in extrapolation domain of K0](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/kkk0_curves_elite%5B0%5D.png)

The comparition heat map of outputs of SRNet layer vs the NN layer of K0 (9 random instances):
![The comparition heat map of outputs of SRNet layer vs the NN layer of K0](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/kkk0_elite%5B0%5D_0.png)

![The comparition heat map of outputs of SRNet layer vs the NN layer of K0](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/kkk0_elite%5B0%5D_1.png)

## Compare to LIME and MAPLE on Extrapolation Domain
Simply run compare_experimetns.py to get the best individual SRNet in 30 times runing vs LIME vs MAPLE on the interpolation
and extrapolation domain:
![SRNet vs LIME vs MAPLE in extrapolation domain](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/extrapolation.png)

## Get Hidden Semantics (Mathematical Expressions)
After running the cgp_net_experiments.py, simply run 'analyse/see_results.py' can get all the hidden semantics for all network layers.

You may get a table of mathematical expressions and their fitness at '/cgpnet_result/b_logs/semantics.csv':
![Table of hidden semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-reg/IMG/hidden_semantics.png)
