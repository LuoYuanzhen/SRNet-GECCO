# SRNet for Classification Task
This is a project for SRNet to explore hidden semantics for neural networks (MLPs) classifier. 

Note again that it is easy to combine both classification task and regression task in a single project. However,
for the simplicity of experimental analysis, we divide them into two projects (i.e. namely srnet-clas and srnet-reg).
## Content
srnet-clas/neural_networks: all the MLP models and training methods used in our paper. You may train your own MLP models
to test our SRNet

srnet-clas/CGPNet: package of our SRNet model

srnet-clas/dataset: all classification dataset used in our paper. **Note**: You may train your own MLP models using
pmlb dataset and to get better result.

srnet-clas/result: experiment figures used in our paper.
## How to Run
### Training MLPs
For example, run

`python neural_networks/train_clas.py --dataname adult`

to training classifier MLP for 'adult' dataset.
### Training SRNet
`python run_all.py --start_trial 1 --end_trial 30 --device cpu`

This can easily run all classification experiments in our paper for 30 times. Note that GPU version has not 
been tested yet.

### Hidden Semantics (Mathamatical Expressions)
All the hidden semantics would be stored at `result/` as json file after training the SRNet. Here we show our
experiments semantics in our paper:

![Hidden Semantics](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/semantics.png)

## Compare to LIME and MAPLE
### Training LIME and MAPLE
`python -u run_maple_lime.py --start_trial 1 --end_trial 30`

Easily run LIME and MAPLE for all classification tasks 30 times. We can get the training accuracy and testing accuracy
for LIME and MAPLE.

### Decision Boundary
After you run SRNet, LIME and MAPLE for all classification datasets, you can run `analysis/compare_DB.py` to see 
each desicion boundary (DB) for 3 models. For example, in our paper we show that, the DB of both LIME
and MAPLE are all linear in local area while the DB of our SRNet is non-linear in globe area:

![DB of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/local_db.png)

Note that this figure is edited after running the code for better view. Also, it is possible for this code to raise a
warning due to the LIME and MAPLE may fail to find a DB line in a local area (e.g. the points in the selected local area
are all in the same class). You may run serveral times to get the similar figure in our paper.
### Accuracy
It is easy to obtain the mean accuracy (acc) for these models: simply run `analysis/compare_accuracy.py` to get
the mean training acc and testing acc of SRNet for all datasets.

we show the training acc and testing acc in all classification tasks in our paper:

![Acc of SRNet vs. LIME vs. MAPLE](https://github.com/LuoYuanzhen/SRNet-GECCO/blob/master/srnet-clas/result/accs.png)

