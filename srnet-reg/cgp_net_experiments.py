"""Experiment codes for CGPNet training."""
import datetime
import json

import joblib
import numpy as np
from joblib import Parallel, delayed

from CGPNet.config import clas_net_map, clas_cgp_map, clas_optim_map
from CGPNet.functions import default_functions
from CGPNet.methods import Evolution
from data_utils import io, draw
from dataset_config import VALID_MAP, vars_map
from exp_utils import save_cfs, draw_f_trend, individual_to_dict, generate_domains_data
from neural_networks.nn_models import NN_MAP


# where is the dataset
data_dir, xlabel = '/home/luoyuanzhen/STORAGE/dataset/pmlb/', 'F'
# what logs and imgs directory you want to save the result
log_dir, img_dir = 'cgpnet_result/if_logs/', 'cgpnet_result/if_imgs/'

io.mkdir(log_dir)
io.mkdir(img_dir)

# hyperparameters for srnet, note that some of them are useless, for the result consistense do not change
evo_params = {
    'clas_net': 'OneVectorCGPNet',  # do not change
    'clas_cgp': 'OneExpOneOutCGPLayer',  # do not change
    'optim': 'Newton',  # Newton-Rapson optimization method do not change
    'n_rows': 5,  # rows of function nodes in each CGP
    'n_cols': 5,  # cols of function nodes in each CGP
    'levels_back': None,
    'function_set': default_functions,
    'n_eph': 1,  # number of constant added in each CGP
    'add_bias': True,  # do not change

    'n_population': 200,  # population size in each generation
    'n_generation': 5000,  # number of evoled generation
    'prob': 0.4,  # point mutation prob
    'verbose': 10,  # 0 would not be reported
    'stop_fitness': 1e-5,
    'random_state': None,
    'n_epoch': 0,  # useless, but do not delete
    'end_to_end': False,  # do not change
    'validation': True,  # do not change
    'evolution_strategy': 'chromosome_select'  # do not change
}

# name of datasets
all_names = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
             'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']

# how many times you want to run for each dataset
run_times = 30


def _train_process(controller, trainer, data_list, msg, valid_data_list):
    print(msg)
    start_time = datetime.datetime.now()
    elites, convf = controller.start(data_list=data_list,
                                     trainer=trainer,
                                     valid_data_list=valid_data_list
                                     )
    end_time = datetime.datetime.now()

    return elites, convf, (end_time - start_time).seconds / 60


def run_srnet_experiments(evo_params, all_names, data_dir, log_dir, img_dir, xlabel=None, run_n_epoch=30):
    trainer = clas_optim_map[evo_params['optim']](end_to_end=evo_params['end_to_end'])
    srnn_fs_list = []
    for fname in all_names:
        var_names = vars_map[fname][0] if fname in vars_map else None

        # run dcgp net without save
        nn_dir = f'{data_dir}{fname}_nn/'
        nn_data_list = io.get_nn_datalist(nn_dir)

        valid_data_list = None
        if evo_params['validation']:
            # generate extrapolation data for model selection
            nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[fname]).cpu()

            num_sample = max(nn_data_list[0].shape[0] // 10 * 3, 10)  # train:valid = 7:3
            valid_input = generate_domains_data(num_sample, VALID_MAP[fname])

            valid_data_list = [valid_input] + list(nn(valid_input))

        clas_net, clas_cgp = clas_net_map[evo_params['clas_net']], clas_cgp_map[evo_params['clas_cgp']]
        controller = Evolution(evo_params=evo_params,
                               clas_net=clas_net,
                               clas_cgp=clas_cgp)
        results = Parallel(n_jobs=run_n_epoch)(
            delayed(_train_process)(controller,
                                    trainer,
                                    nn_data_list,
                                    f'{fname}-{epoch} start:\n',
                                    valid_data_list
                                    )
            for epoch in range(run_n_epoch))

        srnn_fs, srnn_ts = [], []  # for log
        srnn_cfs = []  # for trend draw
        elites = []  # All top10 best elites from each runtimes. For log
        for result in results:
            process_elites, convf, time = result
            srnn_fs.append(process_elites[0].fitness)
            srnn_ts.append(time)
            srnn_cfs.append(convf)
            elites += process_elites[:min(10, len(process_elites))]

        elites.sort(key=lambda x: x.fitness)

        srnn_fs_list.append(srnn_fs)

        log_dict = {
            'name': fname,
            'evolution_parameters': evo_params,
            'neurons': list([data.shape[1] for data in nn_data_list]),
            'srnn_mean_time': np.mean(srnn_ts),
            'srnn_mean_fitness': np.mean(srnn_fs),
            'srnn_min_fitness': np.min(srnn_fs),
            'srnn_max_fitness': np.max(srnn_fs),
            'srnn_fitness': srnn_fs,
        }

        elite_results = Parallel(n_jobs=joblib.cpu_count())(
            delayed(individual_to_dict)(elite, var_names)
            for elite in elites)

        for num, result in enumerate(elite_results):
            log_dict[f'elite[{num}]'] = result

        with open(f'{log_dir}{fname}_30log.json', 'w') as f:
            json.dump(log_dict, f, indent=4)

        save_cfs(f'{log_dir}{fname}_30cfs', srnn_cfs)
        draw_f_trend(f'{img_dir}{fname}_trend.pdf', evo_params['n_generation'], [srnn_cfs], legends=['srnn'], title=fname)

    draw.draw_fitness_box(f'{img_dir}{xlabel}_box_fit.pdf', srnn_fs_list, xlabel=xlabel)


run_srnet_experiments(evo_params, all_names, data_dir, log_dir, img_dir, xlabel, run_n_epoch=run_times)






