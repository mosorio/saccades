""" These simulations provide a proof of concept demonstration of the benefit of neural designs inspired by the
dorsal-ventral division of the primate visual system, the functional properties of posterior parietal cortex, and
learning abstractions grounded in action (here, eye movements) for zero-shot visual reasoning.

Author: Jessica Thompson
Date: 14/03/2023
Issues: N/A
"""
import os
import numpy as np
import pandas as pd
from itertools import product
import random

import torch

from config import get_config, get_base_name
from trainers import choose_trainer
from loaders import choose_loader
from models import choose_model
from utils import Timer
import json
from pathlib import Path

# def set_device(config):
#     """Specify the compute resource (CUDA, MPS, or CPU) to train model with."""
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = torch.device("mps")
#     elif torch.cuda.is_available() and not config.no_cuda:
#         device = torch.device(f"cuda:{config.gpu}")
#     else:
#         device = torch.device("cpu")
#     print(f'Using device: {device}')
#     return device

def set_device(config):
    """Specify the compute resource (CUDA or CPU) to train model with"""
    use_cuda = (not config.no_cuda) and torch.cuda.is_available()
    device = torch.device(f"cuda:{config.gpu}" if use_cuda else "cpu")
    print(f'Using device: {device}')
    return device


def main(config):
    timer = Timer()

    # make sure all results directories exist
    # model_dir = 'models/toy/letters'
    # results_dir = 'results/toy/letters'
    # fig_dir = 'figures/toy/letters'
    suffix = f"min{config.min_num}_max{config.max_num}_{config.n_glimpses}_{config.shape_input}_{config.head}_{config.task_type}_{config.count_mode}_seed{config.seed}_map_mode{config.map_mode}_bcew{config.bce_weight}_cew{config.ce_weight}_posw{config.pos_weight}"
    results_dir = f"results_{suffix}/logpolar"
    model_dir  = f"models_{suffix}/logpolar"
    fig_dir = f"figures_{suffix}/logpolar"

    config.results_dir = results_dir
    config.model_dir = model_dir
    config.fig_dir = fig_dir

    dir_list = [model_dir, results_dir, fig_dir]
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    # Load data, init model and trainer
    base_name = get_base_name(config)
    if os.path.isfile(f'{fig_dir}/accuracy_{base_name}.png') and config.if_exists != 'force':
        if config.if_exists == 'ask':
            ui = input(f"{base_name} exists. Would you like to increment rep counter(i), skip (s), or overwrite (o)? ")
            if ui == 'i':
                config.rep += 1
            elif ui == 's':
                print(f'Skipping {base_name}.')
                exit()
        elif config.if_exists == 'skip':
            print(f'{base_name} already exists. \n QUITTING.')
            exit()
    config.base_name = base_name

    seed = getattr(config, 'seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # loaders, test_xarray = choose_loader(config)
    loaders = choose_loader(config)
    config.height, config.width = loaders[0].image_height, loaders[0].image_width
    model = choose_model(config, model_dir)
    trainer = choose_trainer(model, loaders, config)

    # Train model and save trained model
    model, results = trainer.train_network()
    print('Saving trained model and results files...')
    # torch.save(model.state_dict(), f'{model_dir}/toy_model_{base_name}_ep-{config.n_epochs}.pt')
    model_file_name = f'{model_dir}/{base_name}_ep-{config.n_epochs}.pt'
    torch.save(model, model_file_name)
    print(f'model file: {model_file_name}')

    # Organize and save results

    if config.head == 'counting':
        (train_losses, train_accs, test_losses, test_accs, confs, test_results, per_shape_stats) = results
    else:
        (train_losses, train_accs, test_losses, test_accs, confs, test_results) = results

    (train_num_losses, train_map_losses, train_shape_loss) = train_losses
    (train_acc_count, train_acc_dist, train_acc_all) = train_accs
    (train_count_num_loss, train_dist_num_loss, train_all_num_loss) = train_num_losses
    (train_count_map_loss, train_dist_map_loss, train_full_map_loss) = train_map_losses
    (test_num_losses, test_map_losses, test_shape_loss) = test_losses
    (test_acc_count, test_acc_map, test_acc_dist, test_acc_all) = test_accs
    (test_count_num_loss, test_dist_num_loss, test_all_num_loss) = test_num_losses
    (test_count_map_loss, test_dist_map_loss, test_full_map_loss) = test_map_losses

    if config.head == 'counting':
        (train_percls_loss, train_percls_acc, train_present_loss, train_absent_loss,
        test_percls_loss, test_percls_acc, test_present_loss, test_absent_loss) = per_shape_stats

    # train_loss, train_acc, train_num_loss, train_shape_loss, train_full_map_loss, train_count_map_loss, test_loss, test_acc, test_num_loss, test_shape_loss, test_full_map_loss, test_count_map_loss, conf, test_results = results
    test_results.to_pickle(f'{results_dir}/test_results_{base_name}.pkl')
    df_train = pd.DataFrame()
    df_test_list = [pd.DataFrame() for _ in range(4)]
    # df_train['loss'] = train_loss
    # df_train['map loss'] = train_map_loss
    df_train['full map loss'] = train_full_map_loss
    df_train['dist map loss'] = train_dist_map_loss
    df_train['count map loss'] = train_count_map_loss
    df_train['count num loss'] = train_count_num_loss
    df_train['dist num loss'] = train_dist_num_loss
    df_train['all num loss'] = train_all_num_loss
    df_train['shape loss'] = train_shape_loss
    df_train['accuracy count'] = train_acc_count
    df_train['accuracy dist'] = train_acc_dist
    df_train['accuracy all'] = train_acc_all
    df_train['epoch'] = np.arange(config.n_epochs + 1)

    if config.head == 'counting':
        df_train['percls loss'] = list(train_percls_loss)
        df_train['percls acc'] = list(train_percls_acc)
        # df_train['percls strict acc'] = list(train_percls_strict_acc)
        # df_train['percls precision'] = list(train_percls_precision)
        # df_train['percls recall'] = list(train_percls_recall)
        # df_train['percls specificity'] = list(train_percls_specificity)
        df_train['present loss'] = train_present_loss
        df_train['absent loss'] = train_absent_loss

    # df_train['rnn iterations'] = config.n_iters
    df_train['dataset'] = 'train'
    _, test_loaders = loaders
    # for ts, (test_shapes, test_lums) in enumerate(product(config.test_shapes, config.lum_sets)):
    for ts, loader in enumerate(test_loaders):
        # df_test_list[ts]['loss'] = test_loss[ts]
        df_test_list[ts]['count num loss'] = test_count_num_loss[ts]
        df_test_list[ts]['dist num loss'] = test_dist_num_loss[ts]
        df_test_list[ts]['all num loss'] = test_all_num_loss[ts]
        df_test_list[ts]['shape loss'] = test_shape_loss[ts]
        # df_test_list[ts]['map loss'] = test_map_loss[ts]
        df_test_list[ts]['full map loss'] = test_full_map_loss[ts]
        df_test_list[ts]['count map loss'] = test_count_map_loss[ts]
        df_test_list[ts]['dist map loss'] = test_count_map_loss[ts]
        df_test_list[ts]['accuracy count'] = test_acc_count[ts]
        df_test_list[ts]['accuracy map'] = test_acc_map[ts]
        df_test_list[ts]['accuracy dist'] = test_acc_dist[ts]
        df_test_list[ts]['accuracy all'] = test_acc_all[ts]
        df_test_list[ts]['dataset'] = loader.testset
        df_test_list[ts]['viewing'] = loader.viewing
        df_test_list[ts]['test shapes'] = str(loader.shapes)
        df_test_list[ts]['test lums'] = str(loader.lums)
        df_test_list[ts]['epoch'] = np.arange(config.n_epochs + 1)

        if config.head == 'counting':
            df_test_list[ts]['percls loss'] = list(test_percls_loss[ts])
            df_test_list[ts]['percls acc'] = list(test_percls_acc[ts])
            # df_test_list[ts]['percls strict acc'] = list(test_percls_strict_acc[ts])
            # df_test_list[ts]['percls precision'] = list(test_percls_precision[ts])
            # df_test_list[ts]['percls recall'] = list(test_percls_recall[ts])
            # df_test_list[ts]['percls specificity'] = list(test_percls_specificity[ts])
            df_test_list[ts]['present loss'] = test_present_loss[ts]
            df_test_list[ts]['absent loss'] = test_absent_loss[ts]

    np.save(f'{results_dir}/confusion_{base_name}', confs)
    if config.save_batch_confusion:
        np.save(f'{results_dir}/batch_confusion_{base_name}', trainer.batch_confusion)

    df_test = pd.concat(df_test_list)
    # df_test['rnn iterations'] = config.n_iters
    df = pd.concat((df_train, df_test))
    df.to_pickle(f'{results_dir}/results_{base_name}.pkl')
    timer.stop_timer()


if __name__ == '__main__':
    """ Example use from command line:
    $  python3 main.py --model_type=pretrained_ventral-cnn-ce-sm --sort --train_on=xy --use_loss=num --opt=SGD --target_type=all --solarize --same --challenge=distract --max_pass=13 --n_glimpses=12 --h_size=1024 --shape_input=noise --min_num=1 --max_num=5 --train_shapes=ESUZ --test_shapes ESUZ FCKJ --noise_level=0.9 --train_size=100000 --test_size=5000 --n_epochs=100 --min_pass=0 --act=lrelu --dropout=0.5 --rep=0 --grid=6 --ventral='ventral_ventral_cnn-lrelu_hsize-25_noise_num1-5_nl-0.9_diff-0-8_grid6_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-ESUZFCKJsame_distract_gw6_100000_loss-ce_opt-Adam_drop0.5_sort_300eps_rep0_ep-300.pt'

    """
    config = get_config()

    # config_dir = Path("/Users/mosorio/Documents/ChrisWork/saccades/datasets/config_dir") 
    config_dir = Path("/mnt/quick/maria/saccades/datasets/config_dir")
    config_dir.mkdir(parents=True, exist_ok=True) 
    
    config_path = config_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, sort_keys=True)

    config.device = set_device(config)
    main(config)