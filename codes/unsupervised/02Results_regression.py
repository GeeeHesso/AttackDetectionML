# -*- coding: utf-8 -*-


# %% [0] PACKAGES
import os
import pickle
# import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from os.path import join as pjoin

# os.chdir('..')  # Working directory in 10_Armasuisse2024/codes
os.chdir(os.getcwd().split('10_Armasuisse2024')[0] + '10_Armasuisse2024/codes')
import sys
sys.path.append(os.getcwd())
from functions import load_models, get_gen_names, get_p_nom


# MATPLOTLIB PARAMETERS
from pylab import rcParams
rcParams['figure.figsize'] = 8, 3
rcParams['figure.dpi'] = 400
plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-v0_8-dark-palette')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors.append('#FF6347')

""" 
02Results_regression.py :  Compile all regression results from 
						   01GSCV.py armasuisse 2023 WP2.
                                    
"""


# %% [1] PARAMETERS
os.chdir('..')  # Working directory in 10_Armasuisse2024

case = 'CH'

nets_dict = [case]  # List with all nets
models_dict = load_models()  # Dictionary with all regression models
models_dict = {'mlpr': models_dict['mlpr']}  # Dictionary with all regression models
# models_dict = {}  # Dictionary with all regression models for debug lstm only
models_dict['lstm'] = {}  # From google colab

types_dict = ['generation', 'injection']  # List with all dataset type
# types_dict = ['generation']
sequence_lens = [4, 24]
contextual_lens = ['t', 'hist']  # W/O historical values for contextual variable
# contextual_lens = ['hist']  # W/O historical values for contextual variable

# sorted_nodes = ['Innertkirchen', 'Sils', 'Löbbia', 'Tavanasa', 'Sedrun', 'Pradella',
#                 'Rothenbrunnen', 'Riddes', 'Stalden', 'Cavergno']


# %% [2] LOAD RESULTS DF
cartesian = product(nets_dict, models_dict.keys(), types_dict,
                    sequence_lens, contextual_lens)

net_key_old = ''
result = pd.DataFrame()

hyper_idx = ['case', 'model', 'node', 'ds_type', 'sequence', 'contextual', 'params']
hyper_all = pd.DataFrame(index=hyper_idx)
learning_idx = ['case', 'model', 'node', 'ds_type', 'sequence', 'contextual',
                'iter (epochs)', 'loss', 'best_loss']
learning_mlpr = pd.DataFrame(index=learning_idx)
for net_key, model_key, ds_type, seq, contextual in cartesian:
    
    if net_key != net_key_old:  # Avoid doing same things 
        net_key_old = net_key
        
        ds_path = pjoin('datasets', net_key)
        res_dir = pjoin('results', 'unsupervised', net_key, 'regression')
        
        attacked_gens = pd.read_pickle(pjoin(ds_path, 'attacked_gens.p'))

    
    # [2.1] ANOMALY LOAD LOOP
    for i, attack_gen in enumerate(attacked_gens):
        res_path = pjoin(res_dir,
                          f'{model_key}',
                          ds_type,
                          f'{attack_gen}',
                          f'sequence_len-{seq}',
                          f'contextual_{contextual}',
                          )
        if not os.path.isdir(res_path): continue  ## GSCV not finish
        
        
        # [2.2] LOAD HYPERPARAMETERS
        if model_key == 'mlpr':
            f_path = pjoin(res_path,  'gscv_best_params.p')
            if os.path.isfile(f_path):
                params = pickle.load(open(f_path, 'rb'))
            
            hyper = pd.DataFrame([net_key, model_key, attack_gen, ds_type,
                                  seq, contextual, params], index=hyper_idx)
            hyper_all = pd.concat([hyper_all, hyper], axis=1)
            
            
            # [2.2] LOAD LEARNING
            estimator = pickle.load(open(pjoin(res_path, 'gscv_trained.p'), 'rb'))
            
            learning = pd.DataFrame(
                [net_key, model_key, attack_gen, ds_type,
                 seq, contextual, estimator.n_iter_, estimator.loss_, estimator.best_loss_,
                 ], index=learning_idx)
            learning_mlpr = pd.concat([learning_mlpr, learning], axis=1)
            
        
        # [2.3] LOAD RESULTS
        f_path = pjoin(res_path, 'gscv_regression_metrics.p')
        if not os.path.isfile(f_path):
            continue
            
        df = pd.read_pickle(f_path).reset_index()  # Load results
        
        df['net'] = net_key
        df['attack_gen'] = attack_gen
        df['model'] = model_key
        df['ds_type'] = ds_type
        df['sequence'] = seq
        df['contextual'] = contextual
        
        result = result.append(df, ignore_index=True)

result = result.rename(columns={'index': 'set'})
print(result.columns)
print('all fits:', result.shape[0], '\n')

# cols = ['net', 'attack_gen', 'model', 'ds_type',
#         'fn_rate', 'f2_score', 'f5_score', 'tn', 'fp', 'fn', 'tp', 'test_hacked',
#         'test_occ', 'train_size', 'train_sample', 'train_hacked', 'train_occ']

# result = result[cols]

result.sort_values(by='rmse', ascending=True, inplace=True)
result = result.round(3)

# normalized rmse
result['normalized_rmse'] = result.rmse / result.attack_gen.replace(get_p_nom(case))

# Rename gen
result.attack_gen = result.attack_gen.replace(get_gen_names(case))


learning_mlpr = learning_mlpr.T
learning_mlpr['delta'] = learning_mlpr.loss-learning_mlpr.best_loss

hyper_all = hyper_all.T


# %% [4] MLPR LEARNING ANALYSIS
# for ds_type, df_type in learning_mlpr.groupby('ds_type'):
#     fig, ax = plt.subplots(figsize=(6,1))
#     df_type['iter (epochs)'].plot.box(vert=False, ax=ax)
#     ax.set_yticklabels('')
#     ax.set(ylabel='epochs', title=ds_type, xlim=(0,201),)

fig, ax = plt.subplots(figsize=(8,1.5))
learning_mlpr['iter (epochs)'].plot.box(vert=False, ax=ax)
ax.set_yticklabels('')
ax.set(
       xlabel="Nombre d'époques",
       xlim=(0,201),
       # title='MLPR',
       yticks=())
fig.tight_layout()
fig.savefig(pjoin('figures', 'unsupervised_results_regression', f'{case}_learning.pdf'), dpi=600)
plt.close(fig)



# %% [5] MLPR ANALYSIS
result_net = result[result.net == case]  # Useless for the moment
result_mlpr = result_net[result_net.model == 'mlpr']


# [5.1] METRICS BY DATASET TYPES
for metric in ['r²', 'rmse', 'normalized_rmse']:
    for param in ['ds_type', 'sequence', 'contextual']:
        ds_param = pd.DataFrame()
        for col, df in result_mlpr.groupby(param):
            # ds_param[col] = df[metric].values
            ser = df[metric]
            if col == 't':
                ser.name = 'Time t only'
            elif col == 'hist':
                ser.name = 'History with\n 4 or 24 steps'
            elif col == 'injection':
                ser.name = ' Generation\n + loads'
            elif col == 'generation':
                ser.name = ' Generation\n only'
            else:
                ser.name = col
            ds_param = pd.concat([ds_param, ser], axis=1)
        
        print(ds_param.shape)
        fig, ax = plt.subplots(figsize=(6,1.8))  # metric comparison by param
        ds_param.plot.box(ax=ax, vert=False)
        # ax.set(xlabel=metric)
        if metric=='r²': ax.set(xlabel=metric)
        if metric=='rmse': ax.set(xlabel="RMSE [MW]")
        if metric=='normalized_rmse': ax.set(xlabel="Relative error", xlim=0)
        if param == 'sequence': ax.set(ylabel='History length')
        if param == 'contextual': ax.set(ylabel='Context')
        if param == 'ds_type': ax.set(ylabel='Injections')
        if metric == 'normalized_rmse': ax.set(xlim=(0,0.15))
        fig.tight_layout()

        metric_name = metric
        if metric_name == 'r²': metric_name = 'r2'
        fig.savefig(pjoin('figures', 'unsupervised_results_regression',
                          f'{case}_mlpr_{param}_{metric_name}.pdf'), dpi=600)
        plt.close(fig)


# [5.2] METRICS BY NODE
for metric in ['r²', 'rmse', 'normalized_rmse']:
    df_node = pd.DataFrame()
    for col, df in result_mlpr.groupby('attack_gen'):
        # df_node[col] = df[metric].values
        ser = df[metric]
        ser.name = col
        df_node = pd.concat([df_node, ser], axis=1)

    sorted_nodes = list(df_node.mean().sort_values(ascending=False).index)
    df_node = df_node[sorted_nodes]
    
    print(df_node.shape)
    fig, ax = plt.subplots()  # metric comparison by param
    df_node.plot.box(ax=ax, vert=False)
    # ax.set(xlabel=metric)
    if metric=='r²': ax.set(xlabel=metric, xlim=(.8,1), title='MLPR')
    if metric=='rmse': ax.set(xlabel='RMSE [MW]', xlim=0, title='MLPR')
    if metric=='normalized_rmse': ax.set(xlabel='Relative error', xlim=0, title='MLPR')
    fig.tight_layout()

    metric_name = metric
    if metric_name == 'r²': metric_name = 'r2'
    fig.savefig(pjoin('figures', 'unsupervised_results_regression',
                      f'{case}_mlpr_{metric_name}.pdf'), dpi=600)
    plt.close(fig)


# %% [6] LSTM ANALYSIS
result_lstm = result_net[result_net.model == 'lstm']


# [6.1] METRICS BY DATASET TYPES
for metric in ['r²', 'rmse', 'normalized_rmse']:
    for param in ['ds_type']:
        ds_param = pd.DataFrame()
        for col, df in result_lstm.groupby(param):
            # ds_param[col] = df[metric].values
            ser = df[metric]
            ser.name = col
            ds_param = pd.concat([ds_param, ser], axis=1)

        print(ds_param.shape)
        fig, ax = plt.subplots()  # metric comparison by param
        ds_param.plot.box(ax=ax, vert=False)
        # ax.set(xlabel=metric)
        if metric=='r²': ax.set(xlabel=metric,)
        if metric=='rmse': ax.set(xlabel='RMSE [MW]',)
        if metric=='normalized_rmse': ax.set(xlabel='Relative error', xlim=0)
        fig.tight_layout()

        metric_name = metric
        if metric_name == 'r²': metric_name = 'r2'
        fig.savefig(pjoin('figures', 'unsupervised_results_regression',
                          f'{case}_lstm_{param}_{metric_name}.pdf'), dpi=600)
        plt.close(fig)


# [6.2] METRICS BY NODE
for metric in ['r²', 'rmse', 'normalized_rmse']:
    df_node = pd.DataFrame()
    for col, df in result_lstm.groupby('attack_gen'):
        # df_node[col] = df[metric].values
        ser = df[metric]
        ser.name = col
        df_node = pd.concat([df_node, ser], axis=1)
    df_node = df_node[sorted_nodes]

    # df_node = df_node[sorted_nodes]
    print(df_node.shape)
    fig, ax = plt.subplots()  # metric comparison by param
    df_node.plot.box(ax=ax, vert=False)
    # ax.set(xlabel=metric)
    # ax.set(title=metric)
    if metric=='r²': ax.set(xlabel=metric, xlim=0)
    if metric=='rmse': ax.set(xlabel='RMSE [MW]', xlim=0)
    if metric=='normalized_rmse': ax.set(xlabel='Relative error', xlim=0)
    ax.set(title='LSTMR')
    fig.tight_layout()

    metric_name = metric
    if metric_name == 'r²': metric_name = 'r2'
    fig.savefig(pjoin('figures', 'unsupervised_results_regression',
                      f'{case}_lstm_{metric_name}.pdf'), dpi=600)
    plt.close(fig)


# %% [6.2.1] METRICS BY NODE & DATASET
# df_node = pd.DataFrame()
# for metric in ['r²', 'rmse']:
#     for col, df in result_lstm.groupby('attack_gen'):
#         # df1 = df[df.ds_type=='injection']
#         df1 = df[df.ds_type=='generation']
#         df_node[col] = df1[metric].values
#
#     print(df_node.shape)
#     fig, ax = plt.subplots()  # metric comparison by param
#     df_node.plot.box(ax=ax, vert=False)
#     # ax.set(xlabel=metric)
#     if metric=='r²': ax.set(xlabel=metric, xlim=(.8,1))
#     if metric=='rmse': ax.set(xlabel=metric, xlim=(0,40))
#     fig.tight_layout()
#
#     metric_name = metric
#     if metric_name == 'r²': metric_name = 'r2'
#     fig.savefig(pjoin('figures', 'unsupervised_results_regression',
#                       f'{case}_lstm_{metric_name}_xxx.pdf'), dpi=600)
#     plt.close(fig)


# %% [7.1] MLPR vs LSTM
## GLOBAL

metric_dict = {
    # 'r²': 'coéfficient de détermination (r²)',
    'rmse': 'Root mean square error [MW]',
    'normalized_rmse': 'Relative error'
}

model_dict ={'mlpr': 'MLPR', 'lstm':'LSTMR',}

for set_type, df_set in result.groupby('set'):
    for metric in metric_dict.keys():
        data = dict()
        for model, df in df_set.groupby('model'):
            data[model] = df[metric].tolist()
            # print(set_type, model, df.shape)

        data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
        data = data.rename(columns=model_dict)
        # print(data.median())
        fig, ax = plt.subplots(figsize=(8, 2.5))
        data.plot.box(ax=ax, vert=False)
        ax.set(xlabel=metric_dict[metric])
        ax.set(title=set_type)
        if metric == 'rmse': ax.set(xlim=0)
        if metric == 'normalized_rmse': ax.set(xlim=0)

        ax.set(xlabel=metric_dict[metric])
        fig.tight_layout()

        if set_type == 'validation set':
            metric_name = metric
            if metric_name == 'r²': metric_name = 'r2'
            fig.savefig(pjoin('figures', 'unsupervised_results_regression',
                              f'{case}_mlpr-lstm_{metric_name}.pdf'), dpi=600)
            plt.close(fig)

            # if set_type=='validation set' and metric=='rmse':
            #     figname = 'regression_lstm-mlpr_rmse_val-set.svg'
            #     plt.savefig(pjoin('figures', figname), dpi=600)


# # %% BEST by node

# # =============================================================================
# # à améliorer pour avoir les barres d'erreur min et max correctes !!
# # =============================================================================

# medians = {key: list() for key in result_net.model.unique()}
# for node, df_node in result.groupby('attack_gen'):
    
#     for model, df_model in df_node.groupby('model'):
        
#         medians[model].append(df_model.rmse.median())
    
# medians = pd.DataFrame(medians, index=result_net.attack_gen.unique())
# medians = medians.sort_values('lstm', ascending=False)

# fig, ax = plt.subplots(figsize=(8,3.5))  # f2_score comparison by model and node
# # medians.plot.bar(ax=ax, color=colors, yerr=np.array([medians*.1, medians*.5]))
# medians.plot.bar(ax=ax, color=colors, yerr=np.array([medians.T*.1, medians.T*.5]))
# ax.set(ylabel='rmse')
# ax.set(title="à améliorer pour avoir les barres d'erreur min et max correctes !!")
# # ax.set(ylabel='rmse', ylim=(0,1))
# ax.tick_params(axis='x', labelrotation=20)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#           ncol=len(medians.columns))
# # ax.legend(ncol=len(medians.columns))
# fig.tight_layout()
