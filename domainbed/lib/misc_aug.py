# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from shutil import copyfile

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import tqdm

from domainbed.lib.misc import (PredictiveEntropy, Tee, accuracy,
                  calculate_cosine_similarity_loss, compare_models, load_obj,
                  make_weights_for_balanced_classes, print_row,
                  print_separator, random_pairs_of_minibatches,
                  save_obj_with_filename, seed_hash)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    # if len(dataset) > 1:
    # in_sp = [[]]*len(dataset)
    # out_sp = [[]]*len(dataset)
    # assert(n <= len(dataset[0]))
    # keys = list(range(len(dataset)))
    # np.random.RandomState(seed).shuffle(keys)
    # keys_1 = keys[:n]
    # keys_2 = keys[n:]
    # for i in range(len(dataset)):
    #     out_sp[i], in_sp[i]= _SplitDataset(dataset[i], keys_1), _SplitDataset(dataset[i], keys_2)
    # return out_sp, in_sp[0]
    # else:
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)




def MULDENS_beta_grads(loaders, test_env_loader, model_chosen, device):
    test_env_grads = []
    
    for x, y in test_env_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model_chosen(x)
        loss = F.cross_entropy(pred, y)
        test_env_grads.append(torch.autograd.grad(loss, model_chosen.parameters(), allow_unused=True))
    
    test_grads=[]
    for i in range(len(list(model_chosen.parameters()))):
        corresponding = []
        for l in test_env_grads:
            corresponding.append(l[i])
        test_grads.append(torch.mean(torch.stack((corresponding)), dim = 0))

    loader_iterator = zip(*loaders)
    env_grads = []
    for step in range(32000):
        try:
            minibatches = [(x.to(device), y.to(device)) for x,y in next(loader_iterator)] 
            all_x = torch.cat([x for x,y in minibatches])
            all_y = torch.cat([y for x,y in minibatches])
            pred = model_chosen(all_x)
            loss = F.cross_entropy(pred, all_y)
            env_grads.append(torch.autograd.grad(loss, model_chosen.parameters(), allow_unused=True))
        except Exception as e:
            print(e)
            break
    fenv_grads=[]
    for i in range(len(list(model_chosen.parameters()))):
        corresponding = []
        for l in env_grads:
            corresponding.append(l[i])
        fenv_grads.append(torch.mean(torch.stack((corresponding)), dim = 0))
    return calculate_cosine_similarity_loss(fenv_grads,test_grads)
def ensemble_accuracy(networks, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    [network.eval() for network in networks]
    predictions_=[]
    pred_entropies_all=[]
    max_probs= []
    labels_=[]
    entropy = PredictiveEntropy()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            p = [network(x) for network in networks]

            p_mean = torch.mean(torch.stack(p),dim =0)
            
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p_mean.size(1) == 1:
                correct += (p_mean.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p_mean.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            predictions_.append(p_mean.detach().cpu().numpy())
            labels_.append(y.detach().cpu().numpy())
            pred_entropies_all.append(entropy(p_mean).detach().cpu().numpy())

    [network.train() for network in networks]
    return_dict={}
    return_dict['acc']=correct / total
    return_dict['labels']= np.concatenate(labels_)
    return_dict['preds']= np.concatenate(predictions_)
    return_dict['pred_entropies']= np.concatenate(pred_entropies_all)
    return return_dict
def MULDENS_accuracy(algorithm,eval_dict, test_envs,correct_models_selected_for_each_domain,device,acc_flags):
    compute_test_beta=acc_flags['compute_test_beta'] # setting this to false will give you ensemble
    ensemble_for_obs= acc_flags['ensemble_for_obs']
    correct = 0
    total = 0
    weights_offset = 0 
    eval_loader_names= list(eval_dict.keys())
    

    beta_all =[]
    test_env = test_envs[0]

    # obs_loader_insplit_names = ['env{}_in'.format(i)
    #     for i in range(len(eval_loader_names)//2) if i not in test_envs]
    # obs_loader_outsplit_names= ['env{}_out'.format(i)
    #     for i in range(len(eval_loader_names)//2) if i not in test_envs]

    # un_obs_insplit_name = ['env{}_in'.format(i) for i in test_envs]
    # un_obs_outsplit_name = ['env{}_out'.format(i) for i in test_envs]

    
    for network_i in algorithm.MULDENS_networks:
        network_i.eval()

    
    domains_selected_for_each_model=  [[] for i in range(len(algorithm.MULDENS_networks))]
    model_domains = []
    for model in range(len(algorithm.MULDENS_networks)):
        for i,ms in enumerate(correct_models_selected_for_each_domain):
            if ms is not np.nan:
                if ms == model:
                    domains_selected_for_each_model[model].append(i)
        

    
    # for observed domains, we know what models to select. 
    # So directly get the accuracies from corresponding model
    if ensemble_for_obs:# ensemble and individual for both observed and unobserved domains
        results ={}
        for i in range(len(eval_loader_names)//2):
            # for split in ['_in','_out']:

            for split in ['_out0']:

                
                name = 'env'+str(i)+split
                loader= eval_dict[name][0]
                weights= eval_dict[name][1]
                if i in test_envs: 
                    name = 'unobs_'+'env'+str(i)+'_in0' 
                    loader = eval_dict['env'+str(i)+'_in0'][0]# for test env we need 'in' not 'out
                    weights= eval_dict['env'+str(i)+'_in0'][1]
                for m in range(len(algorithm.MULDENS_networks)):
                    acc= accuracy(algorithm.MULDENS_networks[m],loader,weights,device)
                     
                    results[name+'_m_'+str(m)+'_acc'] = acc
             
                ensemble_result_dict= ensemble_accuracy(algorithm.MULDENS_networks,loader,weights,device)
                
                results[name+'_ens_acc']= ensemble_result_dict['acc']
                results[name+'_preds_ens']= ensemble_result_dict['preds']
                results[name+'_labels']= ensemble_result_dict['labels']
                results[name+'_entropies'] = ensemble_result_dict['pred_entropies']

    else:
        
        results={}
        eval_out_loader_names = [i for i in eval_loader_names if '_in' not in i]
        for i, name in enumerate(eval_loader_names):
            if (int(name[3]) not in test_envs):
                loader= eval_dict[name][0]
                weights= eval_dict[name][1]
                if '_in' in name:
                    model_domain_name = 'env'+str(i)+'_out0'
                    model_num_idx = eval_out_loader_names.index(model_domain_name)
                else: 
                    model_num_idx = eval_out_loader_names.index(name)
                model_num = int(correct_models_selected_for_each_domain[model_num_idx])
                acc=accuracy(algorithm.MULDENS_networks[model_num],loader,weights,device)
                results[name+'_acc'] = acc


        # for unobserved domains we will pick top k models from beta  and either do an ensemble or directly pick the best
        #model and return the accuracy
        #beta is a (num_testenvs X num_models)
        if compute_test_beta:
            beta = torch.zeros((len(test_envs), len(algorithm.MULDENS_networks)))
            for j, test_env in enumerate(test_envs):
                for i, domain_idx in enumerate(domains_selected_for_each_model):
                    loaders = []
                    for domain in domain_idx:
                        domain_name = eval_out_loader_names[domain]
                        loaders.append(eval_dict[domain_name][0])
                    test_env_domain_name = 'env'+str(test_env)+'_out0'
                    test_env_loader= eval_dict[test_env_domain_name][0]
                    if len(domain_idx) != 0:
                        beta[test_env,i] = MULDENS_beta_grads(loaders, test_env_loader, algorithm.MULDENS_networks[i], device)
                    else:
                        beta[test_env,i] = 0
            for i,test_env in enumerate(test_envs):
                beta_test_env = beta[i,:]
                best_model_num = np.argmax(beta_test_env)
                for split in ['_in','_out']:
                    name = 'env'+str(test_env)+split+str(0)
                    loader= eval_dict[name][0]
                    weights= eval_dict[name][1]
                    acc=accuracy(algorithm.MULDENS_networks[best_model_num],loader,weights,device)
                    results[name+'_acc'] = acc
        else: 
            """
            if we dont want to compute betas we want to get results using all the models and also an ensemble of them
            """
            for i,test_env in enumerate(test_envs):
                for split in ['_in','_out']:
                    name = 'env'+str(test_env)+split+str(0)
                    loader= eval_dict[name][0]
                    weights= eval_dict[name][1]
                    for m in range(len(algorithm.MULDENS_networks)):
                        acc= accuracy(algorithm.MULDENS_networks[m],loader,weights,device)
                        results[name+'_m_'+str(m)+'_acc'] = acc
                    ensemble_results= ensemble_accuracy(algorithm.MULDENS_networks,loader,weights,device)
                    results[name+'_ens_acc']= ensemble_results['acc']
                    results[name+'_preds_models']= ensemble_results['preds']
                    results[name+'_labels']= ensemble_results['labels']

    return results

