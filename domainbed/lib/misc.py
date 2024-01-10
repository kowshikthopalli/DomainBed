# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
import torch.nn.functional as F
import torch.autograd as autograd
from shutil import copyfile
import pickle
import numpy as np
import torch
import tqdm
from collections import Counter
import torch.nn as nn

class PredictiveEntropy(nn.Module):
    def __init__(self):
        super(PredictiveEntropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

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
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist() 
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            try:
                p = network.predict(x)
            except:
                p = network(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def myensemble_accuracy(networks, loader, weights, device,avg_after_softmax=False):
    correct = 0
    total = 0
    weights_offset = 0

    [network.eval() for network in networks]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            predictions = [network.predict(x) for network in networks]
            
            if avg_after_softmax:
                predictions = [torch.softmax(j,dim=1) for j in predictions]
            predictions = torch.mean(torch.stack(predictions),dim =0)

            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if predictions.size(1) == 1:
                correct += (predictions.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (predictions.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    [network.train() for network in networks]

    return correct / total

def calculate_cosine_similarity_loss(list_gradients1,list_gradients2):
    cos = torch.nn.CosineSimilarity()
    cos_params = []
    for i,j in zip(list_gradients1,list_gradients2):
        cos_sim = cos(i.view(1,-1),j.view(1,-1))
        cos_params.append(cos_sim)
    return torch.mean(torch.Tensor(cos_params))

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
    labels_=[]
    pred_entropies_all=[]
    max_probs= []
    entropy = PredictiveEntropy().to(device)
    eceloss = ECELoss().to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            try:
               
                p = [network.predict(x) for network in networks]
            except:
                p = [network(x) for network in networks]
                p = [torch.softmax(j,dim=1) for j in p] 
            
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
            pred_entropies_all.append(entropy(p_mean).detach().cpu().numpy())

            labels_.append(y.detach().cpu().numpy())
    [network.train() for network in networks]
    return_dict={}
    return_dict['acc']=correct / total
  
    return_dict['preds']= np.concatenate(predictions_)
    
    return_dict['labels']= np.concatenate(labels_)
    return_dict['pred_entropies']= np.concatenate(pred_entropies_all)
    return_dict['ece']= eceloss(torch.tensor(return_dict['preds']).to(device),\
        torch.tensor(return_dict['labels']).to(device) ).item()
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

    obs_loader_insplit_names = ['env{}_in'.format(i)
        for i in range(len(eval_loader_names)//2) if i not in test_envs]
    obs_loader_outsplit_names= ['env{}_out'.format(i)
        for i in range(len(eval_loader_names)//2) if i not in test_envs]

    un_obs_insplit_name = ['env{}_in'.format(i) for i in test_envs]
    un_obs_outsplit_name = ['env{}_out'.format(i) for i in test_envs]

    
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
            for split in ['_in','_out']:
                
                name = 'env'+str(i)+split
                loader= eval_dict[name][0]
                weights= eval_dict[name][1]
                if i in test_envs: name = 'unobs_'+name
                for m in range(len(algorithm.MULDENS_networks)):
                    acc= accuracy(algorithm.MULDENS_networks[m],loader,weights,device)
                     
                    results[name+'_m_'+str(m)+'_acc'] = acc
                # for m in range(len(algorithm.MULDENS_networks)):
                #         acc= accuracy(algorithm.MULDENS_networks[m],loader,weights,device)
                #         results[name+'_m_'+str(m)+'_acc'] = acc
                ensemble_results= ensemble_accuracy(algorithm.MULDENS_networks,loader,weights,device)
                
                results[name+'_ens_acc']= ensemble_results['acc']
                # results[name+'_preds_models']= ensemble_results['preds']
                # results[name+'_labels']= ensemble_results['labels']

    else:
        results={}
        for i in range(len(eval_loader_names)//2):
            if i not in test_envs:
                for split in ['_in','_out']:
                    name = 'env'+str(i)+split
                    loader= eval_dict[name][0]
                    weights= eval_dict[name][1]
                    model_num = int(correct_models_selected_for_each_domain[i])
                    acc=accuracy(algorithm.MULDENS_networks[model_num],loader,weights,device)
                    results[name+'_acc'] = acc
        # for unobserved domains we will pick top k models from beta  and either do an ensemble or directly pick the best
        #model and return the accuracy
        #beta is a (num_testenvs X num_models)
        if compute_test_beta:
            beta = torch.zeros((len(test_envs), len(algorithm.MULDENS_networks)))
            for test_env in range(len(test_envs)):
                for i, domain_idx in enumerate(domains_selected_for_each_model):
                    loaders = []
                    for domain in domain_idx:
                        domain_name = 'env'+str(domain)+'_out'
                        loaders.append(eval_dict[domain_name][0])
                    test_env_domain_name = 'env'+str(test_env)+'_out'
                    test_env_loader= eval_dict[test_env_domain_name][0]
                    if len(domain_idx) != 0:
                        beta[test_env,i] = MULDENS_beta_grads(loaders, test_env_loader, algorithm.MULDENS_networks[i], device)
                    else:
                        beta[test_env,i] = 0
            for i,test_env in enumerate(test_envs):
                beta_test_env = beta[i,:]
                best_model_num = np.argmax(beta_test_env)
                for split in ['_in','_out']:
                    name = 'env'+str(test_env)+split
                    loader= eval_dict[name][0]
                    weights= eval_dict[name][1]
                    acc=accuracy(algorithm.MULDENS_networks[best_model_num],loader,weights,device)
                    results[name+'_acc'] = acc
            results['beta_test']= beta
        else:
            """
            if we dont want to compute betas we want to get results using all the models and also an ensemble of them
            """
            for i,test in enumerate(test_envs):
                for split in ['_in','_out']:
                    name = 'env'+str(test_env)+split
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
    # domains_selected_for_each_model=  [[] for i in range(len(algorithm.MULDENS_networks))]
    # for m in range(len(algorithm.MULDENS_networks)):
    #     for i,ms in enumerate(correct_models_selected_for_each_domain):
    #         if ms is not np.nan:
    #             if ms ==m:
    #                 domains_selected_for_each_model[m].append(i)
    # # compute betas
    # for i, model in enumerate(algorithm.MULDENS_networks):
    #     # compute gradients with the corresponding domains selected for this model
    #     domains_selected = domains_selected_for_each_model[i]
    #     obs_loaders= []
    #     for d in domains_selected: # train_loader
    #         loader_name= 'env'+str(d)+'_in'
    #         obs_loaders.append(eval_dict[loader_name])



    # with torch.no_grad():

        






    #     for x, y in loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         p = network.predict(x)
    #         if weights is None:
    #             batch_weights = torch.ones(len(x))
    #         else:
    #             batch_weights = weights[weights_offset : weights_offset + len(x)]
    #             weights_offset += len(x)
    #         batch_weights = batch_weights.to(device)
    #         if p.size(1) == 1:
    #             correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
    #         else:
    #             correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
    #         total += batch_weights.sum().item()
    # network.train()

    # return correct / total
def save_obj_with_filename(data,filename):
    with open(filename,"wb") as f:
        return pickle.dump(data,f)
def load_obj(filename ):
    with open(filename, 'rb') as f:
        return pickle.load(f)
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
