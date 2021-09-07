import pandas as pd
import numpy as np

total_domains = 4

ONLY_ENSEMBLE_OBS=False





       

if ONLY_ENSEMBLE_OBS:

    for test_env in range(total_domains):

        out_file = 'Inevnio_withoutaug/VLCS_M3/env'+str(test_env)+'/indiv_ensemble_no_beta/out_eval copy.txt'
        
        out_df = pd.read_fwf(out_file)

        train_envs = np.setdiff1d(np.arange(total_domains),test_env)
        train_out_split_names= ['env'+str(i)+'_out_ens_acc' for i in train_envs]

        train_in_split_names= ['env'+str(i)+'_in_ens_acc' for i in train_envs]

        test_ensemble_name= 'unobs_env'+str(test_env)+'_in_ens_acc'
        test_ensemble_out_name= 'unobs_env'+str(test_env)+'_out_ens_acc'
        #step = out_df['step']
        validation_acc= np.mean(np.stack([out_df[i].values for i in train_in_split_names+train_out_split_names]) ,axis=0)
        max_val_idx= np.argmax(validation_acc)
        max_val_idx_1 = np.argmax( out_df[test_ensemble_out_name].values)
        print('test_train_Domain_val','test_oracle',out_df[test_ensemble_name].iloc[max_val_idx]*100,out_df[test_ensemble_name].iloc[max_val_idx_1]*100)





else:

    for test_env in range(total_domains):

        out_file = 'Inevnio_withoutaug/ColoredMNIST_M3/env'+str(test_env)+'/indiv_ensemble_no_beta/out copy.txt'
        out_df= pd.read_fwf(out_file)
        out_df= out_df[out_df['step']<5001]
        out_df= out_df[out_df['step']>1]
        
        train_envs = np.setdiff1d(np.arange(total_domains),test_env)

        train_out_split_names= ['env'+str(i)+'_out_acc' for i in train_envs]

        test_ensemble_name= 'env'+str(test_env)+'_in_ens_acc'
        test_ensemble_out_name= 'env'+str(test_env)+'_out_ens_acc'
        epochs= out_df['epoch']
        validation_acc= np.sum(np.stack([out_df[i].values for i in train_out_split_names]) ,axis=0)/3.

        max_val_idx= np.argmax(validation_acc)
        max_val_idx_1 = np.argmax( out_df[test_ensemble_out_name].values)

        print('test_train_Domain_val','test_oracle',out_df[test_ensemble_name].iloc[max_val_idx]*100,out_df[test_ensemble_name].iloc[max_val_idx_1]*100)




