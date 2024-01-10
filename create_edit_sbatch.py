# script to automatically create sbatch/bsub files

import os
import numpy as np
datasets = ["CIFAR10Splits","CIFAR100Splits"]


#srun --mpibind=off python3 -m domainbed.scripts.train_aug --output_dir /usr/workspace/thopalli/2024/muldens_ensembles/logs/MULDENS_cifar10_indiv --dataset "CIFAR10Splits" --hparams='{"batch_size":32,"resnet18":false,"data_augmentation":1,"MULDENS_num_models":3,"MULDENS_inner_joint_training":false,"lr_MULDENS":1e-3}' 

seeds =[0,2023,2024,2025,2026]
networktype = "vit_b_16"
cluster= 'pascal'
algorithms = ["ERM"]
train_script = "train" #"train_aug"
for algorithm in algorithms:
    for dataset in datasets:

        for seed in seeds:
            filename = f"{cluster}_job_pretrain_dataset_{pretrain_dataset}_dataset_{dataset}_ckpt_{ckpt.split('/')[-1]}.sh"
            filename = os.path.join(f'/usr/workspace/thopalli/ICLR2024/bash_files_LP/all_files_{networktype}_{pretrain_dataset}',filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, "w") as f:
                
                
                if cluster =='pascal':
                    f.write("#!/bin/bash\n")
                    f.write("#SBATCH -N 1\n")
                    f.write("#SBATCH --ntasks-per-node=1\n")
                    f.write("#SBATCH --time=23:59:00\n")
                    f.write("#SBATCH --partition=pbatch\n")
                    f.write("#SBATCH --account=iebdl\n")
                    f.write("#SBATCH --output=./L%J.out\n\n")
                    f.write("source ~/.bashrc\n")
                    f.write("echo 'sourced'\n")
                    f.write("conda activate pytorch2.0\n")
                    f.write("echo 'conda done'\n")
                    f.write("cd /usr/workspace/thopalli/2024/muldens_ensembles\n")
                    #f.write(f"srun python -u linear_probing_ILMVP_duq.py --pretrain_dataset {pretrain_dataset} --dataset {dataset} --DUQ_ckpt_path {ckpt} --network {networktype} > ./out_files_{networktype}_ema/pretrain_dataset_{pretrain_dataset}/dataset_{dataset}_DUQ_ckpt_path_{ckpt.split('/')[-1]}.out\n")
                    f.write(f"srun --mpibind=off python3 -m domainbed.scripts.{train_script} --algorithm {algorithm} --output_dir /usr/workspace/thopalli/2024/muldens_ensembles/logs/MULDENS_cifar10_indiv --dataset {dataset} --seed {seed} --hparams='{"batch_size":32,"resnet18":false,"data_augmentation":1,"MULDENS_num_models":3,"MULDENS_inner_joint_training":false,"lr_MULDENS":1e-3}'")

                    
                else:
                    f.write("#!/bin/sh\n")
                    f.write("#BSUB -nnodes 1\n")
                    f.write("#BSUB -q pbatch\n")
                    f.write("#BSUB -G ams\n")
                    f.write("#BSUB -W 04:59\n")
                    f.write("#BSUB -o ./L%J.out\n\n")
                    #f.write("jsrun -r 1 hostname\n")
                    #f.write("firsthost=$(jsrun --nrs 1 -r 1 hostname)\n")
                    #f.write("echo \"first host: $firsthost\"\n")
                    #f.write("export MASTER_ADDR=$firsthost\n")
                   # f.write("export MASTER_PORT=23456\n")
                    f.write("cd /usr/workspace/thopalli/ICLR2024\n")
                    f.write("source /usr/workspace/thopalli/python-envs/lassen/bin/activate\n")
                    f.write("module load cuda/11.6\n")
                    f.write("export LD_LIBRARY_PATH=/usr/tce/packages/cuda/cuda-11.6.1/lib64:/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib:/usr/tce/packages/cuda/cuda-11.3.0/lib64:/opt/ibm/spectrumcomputing/lsf/10.1.0.10/linux3.10-glibc2.17-ppc64le-csm/lib:/usr/workspace/AMS/ams-spack-environments/0.19/blueos_3_ppc64le_ib_p9-gpu/local/lib\n")
                    f.write(f"jsrun --smpiargs=\"off\" --bind=none -r 1 -g 1 -c 10 python -u linear_probing_ILMVP_duq.py --pretrain_dataset {pretrain_dataset} --dataset {dataset} --DUQ_ckpt_path {ckpt} > out_files_newv2/pretrain_dataset_{pretrain_dataset}/dataset_{dataset}_DUQ_ckpt_path_{ckpt.split('/')[-1]}.out\n")
            os.makedirs(f"/usr/workspace/thopalli/ICLR2024/out_files_{networktype}_ema/pretrain_dataset_{pretrain_dataset}", exist_ok=True)
            print(f"Created file: {filename}")
