#!/bin/bash
#SBATCH -A conf-2021-cvpr
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J tf-vaegan-1
#SBATCH -o tf-vaegan-1.%J.out
#SBATCH -e tf-vaegan-1.%J.err
#SBATCH --mail-user=kai.yi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gtx1080ti|rtx2080ti|v100]


#run the application:
cd /ibex/scratch/yik/tfvaegan/tfvaegan+grawd/

# awa2
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
--gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot data --dataset AWA2 \
--batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
--lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
--feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01 --rw_weight 1 --creative_weight 0.01 \
--n_examples_per_proto 10


# cub
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8  python train_tfvaegan_inductive.py --gammaD 10 --gammaG 10 \
--gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot data --dataset CUB \
--nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
--recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2 --rw_weight 1 --creative_weight 0.01 \
--n_examples_per_proto 10

# flo
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py \
--gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 \
--syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
--ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
--classifier_lr 0.001 --cuda --image_embedding res101 --dataroot data \
--recons_weight 0.01 --feedback_loop 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --dec_lr 0.0001 --rw_weight 1 --creative_weight 0.01 \
--n_examples_per_proto 10


# sun
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 1 --gammaG 1 --gzsl \
--manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 \
--ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.01 \
--classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot data \
--recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --rw_weight 1 --creative_weight 0.001 \
--n_examples_per_proto 10

