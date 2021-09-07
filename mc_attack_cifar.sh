###
#### ep_1
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_1000 --load_dir checkpoint_cifar/eps_1/diff_iter/iteration1000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_5000 --load_dir checkpoint_cifar/eps_1/diff_iter/iteration5000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_10000 --load_dir checkpoint_cifar/eps_1/diff_iter/iteration10000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=5 python mc_attack.py  --name ep1_20000 --load_dir checkpoint_cifar/eps_1/diff_iter/iteration20000.ckpt --dataset cifar_10 --exp mc_attack

#### ep_10
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep10_1000 --load_dir checkpoint_cifar/eps_10/diff_iter/iteration1000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep10_5000 --load_dir checkpoint_cifar/eps_10/diff_iter/iteration5000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep10_10000 --load_dir checkpoint_cifar/eps_10/diff_iter/iteration10000.ckpt --dataset cifar_10 --exp mc_attack
CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep10_20000 --load_dir checkpoint_cifar/eps_10/diff_iter/iteration20000.ckpt --dataset cifar_10 --exp mc_attack

#### ep_100
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep100_1000 --load_dir checkpoint_cifar/eps_100/diff_iter/iteration1000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep100_5000 --load_dir checkpoint_cifar/eps_100/diff_iter/iteration5000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep100_10000 --load_dir checkpoint_cifar/eps_100/diff_iter/iteration10000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=5 python mc_attack.py  --name ep100_20000 --load_dir checkpoint_cifar/eps_100/diff_iter/iteration20000.ckpt --dataset cifar_10 --exp mc_attack

#### ep_1000
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1000_1000 --load_dir checkpoint_cifar/eps_1000/diff_iter/iteration1000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1000_5000 --load_dir checkpoint_cifar/eps_1000/diff_iter/iteration5000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1000_10000 --load_dir checkpoint_cifar/eps_1000/diff_iter/iteration10000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=5 python mc_attack.py  --name ep1000_20000 --load_dir checkpoint_cifar/eps_1000/diff_iter/iteration20000.ckpt --dataset cifar_10 --exp mc_attack

#### ep_inf
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name epinf_1000 --load_dir checkpoint_cifar/eps_inf/diff_iter/iteration1000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name epinf_5000 --load_dir checkpoint_cifar/eps_inf/diff_iter/iteration5000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name epinf_10000 --load_dir checkpoint_cifar/eps_inf/diff_iter/iteration10000.ckpt --dataset cifar_10 --exp mc_attack
#CUDA_VISIBLE_DEVICES=5 python mc_attack.py  --name epinf_20000 --load_dir checkpoint_cifar/eps_inf/diff_iter/iteration20000.ckpt --dataset cifar_10 --exp mc_attack

