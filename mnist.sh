CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name eps_inf_z32 --noise 0
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name eps_1_z32 --noise 14.5
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name eps_10_z32 --noise 1.45
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name eps_100_z32 --noise 0.531
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name eps_1000_z32 --noise 0.41
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name n_01_z32 --noise 0.1
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name n_02_z32 --noise 0.2
CUDA_VISIBLE_DEVICES=0 python mnist_torch3.py --g_dim 32 --exp_name n_001_z32 --noise 0.01
