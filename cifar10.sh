CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name eps_inf_z100 --noise 0
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name eps_1_z100 --noise 14.5
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name eps_10_z100 --noise 1.45
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name eps_100_z100 --noise 0.531
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name eps_1000_z100 --noise 0.41
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name n_01_z100 --noise 0.1
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name n_02_z100 --noise 0.2
CUDA_VISIBLE_DEVICES=5 python cifar_torch3.py --g_dim 100 --exp_name n_001_z100 --noise 0.01

CUDA_VISIBLE_DEVICES=5 python cifar_ts_torch3.py --g_dim 100 --exp_name eps_10_ts_z100 --noise 1.45
CUDA_VISIBLE_DEVICES=5 python cifar_mp_torch3.py --g_dim 100 --exp_name eps_10_mp_z100 --noise 1.45
CUDA_VISIBLE_DEVICES=5 python cifar_lap_torch3.py --g_dim 100 --exp_name eps_10_lap_z100 --noise 1.45
CUDA_VISIBLE_DEVICES=5 python cifar_ge_torch3.py --g_dim 100 --exp_name eps_10_ge_z100 --noise 1.45
