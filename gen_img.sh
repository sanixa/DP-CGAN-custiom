# +

if [[ "$1" == "cifar" ]]
then

	for i in 1 10 100 1000 inf
	do
		for (( j=1000; j<=20000; j=j+1000 ))
		do
			python img_gen.py --dataset cifar_10 --g_dim 100 --model-path checkpoint_cifar/eps_${i}_z100/iteration${j}.ckpt --save-dir data_cifar/eps_${i}/diff_iter/${j}
		done
	done
    
	for i in 01 02 001 
	do
		for (( j=1000; j<=20000; j=j+1000 ))
		do
			python img_gen.py --dataset cifar_10 --g_dim 100 --model-path checkpoint_cifar/n_${i}_z100/iteration${j}.ckpt --save-dir data_cifar/n_${i}/diff_iter/${j}
		done
	done
# -

else


	for i in 1 10 100 1000 inf
	do
		for (( j=1000; j<=20000; j=j+1000 ))
		do
			python img_gen.py --g_dim 32 --dataset mnist --model-path checkpoint/eps_${i}_z32/iteration${j}.ckpt --save-dir data/eps_${i}/diff_iter/${j}
		done
	done
fi

	for i in 01 02 001
	do
		for (( j=1000; j<=20000; j=j+1000 ))
		do
			python img_gen.py --g_dim 32 --dataset mnist --model-path checkpoint/n_${i}_z32/iteration${j}.ckpt --save-dir data/n_${i}/diff_iter/${j}
		done
	done
