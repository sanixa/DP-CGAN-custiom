# +

if [[ "$1" == "cifar" ]]
then

	for i in ge lap mp ts
	do
		for j in 1000 5000 10000 20000
		do
			python img_gen.py --dataset cifar_10 --g_dim 100 --model-path checkpoint_cifar/eps_10_${i}_z100/iteration${j}.ckpt --save-dir data_cifar/eps_10_${i}/diff_iter/${j}
		done
	done
    
# -

else


	for i in ge lap mp ts
	do
		for j in 1000 5000 10000 20000
		do
			python img_gen.py --g_dim 32 --dataset mnist --model-path checkpoint/eps_10_${i}_z32/iteration${j}.ckpt --save-dir data/eps_10_${i}/diff_iter/${j}
		done
	done

fi