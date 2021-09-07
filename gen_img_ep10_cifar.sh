for i in {1..5}
do
	python img_gen.py --model-path checkpoint_cifar/eps_10/diff_acc/acc0.${i}.ckpt --save-dir data_cifar/eps_10/diff_acc/${i}0 --dataset cifar_10
done

for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	python img_gen.py --model-path checkpoint_cifar/eps_10/diff_iter/iteration${i}.ckpt --save-dir data_cifar/eps_10/diff_iter/${i} --dataset cifar_10
done

