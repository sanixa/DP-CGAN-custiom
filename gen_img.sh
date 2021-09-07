for i in {1..4}
do
	python img_gen.py --model-path checkpoint/eps_1/DPCGAN_MNIST_eps1.1_acc0.${i}.ckpt --save-dir data/eps_1/diff_acc/${i}0
done

for i in {1..5}
do
	python img_gen.py --model-path checkpoint/eps_10/DPCGAN_MNIST_eps10.0_acc0.${i}.ckpt --save-dir data/eps_10/diff_acc/${i}0
done

for i in {1..6}
do
	python img_gen.py --model-path checkpoint/eps_100/DPCGAN_MNIST_eps100.3_acc0.${i}.ckpt --save-dir data/eps_100/diff_acc/${i}0
done

for i in {1..8}
do 
	python img_gen.py --model-path checkpoint/eps_1000/DPCGAN_MNIST_eps1000.8_acc0.${i}.ckpt --save-dir data/eps_1000/diff_acc/${i}0
done

for i in {1..8}
do 
	python img_gen.py --model-path checkpoint/eps_inf/DPCGAN_MNIST_eps_inf_acc0.${i}.ckpt --save-dir data/eps_inf/diff_acc/${i}0
done

for i in 1000 5000 10000 20000
do
	python img_gen.py --model-path checkpoint/eps_1/DPCGAN_MNIST_eps1.1_iteration${i}.ckpt --save-dir data/eps_1/diff_iter/${i}
done

for i in 1000 5000 10000 20000
do
	python img_gen.py --model-path checkpoint/eps_10/DPCGAN_MNIST_eps10.0_iteration${i}.ckpt --save-dir data/eps_10/diff_iter/${i}
done

for i in 1000 5000 10000 20000
do
	python img_gen.py --model-path checkpoint/eps_100/DPCGAN_MNIST_eps100.3_iteration${i}.ckpt --save-dir data/eps_100/diff_iter/${i}
done

for i in 1000 5000 10000 20000
do
	python img_gen.py --model-path checkpoint/eps_1000/DPCGAN_MNIST_eps1000.8_iteration${i}.ckpt --save-dir data/eps_1000/diff_iter/${i}
done
