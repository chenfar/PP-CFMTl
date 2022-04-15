CFMTL：明文多任务联邦学习

crypten：MPC计算框架源码

# secure MNIST experiments
python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 1.0 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.1 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.1 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.001 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.001 --experiment performance-mnist


