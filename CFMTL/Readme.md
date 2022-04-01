# Clustered Federated Multi-Task Learning with Non-IID Data

## Environment

python 3.9.1

pytorch 1.7.1+cu101

## Run Experiments and Get Figures

### Performance on MNIST

python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 1.0 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment performance-mnist --filename fig

python fig.py --experiment performance-mnist --filename fig

### Performance on CIFAR-10

python main.py --dataset cifar --iid iid --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.25 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.5 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.75 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid-single_class --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

python fig.py --experiment performance-cifar --filename fig

### Communication cost

python main.py --dataset cifar --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python fig.py --experiment communication --filename fig

### Various hyperparameters

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.5 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 1.0 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 50 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 1 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 5 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python fig.py --experiment hyperparameters --filename fig

### Different metrics

python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 1.0 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment metric --filename fig

python fig.py --experiment metric --filename fig
