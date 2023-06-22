#!/bin/bash

. "/home/anne/mambaforge/etc/profile.d/mamba.sh"
. "/home/anne/mambaforge/etc/profile.d/conda.sh"

dataset_str_id="qm9"
collec_size=6
n_samples=1000
min_order=2
max_order=5
k=5

mamba activate py-pmi

# solver lsqr
# tol 0.000001
# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000001 --solver lsqr --tol 0.000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00001 --solver lsqr --tol 0.000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0001 --solver lsqr --tol 0.000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0000001 --solver lsqr --tol 0.000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00005 --solver lsqr --tol 0.000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000005 --solver lsqr --tol 0.000001

# # tol 0.00001
# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000001 --solver lsqr --tol 0.00001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00001 --solver lsqr --tol 0.00001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0001 --solver lsqr --tol 0.00001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0000001 --solver lsqr --tol 0.00001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00005 --solver lsqr --tol 0.00001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000005 --solver lsqr --tol 0.00001

# # tol 0.0000001
# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000001 --solver lsqr --tol 0.0000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00001 --solver lsqr --tol 0.0000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0001 --solver lsqr --tol 0.0000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0000001 --solver lsqr --tol 0.0000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00005 --solver lsqr --tol 0.0000001

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000005 --solver lsqr --tol 0.0000001

# solver svd
python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000001 --solver svd

python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00001 --solver svd

python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0001 --solver svd

python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0000001 --solver svd

python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00005 --solver svd

python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000005 --solver svd

# # solver cholesky
# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000001 --solver cholesky

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00001 --solver cholesky

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0001 --solver cholesky

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.0000001 --solver cholesky

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.00005 --solver cholesky

# python random_sampled_collections_RR.py --dataset_str_id $dataset_str_id --collec_size $collec_size --n_samples $n_samples --min_order $min_order --max_order $max_order --k $k --alpha 0.000005 --solver cholesky

mamba deactivate