#!/usr/bin/env bash

kernels=( 'rbf' 'periodic' 'matern' )
models=( 'tnpd' 'convcnp' 'sptx_fast' 'sptx_full' )
seeds=( 7 8 9 )

for k in "${kernels[@]}" ; do
  for s in "${seeds[@]}" ; do
    for m in "${models[@]}" ; do
      python 1D_GP.py +kernel=$k +model=$m +seed=$s +wandb=True
    done
  done
done
