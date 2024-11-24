#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_ssd=32gb:cuda_version=12.6
#PBS -N Training_script
DATADIR=/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD
chmod 700 $SCRATCHDIR
cp -r $DATADIR/* $SCRATCHDIR
module add mambaforge
mamba activate /storage/brno2/home/miapp/.conda/envs/sadenv
ls -R $SCRATCHDIR > $DATADIR/debugprint
cd $SCRATCHDIR
echo "cuda?:"
module add cuda/
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "-cuda-"
python3 __init__.py --datadir $SCRATCHDIR| tee -a $DATADIR/print_outputs.log
cp -R outs $DATADIR/outs/$JOID
clean_scratch