#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=40:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=40gb:scratch_ssd=50gb:cuda_version=12.6
#PBS -N nn_training
DATADIR=/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD
LOGFILE=$DATADIR/out_nn_9.log
chmod 700 $SCRATCHDIR
cp -r $DATADIR/* $SCRATCHDIR
module add mambaforge
mamba activate /storage/brno2/home/miapp/.conda/envs/sadenv
ls -R $SCRATCHDIR > $DATADIR/debugprint
cd $SCRATCHDIR
echo "cuda exists?:"
module add cuda/
nvidia-smi > $LOGFILE
echo $CUDA_VISIBLE_DEVICES
$CUDA_VISIBLE_DEVICES > $LOGFILE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "cuda checked"
python3 -u NN-train.py --datadir $SCRATCHDIR >> $LOGFILE
#cp -R outs $DATADIR/outs/$JOID
mkdir -p $DATADIR/models2
cp -R models/* $DATADIR/models2
clean_scratch