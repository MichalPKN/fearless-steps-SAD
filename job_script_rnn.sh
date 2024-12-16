#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=23:0:0
#PBS -l select=1:ncpus=1:ngpus=2:mem=40gb:scratch_ssd=50gb:cuda_version=12.6
#PBS -N rnn_training
DATADIR=/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD
LOGFILE=$DATADIR/out_rnn_2.log
chmod 700 $SCRATCHDIR
cp -r $DATADIR/* $SCRATCHDIR
module add mambaforge
mamba activate /storage/brno2/home/miapp/.conda/envs/sadenv
ls -R $SCRATCHDIR > $DATADIR/debugprint
cd $SCRATCHDIR
echo "cuda exists?:"
module add cuda/
nvidia-smi > $LOGFILE
echo $CUDA_VISIBLE_DEVICES >> $LOGFILE
$CUDA_VISIBLE_DEVICES >> $LOGFILE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "cuda checked"
python3 -u RNN-train.py --datadir $SCRATCHDIR >> $LOGFILE
#cp -R outs $DATADIR/outs/$JOID
cp -R models $DATADIR/models
clean_scratch