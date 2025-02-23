#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:mem=40gb:scratch_ssd=50gb
#PBS -N NeMo_validate
DATADIR=/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD
LOGFILE=$DATADIR/out_NeMo_1.log
chmod 700 $SCRATCHDIR
cp -r $DATADIR/* $SCRATCHDIR
module add py-pip
$DATADIR/nemopip/bin/activate
ls -R $SCRATCHDIR > $DATADIR/debugprint
cd $SCRATCHDIR
echo "cuda exists?:"
module add cuda/
nvidia-smi > $LOGFILE
echo $CUDA_VISIBLE_DEVICES
$CUDA_VISIBLE_DEVICES > $LOGFILE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "cuda checked"
python3 -u NeMo_VAD_meta.py --datadir $SCRATCHDIR >> $LOGFILE
#cp -R outs $DATADIR/outs/$JOID
clean_scratch