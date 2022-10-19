#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uq4183ef'
export NNI_SYS_DIR='/home/psengo/nni-experiments/uq4183ef/trials/uooiw'
export NNI_TRIAL_JOB_ID='uooiw'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/uq4183ef/trials/uooiw'
export NNI_TRIAL_SEQ_ID='6'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python3 main.py 1>/home/psengo/nni-experiments/uq4183ef/trials/uooiw/stdout 2>/home/psengo/nni-experiments/uq4183ef/trials/uooiw/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/uq4183ef/trials/uooiw/.nni/state'