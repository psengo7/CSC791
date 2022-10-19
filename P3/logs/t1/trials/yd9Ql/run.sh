#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uq4183ef'
export NNI_SYS_DIR='/home/psengo/nni-experiments/uq4183ef/trials/yd9Ql'
export NNI_TRIAL_JOB_ID='yd9Ql'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/uq4183ef/trials/yd9Ql'
export NNI_TRIAL_SEQ_ID='9'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python3 main.py 1>/home/psengo/nni-experiments/uq4183ef/trials/yd9Ql/stdout 2>/home/psengo/nni-experiments/uq4183ef/trials/yd9Ql/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/uq4183ef/trials/yd9Ql/.nni/state'