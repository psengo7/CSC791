#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='uq4183ef'
export NNI_SYS_DIR='/home/psengo/nni-experiments/uq4183ef/trials/y647r'
export NNI_TRIAL_JOB_ID='y647r'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/uq4183ef/trials/y647r'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python3 main.py 1>/home/psengo/nni-experiments/uq4183ef/trials/y647r/stdout 2>/home/psengo/nni-experiments/uq4183ef/trials/y647r/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/uq4183ef/trials/y647r/.nni/state'