#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='clhodkua'
export NNI_SYS_DIR='/home/psengo/nni-experiments/clhodkua/trials/lRM6V'
export NNI_TRIAL_JOB_ID='lRM6V'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/clhodkua/trials/lRM6V'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python main.py 1>/home/psengo/nni-experiments/clhodkua/trials/lRM6V/stdout 2>/home/psengo/nni-experiments/clhodkua/trials/lRM6V/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/clhodkua/trials/lRM6V/.nni/state'