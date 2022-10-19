#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='clhodkua'
export NNI_SYS_DIR='/home/psengo/nni-experiments/clhodkua/trials/yFP3x'
export NNI_TRIAL_JOB_ID='yFP3x'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/clhodkua/trials/yFP3x'
export NNI_TRIAL_SEQ_ID='8'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python main.py 1>/home/psengo/nni-experiments/clhodkua/trials/yFP3x/stdout 2>/home/psengo/nni-experiments/clhodkua/trials/yFP3x/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/clhodkua/trials/yFP3x/.nni/state'