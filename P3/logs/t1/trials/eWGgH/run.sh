#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='clhodkua'
export NNI_SYS_DIR='/home/psengo/nni-experiments/clhodkua/trials/eWGgH'
export NNI_TRIAL_JOB_ID='eWGgH'
export NNI_OUTPUT_DIR='/home/psengo/nni-experiments/clhodkua/trials/eWGgH'
export NNI_TRIAL_SEQ_ID='9'
export NNI_CODE_DIR='/home/psengo/CSC791/CSC791/P3'
cd $NNI_CODE_DIR
eval python main.py 1>/home/psengo/nni-experiments/clhodkua/trials/eWGgH/stdout 2>/home/psengo/nni-experiments/clhodkua/trials/eWGgH/stderr
echo $? `date +%s%3N` >'/home/psengo/nni-experiments/clhodkua/trials/eWGgH/.nni/state'