#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ws1yg2e0'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/ws1yg2e0/trials/X9Z3h'
export NNI_TRIAL_JOB_ID='X9Z3h'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/ws1yg2e0/trials/X9Z3h'
export NNI_TRIAL_SEQ_ID='7'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/ws1yg2e0/trials/X9Z3h/stdout 2>/jet/home/psengo/nni-experiments/ws1yg2e0/trials/X9Z3h/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/ws1yg2e0/trials/X9Z3h/.nni/state'