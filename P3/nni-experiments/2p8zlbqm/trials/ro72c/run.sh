#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='2p8zlbqm'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/ro72c'
export NNI_TRIAL_JOB_ID='ro72c'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/ro72c'
export NNI_TRIAL_SEQ_ID='6'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/ro72c/stdout 2>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/ro72c/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/2p8zlbqm/trials/ro72c/.nni/state'