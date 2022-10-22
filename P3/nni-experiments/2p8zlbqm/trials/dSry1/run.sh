#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='2p8zlbqm'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/dSry1'
export NNI_TRIAL_JOB_ID='dSry1'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/dSry1'
export NNI_TRIAL_SEQ_ID='5'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/dSry1/stdout 2>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/dSry1/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/2p8zlbqm/trials/dSry1/.nni/state'