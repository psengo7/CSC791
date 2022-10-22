#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='81aq62if'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/81aq62if/trials/jUhZ4'
export NNI_TRIAL_JOB_ID='jUhZ4'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/81aq62if/trials/jUhZ4'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/81aq62if/trials/jUhZ4/stdout 2>/jet/home/psengo/nni-experiments/81aq62if/trials/jUhZ4/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/81aq62if/trials/jUhZ4/.nni/state'