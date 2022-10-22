#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='75cgeop3'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/75cgeop3/trials/zT9ri'
export NNI_TRIAL_JOB_ID='zT9ri'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/75cgeop3/trials/zT9ri'
export NNI_TRIAL_SEQ_ID='2'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/75cgeop3/trials/zT9ri/stdout 2>/jet/home/psengo/nni-experiments/75cgeop3/trials/zT9ri/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/75cgeop3/trials/zT9ri/.nni/state'