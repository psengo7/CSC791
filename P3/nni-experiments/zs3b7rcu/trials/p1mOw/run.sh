#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='zs3b7rcu'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/zs3b7rcu/trials/p1mOw'
export NNI_TRIAL_JOB_ID='p1mOw'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/zs3b7rcu/trials/p1mOw'
export NNI_TRIAL_SEQ_ID='7'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/zs3b7rcu/trials/p1mOw/stdout 2>/jet/home/psengo/nni-experiments/zs3b7rcu/trials/p1mOw/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/zs3b7rcu/trials/p1mOw/.nni/state'