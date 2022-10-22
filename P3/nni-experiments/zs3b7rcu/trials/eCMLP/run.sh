#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='zs3b7rcu'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/zs3b7rcu/trials/eCMLP'
export NNI_TRIAL_JOB_ID='eCMLP'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/zs3b7rcu/trials/eCMLP'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/zs3b7rcu/trials/eCMLP/stdout 2>/jet/home/psengo/nni-experiments/zs3b7rcu/trials/eCMLP/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/zs3b7rcu/trials/eCMLP/.nni/state'