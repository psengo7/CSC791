#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='2p8zlbqm'
export NNI_SYS_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/mRI5K'
export NNI_TRIAL_JOB_ID='mRI5K'
export NNI_OUTPUT_DIR='/jet/home/psengo/nni-experiments/2p8zlbqm/trials/mRI5K'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/jet/home/psengo/CSC791/P3'
cd $NNI_CODE_DIR
eval 'python main.py' 1>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/mRI5K/stdout 2>/jet/home/psengo/nni-experiments/2p8zlbqm/trials/mRI5K/stderr
echo $? `date +%s%3N` >'/jet/home/psengo/nni-experiments/2p8zlbqm/trials/mRI5K/.nni/state'