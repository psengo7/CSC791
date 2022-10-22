import nni
from nni.experiment import Experiment

experiment_id = ['2p8zlbqm','75cgeop3','81aq62if','ws1yg2e0','xcspe79w','zs3b7rcu']
for id in experiment_id:
    nni.experiment.Experiment.view(id, 8087, False)