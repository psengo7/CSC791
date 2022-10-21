from nni.experiment import Experiment

#Tuners and configurations - Evolution, GP, Hyperband

TunerParam = {
    #low evolution performance
    'Evolution[Pop_Size = 5]': {
        'tuner': 'Evolution',
        'config_list': {
            'population_size': 5,
        } 
    },
    #better evolution performance
    'Evolution[Pop_Size = 100]': {
        'tuner': 'Evolution',
        'config_list': {
            'population_size': 100,
        }
    },
    #average exploration
    'GP[kappa: 5.0]': {
        'tuner': 'GP',
        'config_list': {
            'kappa': 5.0,
        } 
    },
    #high exploration
    'GP[kappa: 20.0]': {
        'tuner': 'GP',
        'config_list': {
            'kappa': 20.0,
        } 
    },
    #average amount of config survive per round
    'Hyperband[eta = 2]': {
        'tuner': 'Hyperband',
        'config_list': {
            'eta': 2,
        }
    },
    #less config survive per round
    'Hyperband[eta = 10]': {
        'tuner': 'Hyperband',
        'config_list': {
            'eta': 10,
        } 
    },
}

TunerParamTest = {
    'Tuner 5': {
        'tuner': 'Hyperband',
        'config_list': {
            'optimize_mode': 'minimize',
        }
    }
}


search_space = {
    'lr':{'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'batch_size': {"_type": "choice", "_value": [50, 250, 500]}         
}

for key, val in TunerParam.items():
    #configure experiment
    experiment = Experiment('local')

    #configure trial code
    experiment.config.trial_command = 'python main.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.experiment_name = key
    
    #configure search space
    experiment.config.search_space = search_space

    #configure tuning algorithm
    experiment.config.tuner.name = val['tuner']
    experiment.config.tuner.class_args = val['config_list']

    #configure trials to run
    experiment.config.max_trial_number = 15
    experiment.config.trial_concurrency = 20

    #run experiment
    experiment.run(8091)
    
    #pause and wait for user input to run next experiment
    #input("Press any button for next Tuner experiment: ")
    experiment.stop()