from nni.experiment import Experiment

#Tuners and configurations
TunerParamFull = {
    'Tuner 1': {
        'tuner': 'TPE',
        'config_list': {
            'optimize_mode': 'maximize',
        } 
    },
    'Tuner 2': {
        'tuner': 'SMAC',
        'config_list': {
            'optimize_mode': 'maximize',
        }
    },
    'Tuner 3': {
        'tuner': 'Metis',
        'config_list': {
            'optimize_mode': 'maximize',
        } 
    },'Tuner 4': {
        'tuner': 'TPE',
        'config_list': {
            'optimize_mode': 'minimize',
        } 
    },
    'Tuner 5': {
        'tuner': 'SMAC',
        'config_list': {
            'optimize_mode': 'minimize',
        }
    },
    'Tuner 6': {
        'tuner': 'Metis',
        'config_list': {
            'optimize_mode': 'minimize',
        } 
    },
}

TunerParam = {
    'Tuner 1': {
        'tuner': 'TPE',
        'config_list': {
            'optimize_mode': 'maximize',
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
    experiment.config.trial_command = 'python3 main.py'
    experiment.config.trial_code_directory = '.'

    #configure search space
    experiment.config.search_space = search_space

    #configure tuning algorithm
    experiment.config.tuner.name = val['tuner']
    experiment.config.tuner.class_args = val['config_list']

    #configure trials to run
    experiment.config.max_trial_number = 2
    experiment.config.trial_concurrency = 2

    #run experiment
    experiment.run(8080)
    
    #pause and wait for user input to run next experiment
    input("Press any button for next Tuner experiment: ")