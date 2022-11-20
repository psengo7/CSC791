# DIRECTORY STRUCTURE: 


* main.py **(1)**
* modelHPO.py **(2)**
* view.py **(3)**
* nni-experiments/ **(4)** 
* job-proj3.sh **(5)**
* README.md

1. Holds the neural net model with all operations to train and test it
2. Performs the Hyper paramter optimization on the model in main.py
3. Used to view the saved experiments in nni-experiments (make sure to move the nni-experiments folder to $HOME directory in bridges cluster)
4. holds the trial results for the HPO optimizations
5. bash script to run the modelHPO.py program on bridges cluster


# HOW TO RUN PROGRAM: 

1. Open bridges cluster and clone git repository. 
2. Move the nni-experiments folder to the top level directory

**To rerun hyperparameter optimizations:** 

3. run "sbatch job-proj3" in login node

**To View results:**

3. In login node run "module load anaconda3"
4. In login node run "conda activate csc791"
5. In login node run "python view.py"

NOTE: to show next experiment you need to press "CTRL+C" to terminate current browser and continue program to next browser. 
