# comp579_final-project

## Project Files
The components of the program are located in different files including:
- train.py: contains a class called Agent which involves all necessary initializations and the code responsible for training the agent and updating the neural netowork weights.
- DQN.py: contains the DQN implementation.
- memory.py: contains the memory buffer implementations.
- gridWorldEnv.py: contains the customized gridworld environment for experiments 2 and 3.

## Experiments
The 3 experiments have thier own files (.ipynb) which involve everything regarding their parameters and their graph drawings. After running the experiments, we store the experiment results in numpy (.npy) files within a folder called "experiment_data" to have easy access to them later. These experiments all use parallel processing for increasing the speed. Note that in total, it took a lot of trainig time (more than 25 hours) for all the experiments combined.

## How to (Re)run the Experiments
simply run the codes withing experiment{1,2,3}.ipynb files : )
