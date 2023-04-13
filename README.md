# Dynamical inference of RL parameters 
Code accompanying the project Harnessing the Flexibility of Neural Networks to Predict Dynamical Theoretical Parameters of Human Choice Behavior

In this project we developed a new framework we term t-RNN (theoretical-RNN), in which an RNN is trained to infer time-varying RL parameters of learning agent performing a two-armed bandit task

## Human behavioral datasets:
The human behavioral datasets can be found at:
- Dezfouli et al. (2019) *Models that learn how humans learn: The case of decision-making and its disorders*.
https://figshare.com/articles/dataset/Models_that_learn_how_humans_learn_The_case_of_decision-making_and_its_disorders/8257259

- Gershman (2018) *Deconstructing the human algorithms for exploration*.
https://www.sciencedirect.com/science/article/pii/S0010027717303359#b0180

## Experiments with simulated behavior data 
- To generate a synthetic training-set run ```artificial/create_train_data.ipynb```. 
- To train a neural network model with the synthetic training-set run ```artificial/trnn_training.ipynb```.
- Code for both baseline methods can be found at ```artificial/q_fit.py``` for Stationary Q-learning maximum-likelihood and ```artificial/bayesian_fit.py``` for Bayesian particle filtering.
- To create plots and describe results of simulated behavior run ```artificial/plots.ipynb```
- ```checkpoint/checkpoint_trnn_5.pth``` is a state_dict of the trained model weights used for the analysis. 

<!---
## Experiments with human behavior data
- To generate a synthetic training-set run ```behavioral/generate_train_data.ipynb```.
- To train a neural network model with the synthetic training-set run ```behavioral/rnn_train.ipynb```.
-->
