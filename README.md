# Dynamical inference of RL parameters 
Code accompanying the project *Harnessing the Flexibility of Neural Networks to Predict Dynamical Theoretical Parameters of Human Choice Behavior*.

In this project we developed a new framework we term t-RNN (theoretical-RNN), in which an RNN is trained to infer time-varying RL parameters of learning agent performing a two-armed bandit task. 

<a href="url"> <img src="https://github.com/yoavger/harnessing_the_flexibility_of_nn_to_predict_dynamical/blob/main/artificial/plots/for_git.png" height="330" width="500"></a>


## Experiments with simulated behavior data 
- To generate a synthetic training-set run ```artificial/code/create_train_data.ipynb```. 
- To train a t-RNN model with the synthetic training-set run ```artificial/trnn_training.ipynb```.
- Code for both baseline methods can be found at ```artificial/q_fit.py``` for Stationary Q-learning maximum-likelihood and ```artificial/code/bayesian_fit.py``` for Bayesian particle filtering.
- Plots and describe results run ```artificial/plots.ipynb```
- ```artificial/code/checkpoint/checkpoint_trnn_5.pth``` is a state_dict of the trained model weights used for the analysis. 

## Experiments with human dataset (psychiatric individuals)

The human behavioral dataset can be found at:
- Dezfouli et al. (2019) *Models that learn how humans learn: The case of decision-making and its disorders*.
https://figshare.com/articles/dataset/Models_that_learn_how_humans_learn_The_case_of_decision-making_and_its_disorders/8257259

Code:
- To generate a synthetic training-set run ```dezfouli/code/create_train_data.ipynb```.
- To train a t-RNN model with the synthetic training-set run ```dezfouli/code/trnn_training.ipynb```.
- Code for stationary Q-learning with preservation model can be found at ```dezfouli/code/qp_fit.py```
- Code for data-driven RNN can be found at ```dezfouli/code/drnn.ipynb```
- Plots and describe results run  ```dezfouli/code/plots.ipynb```
- ```dezfouli/code/checkpoint_trnn.pth``` is a state_dict of the trained model weights used for the analysis. 

## Experiments with human dataset (exploration behavior)  
The human behavioral dataset can be found at:
- Gershman (2018) *Deconstructing the human algorithms for exploration*.
https://www.sciencedirect.com/science/article/pii/S0010027717303359#b0180

Code:
- To generate a synthetic training-set run ```gershman/code/create_train_data.ipynb```.
- To train a t-RNN model with the synthetic training-set run ```gershman/code/trnn_training.ipynb```.
- Code for stationary hybrid exploration model can be found at ```gershman/code/hybrid_fit.py``` 
- Code for data-driven RNN can be found at ```gershman/code/drnn.ipynb```
- Plots and describe results run ```gershman/code/plots.ipynb```
- ```gershman/code/checkpoint_trnn.pth``` is a state_dict of the trained model weights used for the analysis. 
