# RL-GNN-for-RCPSP
A program implemented by Pytorch for solving RCPSP and RCPSP with resource disruptions, based on graph neural network and reinforcement learning. Paper is https://doi.org/10.1016/j.rcim.2023.102628

## Requirements:  
Python >=3.6  
Pytorch >= 1.8.0  
Be sure that your CUDA is available!  

## To reproduce experiment results:  
check and run test_rcpsp.py to test on RCPSP  
check and run test_srcpsp.py to test on RCPSP-RD  

## To train a new model:  
Check Params.py to set hyperparameters  
run ppo.py to train a model on RCPSP  
run ppo_for_variant.py to train a model on RCPSP-RD  

Use Tensorboard to monitor training process  

## Bottleneck & suggestions
As discussed in the paper, I think the bottleneck of using these GNN-related method to slove scheduling problems lies in the feature extraction capability and receptive field of GNN. In order to improve the performance, I suggest changing the structure of the GNN, using a structure that can effectively extract information from very distant nodes (such as using Mixhop layers?), and then using supervised learning to quickly verify the GNN learning ability. 

## Specially thanks!
Part of the GNN code comes from L2D https://github.com/zcaicaros/L2D, the method of this paper mainly imitates them.  
The code of PPO is modified from CleanRL https://github.com/vwxyzjn/cleanrl  
Specially thanks to them!  


