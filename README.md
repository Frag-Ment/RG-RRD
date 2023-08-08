# RL-GNN-for-RCPSP
A program implemented by Pytorch for solving RCPSP and RCPSP with resource distrubtions, based on graph neural network and reinforcement learning

Requirements:  
Python >=3.6  
Pytorch >= 1.8.0  
Be sure that your CUDA is available!  

To reproduce experiment results:  
check and run test_rcpsp.py to test on RCPSP  
check and run test_srcpsp.py to test on RCPSP-RD  

To train a new model:  
Check Params.py to set hyperparameters  
run ppo.py to train a model on RCPSP  
run ppo_for_variant.py to train a model on RCPSP-RD  

Use Tensorboard to monitor training process  

The code of PPO is modified from CleanRL https://github.com/vwxyzjn/cleanrl  
Part of the GNN code comes from L2D https://github.com/zcaicaros/L2D  
Specially thanks to them!  
