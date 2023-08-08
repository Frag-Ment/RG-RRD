import argparse
import os
from distutils.util import strtobool

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    help="the name of this experiment")
parser.add_argument("--seed", type=int, default=1,
    help="seed of the experiment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, cuda will be enabled by default")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None,
    help="the entity (team) of wandb's project")


# Algorithm specific arguments
parser.add_argument("--num-envs", type=int, default=8,
    help="the number of parallel game environments")
parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
    help="the id of the environment")
parser.add_argument("--total-timesteps", type=int, default=10000000,
    help="total timesteps of the experiments")
parser.add_argument("--learning-rate", type=float, default=1e-4,
    help="the learning rate of the optimizer")

parser.add_argument("--num-steps", type=int, default=32,
    help="the number of steps to run in each environment per policy rollout")
parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Toggle learning rate annealing for policy and value networks")
parser.add_argument("--gamma", type=float, default=1,
    help="the discount factor gamma")
parser.add_argument("--gae-lambda", type=float, default=0.95,
    help="the lambda for the general advantage estimation")
parser.add_argument("--num-minibatches", type=int, default=4,
    help="the number of mini-batches")
parser.add_argument("--update-epochs", type=int, default=4,
    help="the K epochs to update the policy")
parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Toggles advantages normalization")
parser.add_argument("--clip-coef", type=float, default=0.1,
    help="the surrogate clipping coefficient")
parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
parser.add_argument("--ent-coef", type=float, default=0.01,
    help="coefficient of the entropy")
parser.add_argument("--vf-coef", type=float, default=0.5,
    help="coefficient of the value function")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
    help="the maximum norm for the gradient clipping")
parser.add_argument("--target-kl", type=float, default=None,
    help="the target KL divergence threshold")
args = parser.parse_args()
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
