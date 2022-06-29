import numpy as np
import imp
import random
import rl_tuner
import rl_tuner_ops
import sys
import wandb

# Place to save your model checkpoints and composi|
SAVE_PATH = "./Results"

# Model parameter settings
seed = int(sys.argv[1])
np.random.seed(seed)
random.seed(seed)
ALGORITHM = 'q'
REWARD_SCALER = 2
CP_REWARD_SCALER = 40
REWARD_MODE = 'rnn_counterpoint_marginals'
RESTRICT_DOMAIN = False
OUTPUT_EVERY_NTH = 5000
num_steps = 50000
NUM_NOTES_IN_COMPOSITION = 32
PRIME_WITH_MIDI = False
exploration_period = 25000
checkpoint_dir = "./checkpoint"
checkpoint_file = checkpoint_dir + "/model.ckpt-1721"
wandb_project = "counterpoint"

class HParams(object):

  def __init__(self, random_action_probability, store_every_nth, train_every_nth,
               minibatch_size, discount_rate, max_experience, target_network_update_rate, rnn_layer_sizes, one_hot_length):

    self.random_action_probability = random_action_probability
    self.store_every_nth = store_every_nth
    self.train_every_nth = train_every_nth
    self.minibatch_size = minibatch_size
    self.discount_rate = discount_rate
    self.max_experience = max_experience
    self.target_network_update_rate = target_network_update_rate
    self.rnn_layer_sizes = rnn_layer_sizes
    self.one_hot_length = one_hot_length

rl_tuner_hparams = HParams(random_action_probability=0.1,
                                               store_every_nth=1,
                                               train_every_nth=5,
                                               minibatch_size=32,
                                               discount_rate=0.5,
                                               max_experience=100000,
                                               target_network_update_rate=0.01,
                                               rnn_layer_sizes = [64, 64],
                                               one_hot_length = 29)

## Train network
run = wandb.init(wandb_project)
imp.reload(rl_tuner_ops)
imp.reload(rl_tuner)
rl_tuner.reload_files()

rl_net = rl_tuner.RLTuner(SAVE_PATH, 
                          note_rnn_checkpoint_dir=checkpoint_dir,
                          note_rnn_checkpoint_file=checkpoint_file,
                          note_rnn_type='basic_rnn',
                          dqn_hparams=rl_tuner_hparams, 
                          reward_mode=REWARD_MODE,
                          restrict_domain=RESTRICT_DOMAIN,
                          algorithm=ALGORITHM,
                          reward_scaler=REWARD_SCALER,
                          cp_reward_scaler=CP_REWARD_SCALER,
                          output_every_nth=OUTPUT_EVERY_NTH,
                          num_notes_in_melody=NUM_NOTES_IN_COMPOSITION,
                          note_rnn_hparams = rl_tuner_hparams)


run.config.seed = seed
run.config.algorithm = ALGORITHM
run.config.reward_scaler = REWARD_SCALER
run.config.cp_reward_scaler = CP_REWARD_SCALER
run.config.reward_mode = REWARD_MODE
run.config.restrict_domain = RESTRICT_DOMAIN
run.config.num_steps = num_steps
run.config.exploration_period = exploration_period
run.config.new = True
run.config.checkpoint_file = checkpoint_file

rl_net.train(num_steps=num_steps, exploration_period=exploration_period)
