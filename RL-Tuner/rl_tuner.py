# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the main RL Tuner class.
RL Tuner is a Deep Q Network (DQN) with augmented reward to create melodies
by using reinforcement learning to fine-tune a trained Note RNN according
to some music theory rewards.
Also implements two alternatives to Q learning: Psi and G learning. The
algorithm can be switched using the 'algorithm' hyperparameter.
For more information, please consult the README.md file in this directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import random
import urllib
import math

import note_rnn_loader
import rl_tuner_eval_metrics
import rl_tuner_ops
import matplotlib.pyplot as plt
from note_seq import melodies_lib as mlib
from note_seq import midi_io
import numpy as np
import scipy.special
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import reload_module  # pylint: disable=redefined-builtin
from six.moves import urllib  # pylint: disable=redefined-builtin
from subprocess import Popen, PIPE, STDOUT
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import wandb

# Note values of special actions.
NOTE_OFF = 0
NO_EVENT = 1

# Training data sequences are limited to this length, so the padding queue pads
# to this length.
TRAIN_SEQUENCE_LENGTH = 192


def reload_files():
  """Used to reload the imported dependency files (needed for ipynb notebooks).
  """
  reload_module(note_rnn_loader)
  reload_module(rl_tuner_ops)
  reload_module(rl_tuner_eval_metrics)


class RLTuner(object):
  """Implements a recurrent DQN designed to produce melody sequences."""

  def __init__(self, output_dir,

               # Hyperparameters
               dqn_hparams=None,
               reward_mode='music_theory_all',
               restrict_domain=False,
               reward_scaler=1.0,
               cp_reward_scaler=40,
               exploration_mode='egreedy',
               priming_mode='random_note',
               stochastic_observations=False,
               algorithm='q',

               # Trained Note RNN to load and tune
               note_rnn_checkpoint_dir=None,
               note_rnn_checkpoint_file=None,
               note_rnn_type='default',
               note_rnn_hparams=None,

               # Other music related settings.
               num_notes_in_melody=32,
               input_size=rl_tuner_ops.NUM_CLASSES,
               num_actions=rl_tuner_ops.NUM_CLASSES,
               midi_primer=None,

               # Logistics.
               save_name='rl_tuner.ckpt',
               output_every_nth=1000,
               training_file_list=None,
               summary_writer=None,
               initialize_immediately=True):
    """Initializes the MelodyQNetwork class.
    Args:
      output_dir: Where the model will save its compositions (midi files).
      dqn_hparams: A HParams object containing the hyperparameters of
        the DQN algorithm, including minibatch size, exploration probability,
        etc.
      reward_mode: Controls which reward function can be applied. There are
        several, including 'scale', which teaches the model to play a scale,
        and of course 'music_theory_all', which is a music-theory-based reward
        function composed of other functions.
      reward_scaler: Controls the emphasis placed on the music theory rewards.
        This value is the inverse of 'c' in the academic paper.
      exploration_mode: can be 'egreedy' which is an epsilon greedy policy, or
        it can be 'boltzmann', in which the model will sample from its output
        distribution to choose the next action.
      priming_mode: Each time the model begins a new composition, it is primed
        with either a random note ('random_note'), a random MIDI file from the
        training data ('random_midi'), or a particular MIDI file
        ('single_midi').
      stochastic_observations: If False, the note that the model chooses to
        play next (the argmax of its softmax probabilities) deterministically
        becomes the next note it will observe. If True, the next observation
        will be sampled from the model's softmax output.
      algorithm: can be 'default', 'psi', 'g' or 'pure_rl', for different
        learning algorithms
      note_rnn_checkpoint_dir: The directory from which the internal
        NoteRNNLoader will load its checkpointed LSTM.
      note_rnn_checkpoint_file: A checkpoint file to use in case one cannot be
        found in the note_rnn_checkpoint_dir.
      note_rnn_type: If 'default', will use the basic LSTM described in the
        research paper. If 'basic_rnn', will assume the checkpoint is from a
        Magenta basic_rnn model.
      note_rnn_hparams: A HParams object which defines the hyper parameters
        used to train the MelodyRNN model that will be loaded from a checkpoint.
      num_notes_in_melody: The length of a composition of the model
      input_size: the size of the one-hot vector encoding a note that is input
        to the model.
      num_actions: The size of the one-hot vector encoding a note that is
        output by the model.
      midi_primer: A midi file that can be used to prime the model if
        priming_mode is set to 'single_midi'.
      save_name: Name the model will use to save checkpoints.
      output_every_nth: How many training steps before the model will print
        an output saying the cumulative reward, and save a checkpoint.
      training_file_list: A list of paths to tfrecord files containing melody
        training data. This is necessary to use the 'random_midi' priming mode.
      summary_writer: A tf.summary.FileWriter used to log metrics.
      initialize_immediately: if True, the class will instantiate its component
        MelodyRNN networks and build the graph in the constructor.
    """
    # Make graph.
    self.cp_marginals_path = './counterpoint_marginals.jar'
    self.cp_violations_path = './counterpoint_violations.jar'
    self.marginals_path = 'marginals.txt'
    self.violations_path = 'violations.txt'
    mutex_path = 'mutex.txt'
    while os.path.isfile(mutex_path):
      pass
        
    with open(mutex_path, 'w') as f:
      f.write('Hello World')
    self.marginals, self.violations = self.load_marginals()
    
    if os.path.isfile(mutex_path):
      os.remove(mutex_path)
      
    self.graph = tf.Graph()

    with self.graph.as_default():
      # Memorize arguments.
      self.input_size = input_size
      self.num_actions = num_actions
      self.output_every_nth = output_every_nth
      self.output_dir = output_dir
      self.save_path = os.path.join(output_dir, save_name)
      self.reward_scaler = reward_scaler
      self.reward_mode = reward_mode
      self.restrict_domain = restrict_domain
      self.exploration_mode = exploration_mode
      self.num_notes_in_melody = num_notes_in_melody
      self.stochastic_observations = stochastic_observations
      self.algorithm = algorithm
      self.priming_mode = priming_mode
      self.midi_primer = midi_primer
      self.training_file_list = training_file_list
      self.note_rnn_checkpoint_dir = note_rnn_checkpoint_dir
      self.note_rnn_checkpoint_file = note_rnn_checkpoint_file
      self.note_rnn_hparams = note_rnn_hparams
      self.note_rnn_type = note_rnn_type
      self.cp_reward_scaler = cp_reward_scaler
      if priming_mode == 'single_midi' and midi_primer is None:
        tf.logging.fatal('A midi primer file is required when using'
                         'the single_midi priming mode.')
      if note_rnn_checkpoint_dir is None or not note_rnn_checkpoint_dir:
        print('Retrieving checkpoint of Note RNN from Magenta download server.')
        urllib.request.urlretrieve(
            'http://download.magenta.tensorflow.org/models/'
            'rl_tuner_note_rnn.ckpt', 'note_rnn.ckpt')
        self.note_rnn_checkpoint_dir = os.getcwd()
        self.note_rnn_checkpoint_file = os.path.join(os.getcwd(),
                                                     'note_rnn.ckpt')

      if self.note_rnn_hparams is None:
        if self.note_rnn_type == 'basic_rnn':
          self.note_rnn_hparams = rl_tuner_ops.basic_rnn_hparams()
        else:
          self.note_rnn_hparams = rl_tuner_ops.default_hparams()

      if dqn_hparams is None:
        self.dqn_hparams = rl_tuner_ops.default_dqn_hparams()
      else:
        self.dqn_hparams = dqn_hparams
      self.discount_rate = tf.constant(self.dqn_hparams.discount_rate)
      self.target_network_update_rate = tf.constant(
          self.dqn_hparams.target_network_update_rate)

      self.optimizer = tf.train.AdamOptimizer()

      # DQN state.
      self.actions_executed_so_far = 0
      self.experience = collections.deque(
          maxlen=self.dqn_hparams.max_experience)
      self.iteration = 0
      self.summary_writer = summary_writer
      self.num_times_store_called = 0
      self.num_times_train_called = 0

    # Stored reward metrics.
    self.reward_last_n = 0
    self.rewards_batched = []
    self.music_theory_reward_last_n = 0
    self.cp_reward_last_n = 0
    self.music_theory_rewards_batched = []
    self.note_rnn_reward_last_n = 0
    self.note_rnn_rewards_batched = []
    self.eval_avg_reward = []
    self.eval_avg_cp_reward = []
    self.eval_avg_music_theory_reward = []
    self.eval_avg_note_rnn_reward = []

    self.target_val_list = []

    # Variables to keep track of characteristics of the current composition
    # TODO(natashajaques): Implement composition as a class to obtain data
    # encapsulation so that you can't accidentally change the leap direction.
    self.beat = 0
    self.composition = []
    self.composition_direction = 0
    self.leapt_from = None  # stores the note at which composition leapt
    self.steps_since_last_leap = 0

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    if initialize_immediately:
      self.initialize_internal_models_graph_session()
    

  def initialize_internal_models_graph_session(self,
                                               restore_from_checkpoint=True):
    """Initializes internal RNN models, builds the graph, starts the session.
    Adds the graphs of the internal RNN models to this graph, adds the DQN ops
    to the graph, and starts a new Saver and session. By having a separate
    function for this rather than doing it in the constructor, it allows a model
    inheriting from this class to define its q_network differently.
    Args:
      restore_from_checkpoint: If True, the weights for the 'q_network',
        'target_q_network', and 'reward_rnn' will be loaded from a checkpoint.
        If false, these models will be initialized with random weights. Useful
        for checking what pure RL (with no influence from training data) sounds
        like.
    """
    with self.graph.as_default():
      # Add internal networks to the graph.
      tf.logging.info('Initializing q network')
      self.q_network = note_rnn_loader.NoteRNNLoader(
          self.graph, 'q_network',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Initializing target q network')
      self.target_q_network = note_rnn_loader.NoteRNNLoader(
          self.graph,
          'target_q_network',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Initializing reward network')
      self.reward_rnn = note_rnn_loader.NoteRNNLoader(
          self.graph, 'reward_rnn',
          self.note_rnn_checkpoint_dir,
          midi_primer=self.midi_primer,
          training_file_list=self.training_file_list,
          checkpoint_file=self.note_rnn_checkpoint_file,
          hparams=self.note_rnn_hparams,
          note_rnn_type=self.note_rnn_type)

      tf.logging.info('Q network cell: %s', self.q_network.cell)

      # Add rest of variables to graph.
      tf.logging.info('Adding RL graph variables')
      self.build_graph()

      # Prepare saver and session.
      self.saver = tf.train.Saver()
      self.session = tf.Session(graph=self.graph)
      self.session.run(tf.global_variables_initializer())

      # Initialize internal networks.
      if restore_from_checkpoint:
        self.q_network.initialize_and_restore(self.session)
        self.target_q_network.initialize_and_restore(self.session)
        self.reward_rnn.initialize_and_restore(self.session)

        # Double check that the model was initialized from checkpoint properly.
        reward_vars = self.reward_rnn.variables()
        q_vars = self.q_network.variables()
        target_q_vars = self.target_q_network.variables()
        reward_vars = reward_vars[:len(target_q_vars)]

        successful = True
        for i in range(len(reward_vars)):
          reward = self.session.run(reward_vars[i])
          q = self.session.run(q_vars[i])
          target_q = self.session.run(target_q_vars[i])

          if np.sum((q - reward) ** 2) != 0.0 or np.sum((q - target_q) ** 2) != 0.0:
            successful = False

        if successful:
          # TODO(natashamjaques): Remove print statement once tf.logging outputs
          # to Jupyter notebooks (once the following issue is resolved:
          # https://github.com/tensorflow/tensorflow/issues/3047)
          print('\nSuccessfully initialized internal nets from checkpoint!')
          tf.logging.info('\nSuccessfully initialized internal nets from '
                          'checkpoint!')
        else:
          tf.logging.fatal('Error! The model was not initialized from '
                           'checkpoint properly')
      else:
        self.q_network.initialize_new(self.session)
        self.target_q_network.initialize_new(self.session)
        self.reward_rnn.initialize_new(self.session)

    if self.priming_mode == 'random_midi':
      tf.logging.info('Getting priming melodies')
      self.get_priming_melodies()

  def get_priming_melodies(self):
    """Runs a batch of training data through MelodyRNN model.
    If the priming mode is 'random_midi', priming the q-network requires a
    random training melody. Therefore this function runs a batch of data from
    the training directory through the internal model, and the resulting
    internal states of the LSTM are stored in a list. The next note in each
    training melody is also stored in a corresponding list called
    'priming_notes'. Therefore, to prime the model with a random melody, it is
    only necessary to select a random index from 0 to batch_size-1 and use the
    hidden states and note at that index as input to the model.
    """
    (next_note_softmax,
     self.priming_states, lengths) = self.q_network.run_training_batch()

    # Get the next note that was predicted for each priming melody to be used
    # in priming.
    self.priming_notes = [0] * len(lengths)
    for i in range(len(lengths)):
      # Each melody has TRAIN_SEQUENCE_LENGTH outputs, but the last note is
      # actually stored at lengths[i]. The rest is padding.
      start_i = i * TRAIN_SEQUENCE_LENGTH
      end_i = start_i + lengths[i] - 1
      end_softmax = next_note_softmax[end_i, :]
      self.priming_notes[i] = np.argmax(end_softmax)

    tf.logging.info('Stored priming notes: %s', self.priming_notes)

  def prime_internal_model(self, model, restrict_domain=False):
    """Prime an internal model such as the q_network based on priming mode.
    Args:
      model: The internal model that should be primed.
      restrict_domain: The next note will have not-null marginals
    Returns:
      The first observation to feed into the model.
    """
    model.state_value = model.get_zero_state()

    if self.priming_mode == 'random_midi':
      priming_idx = np.random.randint(0, len(self.priming_states))
      model.state_value = np.reshape(
          self.priming_states[priming_idx, :],
          (1, model.cell.state_size))
      priming_note = self.priming_notes[priming_idx]
      next_obs = np.array(
          rl_tuner_ops.make_onehot([priming_note], self.num_actions)).flatten()
      tf.logging.debug(
          'Feeding priming state for midi file %s and corresponding note %s',
          priming_idx, priming_note)
    elif self.priming_mode == 'single_midi':
      model.prime_model()
      next_obs = model.priming_note
    elif self.priming_mode == 'random_note':
      next_obs = self.get_random_note(restrict_domain=restrict_domain)
    else:
      tf.logging.warn('Error! Invalid priming mode. Priming with random note')
      next_obs = self.get_random_note(restrict_domain=restrict_domain)

    return next_obs

  def get_random_note(self, restrict_domain=False):
    """Sample a note uniformly at random.
    Args:
      restrict_domain: The next note will have not-null marginals
    Returns:
      random note
    """
    marginals = self.compute_counterpoint_marginals()
    if restrict_domain:
        domain = np.where(marginals)[0]
        note_idx = np.random.randint(0, self.num_actions)
        if len(domain) > 0:
            index = np.random.randint(0, len(domain))
            note_idx = domain[index]
    else:
        note_idx = np.random.randint(0, self.num_actions)
    return np.array(rl_tuner_ops.make_onehot([note_idx],
                                             self.num_actions)).flatten()

  def reset_composition(self):
    """Starts the models internal composition over at beat 0, with no notes.
    Also resets statistics about whether the composition is in the middle of a
    melodic leap.
    """
    self.beat = 0
    self.composition = []
    self.composition_direction = 0
    self.leapt_from = None
    self.steps_since_last_leap = 0

  def build_graph(self):
    """Builds the reinforcement learning tensorflow graph."""

    tf.logging.info('Adding reward computation portion of the graph')
    with tf.name_scope('reward_computation'):
      self.reward_scores = tf.identity(self.reward_rnn(), name='reward_scores')

    tf.logging.info('Adding taking action portion of graph')
    with tf.name_scope('taking_action'):
      # Output of the q network gives the value of taking each action (playing
      # each note).
      self.action_scores = tf.identity(self.q_network(), name='action_scores')
      tf.summary.histogram(
          'action_scores', self.action_scores)

      # The action values for the G algorithm are computed differently.
      if self.algorithm == 'g':
        self.g_action_scores = self.action_scores + self.reward_scores

        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.g_action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.g_action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                            self.num_actions)
      else:
        # Compute predicted action, which is the argmax of the action scores.
        self.action_softmax = tf.nn.softmax(self.action_scores,
                                            name='action_softmax')
        self.predicted_actions = tf.one_hot(tf.argmax(self.action_scores,
                                                      dimension=1,
                                                      name='predicted_actions'),
                                            self.num_actions)

    tf.logging.info('Add estimating future rewards portion of graph')
    with tf.name_scope('estimating_future_rewards'):
      # The target q network is used to estimate the value of the best action at
      # the state resulting from the current action.
      self.next_action_scores = tf.stop_gradient(self.target_q_network())
      tf.summary.histogram(
          'target_action_scores', self.next_action_scores)

      # Rewards are observed from the environment and are fed in later.
      self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')

      # Each algorithm is attempting to model future rewards with a different
      # function.
      if self.algorithm == 'psi':
        self.target_vals = tf.reduce_logsumexp(self.next_action_scores,
                                               reduction_indices=[1,])
      elif self.algorithm == 'g':
        self.g_normalizer = tf.reduce_logsumexp(self.reward_scores,
                                                reduction_indices=[1,])
        self.g_normalizer = tf.reshape(self.g_normalizer, [-1, 1])
        self.g_normalizer = tf.tile(self.g_normalizer, [1, self.num_actions])
        self.g_action_scores = tf.subtract(
            (self.next_action_scores + self.reward_scores), self.g_normalizer)
        self.target_vals = tf.reduce_logsumexp(self.g_action_scores,
                                               reduction_indices=[1,])
      else:
        # Use default based on Q learning.
        self.target_vals = tf.reduce_max(self.next_action_scores,
                                         reduction_indices=[1,])

      # Total rewards are the observed rewards plus discounted estimated future
      # rewards.
      self.future_rewards = self.rewards + self.discount_rate * self.target_vals

    tf.logging.info('Adding q value prediction portion of graph')
    with tf.name_scope('q_value_prediction'):
      # Action mask will be a one-hot encoding of the action the network
      # actually took.
      self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions),
                                        name='action_mask')
      self.masked_action_scores = tf.reduce_sum(self.action_scores *
                                                self.action_mask,
                                                reduction_indices=[1,])

      temp_diff = self.masked_action_scores - self.future_rewards

      # Prediction error is the mean squared error between the reward the
      # network actually received for a given action, and what it expected to
      # receive.
      self.prediction_error = tf.reduce_mean(tf.square(temp_diff))

      # Compute gradients.
      self.params = tf.trainable_variables()
      self.gradients = self.optimizer.compute_gradients(self.prediction_error)

      # Clip gradients.
      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, 5), var)

      for grad, var in self.gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # Backprop.
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    tf.logging.info('Adding target network update portion of graph')
    with tf.name_scope('target_network_update'):
      # Updates the target_q_network to be similar to the q_network based on
      # the target_network_update_rate.
      self.target_network_update = []
      for v_source, v_target in zip(self.q_network.variables(),
                                    self.target_q_network.variables()):
        # Equivalent to target = (1-alpha) * target + alpha * source
        update_op = v_target.assign_sub(self.target_network_update_rate *
                                        (v_target - v_source))
        self.target_network_update.append(update_op)
      self.target_network_update = tf.group(*self.target_network_update)

    tf.summary.scalar(
        'prediction_error', self.prediction_error)

    self.summarize = tf.summary.merge_all()
    self.no_op1 = tf.no_op()

  def log_stats(self, stats):
    """Uses wandb to log relevant statistics
    Args:
      stats: The statistic dictionary to log
    """
    wandb.log({"naturalNotes": float(stats['naturalNotes'])})
    wandb.log({"intervals": float(stats['intervals'])})
    wandb.log({"tritonOutlines": float(stats['tritonOutlines'])})
    wandb.log({"tonicEnds": float(stats['tonicEnds'])})
    wandb.log({"stepwiseDescentToFinal": float(stats['stepwiseDescentToFinal'])})
    wandb.log({"noRepeat": float(stats['noRepeat'])})
    wandb.log({"coverModalRange": float(stats['coverModalRange'])})
    wandb.log({"characteristicModalSkips": float(stats['characteristicModalSkips'])})
    wandb.log({"skipsStepsRatio": float(stats['skipsStepsRatio'])})
    wandb.log({"avoidSixths": float(stats['avoidSixths'])})
    wandb.log({"skipStepsSequence": float(stats['skipStepsSequence'])})
    wandb.log({"bFlat": float(stats['bFlat'])})
    wandb.log({"naturalNotesPercent": float(stats['naturalNotesPercent'])})
    wandb.log({"intervalsPercent": float(stats['intervalsPercent'])})
    wandb.log({"tritonOutlinesPercent": float(stats['tritonOutlinesPercent'])})
    wandb.log({"tonicEndsPercent": float(stats['tonicEndsPercent'])})
    wandb.log({"stepwiseDescentToFinalPercent": float(stats['stepwiseDescentToFinalPercent'])})
    wandb.log({"noRepeatPercent": float(stats['noRepeatPercent'])})
    wandb.log({"coverModalRangePercent": float(stats['coverModalRangePercent'])})
    wandb.log({"characteristicModalSkipsPercent": float(stats['characteristicModalSkipsPercent'])})
    wandb.log({"skipsStepsRatioPercent": float(stats['skipsStepsRatioPercent'])})
    wandb.log({"avoidSixthsPercent": float(stats['avoidSixthsPercent'])})
    wandb.log({"skipStepsSequencePercent": float(stats['skipStepsSequencePercent'])})
    wandb.log({"bFlatPercent": float(stats['bFlatPercent'])})
    wandb.log({"rnnReward": float(stats['rnn_reward'])})
    wandb.log({"marginalsReward": float(stats['marginals_reward'])})
    wandb.log({"violationsReward": float(stats['violations_reward'])})
    wandb.log({"constraint_performance": float(stats['constraint_performance'])})

  def train(self, num_steps=10000, exploration_period=5000, enable_random=True):
    """Main training function that allows model to act, collects reward, trains.
    Iterates a number of times, getting the model to act each time, saving the
    experience, and performing backprop.
    Args:
      num_steps: The number of training steps to execute.
      exploration_period: The number of steps over which the probability of
        exploring (taking a random action) is annealed from 1.0 to the model's
        random_action_probability.
      enable_random: If False, the model will not be able to act randomly /
        explore.
    """
    tf.logging.info('Evaluating initial model...')
    stats = rl_tuner_eval_metrics.compute_composition_stats(self,
        num_compositions=10,
        composition_length=self.num_notes_in_melody)
    self.log_stats(stats)
    wandb.log({"timesteps": 0})


    self.actions_executed_so_far = 0

    if self.stochastic_observations:
      tf.logging.info('Using stochastic environment')

    sample_next_obs = False
    if self.exploration_mode == 'boltzmann' or self.stochastic_observations:
      sample_next_obs = True

    self.reset_composition()
    last_observation = self.prime_internal_models(restrict_domain=self.restrict_domain)

    for i in tqdm(range(num_steps)):
      # Experiencing observation, state, action, reward, new observation,
      # new state tuples, and storing them.
      state = np.array(self.q_network.state_value).flatten()

      action, new_observation, reward_scores = self.action(
          last_observation, exploration_period, enable_random=enable_random,
          sample_next_obs=sample_next_obs, restrict_domain=self.restrict_domain)

      new_state = np.array(self.q_network.state_value).flatten()
      new_reward_state = np.array(self.reward_rnn.state_value).flatten()

      reward = self.collect_reward(last_observation, new_observation,
                                   reward_scores)

      self.store(last_observation, state, action, reward, new_observation,
                 new_state, new_reward_state)

      # Used to keep track of how the reward is changing over time.
      self.reward_last_n += reward

      # Used to keep track of the current musical composition and beat for
      # the reward functions.
      self.composition.append(np.argmax(new_observation))
      self.beat += 1

      if i > 0 and i % self.output_every_nth == 0:
        tf.logging.info('Evaluating model...')
        stats = rl_tuner_eval_metrics.compute_composition_stats(self,
                                                                num_compositions=10,
                                                                composition_length=self.num_notes_in_melody)
        self.log_stats(stats)
        wandb.log({"timesteps": i})
        #self.save_model(self.algorithm)

        if self.algorithm == 'g':
          self.rewards_batched.append(
              self.music_theory_reward_last_n + self.note_rnn_reward_last_n)
        else:
          self.rewards_batched.append(self.reward_last_n)
        self.music_theory_rewards_batched.append(
            self.music_theory_reward_last_n)
        self.note_rnn_rewards_batched.append(self.note_rnn_reward_last_n)

        # Save a checkpoint.
        save_step = len(self.rewards_batched)*self.output_every_nth
        #self.saver.save(self.session, self.save_path, global_step=save_step)

        r = self.reward_last_n
        tf.logging.info('Training iteration %s', i)
        tf.logging.info('\tReward for last %s steps: %s',
                        self.output_every_nth, r)
        tf.logging.info('\t\tCP reward: %s',
                        self.cp_reward_last_n)
        tf.logging.info('\t\tMusic theory reward: %s',
                        self.music_theory_reward_last_n)
        tf.logging.info('\t\tNote RNN reward: %s', self.note_rnn_reward_last_n)

        # TODO(natashamjaques): Remove print statement once tf.logging outputs
        # to Jupyter notebooks (once the following issue is resolved:
        # https://github.com/tensorflow/tensorflow/issues/3047)
        print('Training iteration', i)
        print('\tReward for last', self.output_every_nth, 'steps:', r)
        print('\t\tCP reward:', self.cp_reward_last_n)
        print('\t\tMusic theory reward:', self.music_theory_reward_last_n)
        print('\t\tNote RNN reward:', self.note_rnn_reward_last_n)

        if self.exploration_mode == 'egreedy':
          exploration_p = rl_tuner_ops.linear_annealing(
              self.actions_executed_so_far, exploration_period, 1.0,
              self.dqn_hparams.random_action_probability)
          tf.logging.info('\tExploration probability is %s', exploration_p)

        self.reward_last_n = 0
        self.music_theory_reward_last_n = 0
        self.note_rnn_reward_last_n = 0
        self.cp_reward_last_n = 0

      # Backprop.
      self.training_step()

      # Update current state as last state.
      last_observation = new_observation

      # Reset the state after each composition is complete.
      if self.beat % self.num_notes_in_melody == 0:
        tf.logging.debug('\nResetting composition!\n')
        self.reset_composition()
        last_observation = self.prime_internal_models(restrict_domain=self.restrict_domain)

    self.save_marginals()

  def action(self, observation, exploration_period=0, enable_random=True,
             sample_next_obs=False, restrict_domain=False):
    """Given an observation, runs the q_network to choose the current action.
    Does not backprop.
    Args:
      observation: A one-hot encoding of a single observation (note).
      exploration_period: The total length of the period the network will
        spend exploring, as set in the train function.
      enable_random: If False, the network cannot act randomly.
      sample_next_obs: If True, the next observation will be sampled from
        the softmax probabilities produced by the model, and passed back
        along with the action. If False, only the action is passed back.
    Returns:
      The action chosen, the reward_scores returned by the reward_rnn, and the
      next observation. If sample_next_obs is False, the next observation is
      equal to the action.
    """
    assert len(observation.shape) == 1, 'Single observation only'

    self.actions_executed_so_far += 1

    if self.exploration_mode == 'egreedy':
      # Compute the exploration probability.
      exploration_p = rl_tuner_ops.linear_annealing(
          self.actions_executed_so_far, exploration_period, 1.0,
          self.dqn_hparams.random_action_probability)
    elif self.exploration_mode == 'boltzmann':
      enable_random = False
      sample_next_obs = True

    # Run the observation through the q_network.
    input_batch = np.reshape(observation,
                             (self.q_network.batch_size, 1, self.input_size))
    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    (action, action_softmax, self.q_network.state_value,
     reward_scores, self.reward_rnn.state_value) = self.session.run(
         [self.predicted_actions, self.action_softmax,
          self.q_network.state_tensor, self.reward_scores,
          self.reward_rnn.state_tensor],
         {self.q_network.melody_sequence: input_batch,
          self.q_network.initial_state: self.q_network.state_value,
          self.q_network.lengths: lengths,
          self.reward_rnn.melody_sequence: input_batch,
          self.reward_rnn.initial_state: self.reward_rnn.state_value,
          self.reward_rnn.lengths: lengths})

    reward_scores = np.reshape(reward_scores, (self.num_actions))
    action_softmax = np.reshape(action_softmax, (self.num_actions))
    action = np.reshape(action, (self.num_actions))

    if enable_random and random.random() < exploration_p:
      note = self.get_random_note(restrict_domain=restrict_domain)
      return note, note, reward_scores
    else:
      marginals = self.compute_counterpoint_marginals()
      if not sample_next_obs:
        if restrict_domain:
            is_valid = np.asarray(marginals) > 0
            valid_actions = action_softmax
            if np.sum(is_valid) > 0:
                valid_actions = action_softmax * is_valid
            note = np.argmax(valid_actions)
            action = np.array(rl_tuner_ops.make_onehot([note],
                                              self.num_actions)).flatten()
        return action, action, reward_scores
      else:
        if restrict_domain:
            is_valid = np.asarray(marginals) > 0
            if np.sum(is_valid) > 0:
                action_softmax = action_softmax * is_valid
        obs_note = rl_tuner_ops.sample_softmax(action_softmax)
        next_obs = np.array(
            rl_tuner_ops.make_onehot([obs_note], self.num_actions)).flatten()
        return action, next_obs, reward_scores

  def store(self, observation, state, action, reward, newobservation, newstate,
            new_reward_state):
    """Stores an experience in the model's experience replay buffer.
    One experience consists of an initial observation and internal LSTM state,
    which led to the execution of an action, the receipt of a reward, and
    finally a new observation and a new LSTM internal state.
    Args:
      observation: A one hot encoding of an observed note.
      state: The internal state of the q_network MelodyRNN LSTM model.
      action: A one hot encoding of action taken by network.
      reward: Reward received for taking the action.
      newobservation: The next observation that resulted from the action.
        Unless stochastic_observations is True, the action and new
        observation will be the same.
      newstate: The internal state of the q_network MelodyRNN that is
        observed after taking the action.
      new_reward_state: The internal state of the reward_rnn network that is
        observed after taking the action
    """
    if self.num_times_store_called % self.dqn_hparams.store_every_nth == 0:
      self.experience.append((observation, state, action, reward,
                              newobservation, newstate, new_reward_state))
    self.num_times_store_called += 1

  def training_step(self):
    """Backpropagate prediction error from a randomly sampled experience batch.
    A minibatch of experiences is randomly sampled from the model's experience
    replay buffer and used to update the weights of the q_network and
    target_q_network.
    """
    if self.num_times_train_called % self.dqn_hparams.train_every_nth == 0:
      if len(self.experience) < self.dqn_hparams.minibatch_size:
        return

      # Sample experience.
      samples = random.sample(range(len(self.experience)),
                              self.dqn_hparams.minibatch_size)
      samples = [self.experience[i] for i in samples]

      # Batch states.
      states = np.empty((len(samples), self.q_network.cell.state_size))
      new_states = np.empty((len(samples),
                             self.target_q_network.cell.state_size))
      reward_new_states = np.empty((len(samples),
                                    self.reward_rnn.cell.state_size))
      observations = np.empty((len(samples), self.input_size))
      new_observations = np.empty((len(samples), self.input_size))
      action_mask = np.zeros((len(samples), self.num_actions))
      rewards = np.empty((len(samples),))
      lengths = np.full(len(samples), 1, dtype=int)

      for i, (o, s, a, r, new_o, new_s, reward_s) in enumerate(samples):
        observations[i, :] = o
        new_observations[i, :] = new_o
        states[i, :] = s
        new_states[i, :] = new_s
        action_mask[i, :] = a
        rewards[i] = r
        reward_new_states[i, :] = reward_s

      observations = np.reshape(observations,
                                (len(samples), 1, self.input_size))
      new_observations = np.reshape(new_observations,
                                    (len(samples), 1, self.input_size))

      calc_summaries = self.iteration % 100 == 0
      calc_summaries = calc_summaries and self.summary_writer is not None

      if self.algorithm == 'g':
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.reward_rnn.melody_sequence: new_observations,
            self.reward_rnn.initial_state: reward_new_states,
            self.reward_rnn.lengths: lengths,
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })
      else:
        _, _, target_vals, summary_str = self.session.run([
            self.prediction_error,
            self.train_op,
            self.target_vals,
            self.summarize if calc_summaries else self.no_op1,
        ], {
            self.q_network.melody_sequence: observations,
            self.q_network.initial_state: states,
            self.q_network.lengths: lengths,
            self.target_q_network.melody_sequence: new_observations,
            self.target_q_network.initial_state: new_states,
            self.target_q_network.lengths: lengths,
            self.action_mask: action_mask,
            self.rewards: rewards,
        })

      total_logs = (self.iteration * self.dqn_hparams.train_every_nth)
      if total_logs % self.output_every_nth == 0:
        self.target_val_list.append(np.mean(target_vals))

      self.session.run(self.target_network_update)

      if calc_summaries:
        self.summary_writer.add_summary(summary_str, self.iteration)

      self.iteration += 1

    self.num_times_train_called += 1

  def collect_reward(self, action, reward_scores):
    """Calls whatever reward function is indicated in the reward_mode field.
    New reward functions can be written and called from here. Note that the
    reward functions can make use of the musical composition that has been
    played so far, which is stored in self.composition. Some reward functions
    are made up of many smaller functions, such as those related to music
    theory.
    Args:
      action: A one-hot encoding of the chosen action.
      reward_scores: The value for each note output by the reward_rnn.
    Returns:
      Float reward value.
    """
    # Gets and saves log p(a|s) as output by reward_rnn.
    note_rnn_reward = self.reward_from_reward_rnn_scores(action, reward_scores)
    self.note_rnn_reward_last_n += note_rnn_reward
    
    if self.reward_mode == 'counterpoint_marginals_violations':
      chosen_note = np.argmax(action)
      marginals = self.get_counterpoint_marginals()
      violations = self.get_counterpoint_violations(np.argmax(action))
      return np.sum(violations) * -1 + marginals[chosen_note] * self.cp_reward_scaler
      
    elif self.reward_mode == 'counterpoint_marginals':
      chosen_note = np.argmax(action)
      marginals = self.get_counterpoint_marginals()
      return marginals[chosen_note] * self.cp_reward_scaler
      
    elif self.reward_mode == 'counterpoint_violations':
      violations = self.get_counterpoint_violations(np.argmax(action))
      return  np.sum(violations) * -1
      
    elif self.reward_mode == 'rnn_counterpoint_marginals_violations':
      chosen_note = np.argmax(action)
      marginals = self.get_counterpoint_marginals()
      violations = self.get_counterpoint_violations(np.argmax(action))
      reward = np.sum(violations) * -1 + marginals[chosen_note] * self.cp_reward_scaler
      return reward * self.reward_scaler + note_rnn_reward
      
    elif self.reward_mode == 'rnn_counterpoint_marginals':
      chosen_note = np.argmax(action)
      marginals = self.get_counterpoint_marginals()
      reward = marginals[chosen_note] * self.cp_reward_scaler
      return reward * self.reward_scaler + note_rnn_reward
      
    elif self.reward_mode == 'rnn_counterpoint_violations':
      violations = self.get_counterpoint_violations(np.argmax(action))
      reward = np.sum(violations) * -1
      return reward * self.reward_scaler + note_rnn_reward
      
    elif self.reward_mode == 'rnn':
      return note_rnn_reward
    
    else:
      tf.logging.fatal('ERROR! Not a valid reward mode. Cannot compute reward')

  def reward_from_reward_rnn_scores(self, action, reward_scores):
    """Rewards based on probabilities learned from data by trained RNN.
    Computes the reward_network's learned softmax probabilities. When used as
    rewards, allows the model to maintain information it learned from data.
    Args:
      action: A one-hot encoding of the chosen action.
      reward_scores: The value for each note output by the reward_rnn.
    Returns:
      Float reward value.
    """
    action_note = np.argmax(action)
    normalization_constant = scipy.special.logsumexp(reward_scores)
    return reward_scores[action_note] - normalization_constant

  def get_reward_rnn_scores(self, observation, state):
    """Get note scores from the reward_rnn to use as a reward based on data.
    Runs the reward_rnn on an observation and initial state. Useful for
    maintaining the probabilities of the original LSTM model while training with
    reinforcement learning.
    Args:
      observation: One-hot encoding of the observed note.
      state: Vector representing the internal state of the target_q_network
        LSTM.
    Returns:
      Action scores produced by reward_rnn.
    """
    state = np.atleast_2d(state)

    input_batch = np.reshape(observation, (self.reward_rnn.batch_size, 1,
                                           self.num_actions))
    lengths = np.full(self.reward_rnn.batch_size, 1, dtype=int)

    rewards, = self.session.run(
        self.reward_scores,
        {self.reward_rnn.melody_sequence: input_batch,
         self.reward_rnn.initial_state: state,
         self.reward_rnn.lengths: lengths})
    return rewards

  def load_marginals(self):
    """Loads the marginals and the violations dictionary from the txt files
    Returns:
      Marginals and violations dictionary
    """
    with open(self.marginals_path) as f:
        marginals = json.load(f)
        
    with open(self.violations_path) as f2:
        violations = json.load(f2)

    return marginals, violations

  def save_marginals(self):
      """Saves the marginals and the violations dictionary in the txt files.
      """

      mutex_path = 'mutex.txt'
      while os.path.isfile(mutex_path):
          pass

      with open(mutex_path, 'w') as f:
          f.write('Hello World')

      print("Saving marginals")
      new_marginals, new_violations = self.load_marginals()
      final_marginals = {**self.marginals, **new_marginals}
      final_violations = {**self.violations, **new_violations}

      with open(self.marginals_path, 'w') as f:
          json.dump(final_marginals, f)
          
      with open(self.violations_path, 'w') as f2:
          json.dump(final_violations, f2)

      print("The marginals and violations have been saved!")
      os.remove(mutex_path)
      
  def get_counterpoint_marginals(self):
      """Obtains the precomputed marginals for the current composition
      Returns:
        Marginals"""

      notes = [str(note) for note in self.composition]
      separator = "_"
      key = separator.join(notes)
      
      marginals_string = self.marginals[key]
      marginals_array = list(map(float, marginals_string.split()))
      
      return marginals_array
      
  def get_counterpoint_violations(self, chosen_note):
      """Obtains and computes the violations array if the marginals are 0 (otherwise, no violations occur)
      Args:
        chosen_note: The action chosen by the agent
      Returns:
        The violations array"""
      notes = [str(note) for note in self.composition]
      separator = "_"
      key = separator.join(notes)
      
      marginals_string = self.marginals[key]
      marginals_array = list(map(float, marginals_string.split()))
      
      if marginals_array[chosen_note] > 0:
        return np.zeros((12))
      
      notes.append(str(chosen_note))
      return self.compute_counterpoint_violations(notes)
      
  def compute_counterpoint_violations(self, notes):
      """Computes the violations array for the current note sequence
      Args:
        notes: the current note sequence including the chosen action
      Returns:
        The violations array"""
      separator = "_"
      key = separator.join(notes)
      
      if key in self.violations:
        violations_string = self.violations[key]
        return list(map(float, violations_string.split()))
        
      jar_array = ['java', '-jar', self.cp_violations_path]
      jar_array.extend(notes)
      p = Popen(jar_array, stdout=PIPE, stderr=STDOUT)
      jar_output = p.stdout.read()
      p.stdout.close()
      try:
        violations = jar_output.decode('utf-8')
        self.violations[key] = violations
        violation_list = list(map(float, violations.split()))
      except:
        print(jar_output)
        print(jar_array)
        raise("Conversion error")
      return violation_list
      
  def compute_counterpoint_marginals(self, notes = None):
      """Computes the marginals array for the current note sequence
      Args:
        notes: the current note sequence (if None, use the composition)
      Returns:
        The marginals array"""
      if not notes:
        notes = self.composition
      notes = [str(note) for note in notes]
      separator = "_"
      key = separator.join(notes)

      if key in self.marginals:
            marginal_string = self.marginals[key]
            return list(map(float, marginal_string.split()))

      # If the key doesn't exist, reading the marginal from jar output
      jar_array = ['java', '-jar', self.cp_marginals_path]
      jar_array.extend(notes)
      p = Popen(jar_array, stdout=PIPE, stderr=STDOUT)
      jar_output = p.stdout.read()
      p.stdout.close()
      try:
        marginals = jar_output.decode('utf-8')
        self.marginals[key] = marginals
        marginal_list = list(map(float, marginals.split()))
      except:
        print(jar_output)
        print(jar_array)
        raise("Conversion error")
      return marginal_list

  def generate_music_sequence(self, save_to_file=True, name='rltuner_sample', visualize_probs=False,
                              prob_image_name=None, length=None, most_probable=False):
    """Generates a music sequence with the current model and returns it. The
    sequence is generated by sampling from the output probabilities at each
    timestep, and feeding the resulting note back in as input to the model.
    Args:
      save_to_file: If True, the sequence will be saved in a MIDI file in the
        model's output_dir directory
      name: The name that will be used to save the output MIDI file if necessary
      visualize_probs: If True, the function will plot the softmax
        probabilities of the model for each note that occur throughout the
        sequence. Useful for debugging.
      prob_image_name: The name of a file in which to save the softmax
        probability image. If None, the image will simply be displayed.
      length: The length of the sequence to be generated. Defaults to the
        num_notes_in_melody parameter of the model.
      most_probable: If True, instead of sampling each note in the sequence,
        the model will always choose the argmax, most probable note.
    """
    if length is None:
      length = self.num_notes_in_melody

    self.reset_composition()
    next_obs = self.prime_internal_models() # Testing the model
    tf.logging.info('Priming with note %s', np.argmax(next_obs))

    lengths = np.full(self.q_network.batch_size, 1, dtype=int)

    if visualize_probs:
      prob_image = np.zeros((self.input_size, length))

    generated_seq = [0] * length
    for i in range(length):
      input_batch = np.reshape(next_obs, (self.q_network.batch_size, 1,
                                          self.num_actions))
      if self.algorithm == 'g':
        (softmax, self.q_network.state_value,
         self.reward_rnn.state_value) = self.session.run(
             [self.action_softmax, self.q_network.state_tensor,
              self.reward_rnn.state_tensor],
             {self.q_network.melody_sequence: input_batch,
              self.q_network.initial_state: self.q_network.state_value,
              self.q_network.lengths: lengths,
              self.reward_rnn.melody_sequence: input_batch,
              self.reward_rnn.initial_state: self.reward_rnn.state_value,
              self.reward_rnn.lengths: lengths})
      else:
        softmax, self.q_network.state_value = self.session.run(
            [self.action_softmax, self.q_network.state_tensor],
            {self.q_network.melody_sequence: input_batch,
             self.q_network.initial_state: self.q_network.state_value,
             self.q_network.lengths: lengths})
      softmax = np.reshape(softmax, (self.num_actions))

      if visualize_probs:
        prob_image[:, i] = softmax  # np.log(1.0 + softmax)

      if most_probable:
        sample = np.argmax(softmax)
      else:
        sample = rl_tuner_ops.sample_softmax(softmax)
      generated_seq[i] = sample
      next_obs = np.array(rl_tuner_ops.make_onehot([sample],
                                                   self.num_actions)).flatten()

    tf.logging.info('Generated sequence: %s', generated_seq)
    # TODO(natashamjaques): Remove print statement once tf.logging outputs
    # to Jupyter notebooks (once the following issue is resolved:
    # https://github.com/tensorflow/tensorflow/issues/3047)
    print('Generated sequence:', generated_seq)

    melody = mlib.Melody(rl_tuner_ops.decoder(generated_seq,
                                              self.q_network.transpose_amount))

    sequence = melody.to_sequence(qpm=rl_tuner_ops.DEFAULT_QPM)
    if save_to_file:
      filename = rl_tuner_ops.get_next_file_name(self.output_dir, name, 'mid')
      midi_io.sequence_proto_to_midi_file(sequence, filename)

      tf.logging.info('Wrote a melody to %s', self.output_dir)

    if visualize_probs:
      tf.logging.info('Visualizing note selection probabilities:')
      plt.figure()
      plt.imshow(prob_image, interpolation='none', cmap='Reds')
      plt.ylabel('Note probability')
      plt.xlabel('Time (beat)')
      plt.gca().invert_yaxis()
      if prob_image_name is not None:
        plt.savefig(self.output_dir + '/' + prob_image_name)
      else:
        plt.show()

    return sequence


  def evaluate_music_theory_metrics(self, num_compositions=10000):
    """Computes statistics about music theory rule adherence.
    Args:
      num_compositions: How many compositions should be randomly generated
        for computing the statistics.
    Returns:
      A dictionary containing the statistics.
    """
    stat_dict = rl_tuner_eval_metrics.compute_composition_stats(self, num_compositions, self.num_notes_in_melody)

    return stat_dict

  def save_model(self, name, directory=None):
    """Saves a checkpoint of the model and a .npz file with stored rewards.
    Args:
      name: String name to use for the checkpoint and rewards files.
      directory: Path to directory where the data will be saved. Defaults to
        self.output_dir if None is provided.
    """
    if directory is None:
      directory = self.output_dir

    save_loc = os.path.join(directory, name)
    self.saver.save(self.session, save_loc,
                    global_step=len(self.rewards_batched)*self.output_every_nth)

    self.save_stored_rewards(name)

  def save_stored_rewards(self, file_name):
    """Saves the models stored rewards over time in a .npz file.
    Args:
      file_name: Name of the file that will be saved.
    """
    training_epochs = len(self.rewards_batched) * self.output_every_nth
    filename = os.path.join(self.output_dir,
                            file_name + '-' + str(training_epochs))
    np.savez(filename,
             train_rewards=self.rewards_batched,
             train_music_theory_rewards=self.music_theory_rewards_batched,
             train_note_rnn_rewards=self.note_rnn_rewards_batched,
             eval_rewards=self.eval_avg_reward,
             eval_music_theory_rewards=self.eval_avg_music_theory_reward,
             eval_note_rnn_rewards=self.eval_avg_note_rnn_reward,
             target_val_list=self.target_val_list)

  def save_model_and_figs(self, name, directory=None):
    """Saves the model checkpoint, .npz file, and reward plots.
    Args:
      name: Name of the model that will be used on the images,
        checkpoint, and .npz files.
      directory: Path to directory where files will be saved.
        If None defaults to self.output_dir.
    """

    self.save_model(name, directory=directory)
    self.plot_rewards(image_name='TrainRewards-' + name + '.eps',
                      directory=directory)
    self.plot_evaluation(image_name='EvaluationRewards-' + name + '.eps',
                         directory=directory)
    self.plot_target_vals(image_name='TargetVals-' + name + '.eps',
                          directory=directory)

  def plot_rewards(self, image_name=None, directory=None):
    """Plots the cumulative rewards received as the model was trained.
    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.
    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.rewards_batched))]
    plt.figure()
    plt.plot(x, self.rewards_batched)
    plt.plot(x, self.music_theory_rewards_batched)
    plt.plot(x, self.note_rnn_rewards_batched)
    plt.xlabel('Training epoch')
    plt.ylabel('Cumulative reward for last ' + str(reward_batch) + ' steps')
    plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_evaluation(self, image_name=None, directory=None, start_at_epoch=0):
    """Plots the rewards received as the model was evaluated during training.
    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.
    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
      start_at_epoch: Training epoch where the plot should begin.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.eval_avg_reward))]
    start_index = math.floor(start_at_epoch / self.output_every_nth)
    plt.figure()
    plt.plot(x[start_index:], self.eval_avg_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_music_theory_reward[start_index:])
    plt.plot(x[start_index:], self.eval_avg_note_rnn_reward[start_index:])
    plt.xlabel('Training epoch')
    plt.ylabel('Average reward')
    plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def plot_target_vals(self, image_name=None, directory=None):
    """Plots the target values used to train the model over time.
    If image_name is None, should be used in jupyter notebook. If
    called outside of jupyter, execution of the program will halt and
    a pop-up with the graph will appear. Execution will not continue
    until the pop-up is closed.
    Args:
      image_name: Name to use when saving the plot to a file. If not
        provided, image will be shown immediately.
      directory: Path to directory where figure should be saved. If
        None, defaults to self.output_dir.
    """
    if directory is None:
      directory = self.output_dir

    reward_batch = self.output_every_nth
    x = [reward_batch * i for i in np.arange(len(self.target_val_list))]

    plt.figure()
    plt.plot(x, self.target_val_list)
    plt.xlabel('Training epoch')
    plt.ylabel('Target value')
    if image_name is not None:
      plt.savefig(directory + '/' + image_name)
    else:
      plt.show()

  def prime_internal_models(self, restrict_domain=False):
    """Primes both internal models based on self.priming_mode.
    Returns:
      A one-hot encoding of the note output by the q_network to be used as
      the initial observation.
    """
    self.prime_internal_model(self.target_q_network, restrict_domain=restrict_domain)
    self.prime_internal_model(self.reward_rnn, restrict_domain=restrict_domain)
    next_obs = self.prime_internal_model(self.q_network, restrict_domain=restrict_domain)
    return next_obs

  def restore_from_directory(self, directory=None, checkpoint_name=None,
                             reward_file_name=None):
    """Restores this model from a saved checkpoint.
    Args:
      directory: Path to directory where checkpoint is located. If
        None, defaults to self.output_dir.
      checkpoint_name: The name of the checkpoint within the
        directory.
      reward_file_name: The name of the .npz file where the stored
        rewards are saved. If None, will not attempt to load stored
        rewards.
    """
    if directory is None:
      directory = self.output_dir

    if checkpoint_name is not None:
      checkpoint_file = os.path.join(directory, checkpoint_name)
    else:
      tf.logging.info('Directory %s.', directory)
      checkpoint_file = tf.train.latest_checkpoint(directory)

    if checkpoint_file is None:
      tf.logging.fatal('Error! Cannot locate checkpoint in the directory')
      return
    # TODO(natashamjaques): Remove print statement once tf.logging outputs
    # to Jupyter notebooks (once the following issue is resolved:
    # https://github.com/tensorflow/tensorflow/issues/3047)
    print('Attempting to restore from checkpoint', checkpoint_file)
    tf.logging.info('Attempting to restore from checkpoint %s', checkpoint_file)

    self.saver.restore(self.session, checkpoint_file)

    if reward_file_name is not None:
      npz_file_name = os.path.join(directory, reward_file_name)
      # TODO(natashamjaques): Remove print statement once tf.logging outputs
      # to Jupyter notebooks (once the following issue is resolved:
      # https://github.com/tensorflow/tensorflow/issues/3047)
      print('Attempting to load saved reward values from file', npz_file_name)
      tf.logging.info('Attempting to load saved reward values from file %s',
                      npz_file_name)
      npz_file = np.load(npz_file_name)

      self.rewards_batched = npz_file['train_rewards']
      self.music_theory_rewards_batched = npz_file['train_music_theory_rewards']
      self.note_rnn_rewards_batched = npz_file['train_note_rnn_rewards']
      self.eval_avg_reward = npz_file['eval_rewards']
      self.eval_avg_music_theory_reward = npz_file['eval_music_theory_rewards']
      self.eval_avg_note_rnn_reward = npz_file['eval_note_rnn_rewards']
      self.target_val_list = npz_file['target_val_list']
