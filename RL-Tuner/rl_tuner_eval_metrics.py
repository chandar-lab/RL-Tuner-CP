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

"""Code to evaluate how well an RL Tuner conforms to music theory rules."""

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm


def compute_composition_stats(rl_tuner,
                              num_compositions=10000,
                              composition_length=32):
  """Uses the model to create many compositions, stores statistics about them.
  Args:
    rl_tuner: An RLTuner object.
    num_compositions: The number of compositions to create.
    composition_length: The number of beats in each composition.
    key: The numeric values of notes belonging to this key. Defaults to
      C-major if not provided.
    tonic_note: The tonic/1st note of the desired key.
  Returns:
    A dictionary containing the computed statistics about the compositions.
  """
  stat_dict = initialize_stat_dict_counterpoint()

  for i in tqdm(range(num_compositions)):
    stat_dict = compose_and_evaluate_piece_counterpoint(
        rl_tuner,
        stat_dict,
        composition_length=composition_length)
    if i % (num_compositions / 10) == 0:
      stat_dict['num_compositions'] = i
      stat_dict['total_notes'] = i * composition_length

  stat_dict['num_compositions'] = num_compositions
  stat_dict['total_notes'] = num_compositions * composition_length
  stat_dict['total_intervals'] = num_compositions * (composition_length - 1)
  compute_constraint_percent_counterpoint(stat_dict)
  compute_rewards(stat_dict)
  return stat_dict
  
def compute_rewards(stat_dict):
  """Computes all the components of the reward fuction based on the number of notes in the sequence.
  Args:
    stat_dict: The current statistics
  """
  stat_dict['rnn_reward'] /= stat_dict['total_notes']
  stat_dict['marginals_reward'] /= stat_dict['total_notes']
  stat_dict['violations_reward'] /= stat_dict['total_notes']
  
def compute_constraint_percent_counterpoint(stat_dict):
  """Computes percentage of constraint satisfaction for each constraint.
  Args:
    stat_dict: The current statistics
  """
  stat_dict['naturalNotesPercent'] = max(0, 1 - (stat_dict['naturalNotes'] / stat_dict['total_notes']))
  stat_dict['intervalsPercent'] = max(0,1 - (stat_dict['intervals'] / stat_dict['total_intervals']))
  stat_dict['tritonOutlinesPercent'] = max(0,1 - (stat_dict['tritonOutlines'] / stat_dict['total_intervals']))
  stat_dict['tonicEndsPercent'] = max(0,1 - (stat_dict['tonicEnds'] / stat_dict['num_compositions']))
  stat_dict['stepwiseDescentToFinalPercent'] = max(0,1 - (stat_dict['stepwiseDescentToFinal'] / stat_dict['num_compositions']))
  stat_dict['noRepeatPercent'] = max(0,1 - (stat_dict['noRepeat'] / stat_dict['total_intervals']))
  stat_dict['coverModalRangePercent'] = max(0,1 - (stat_dict['coverModalRange'] / (stat_dict['num_compositions'] * 41)))
  stat_dict['characteristicModalSkipsPercent'] = max(0, 1 - (stat_dict['characteristicModalSkips']/(stat_dict['num_compositions'] * 3)))
  stat_dict['skipsStepsRatioPercent'] = max(0,1 - (stat_dict['skipsStepsRatio']/(stat_dict['num_compositions'] * 16)))
  stat_dict['avoidSixthsPercent'] = max(0,1 - (stat_dict['avoidSixths']/stat_dict['total_intervals']))
  stat_dict['skipStepsSequencePercent'] = max(0,1 - (stat_dict['skipStepsSequence']/(stat_dict['total_intervals'] * 4)))
  stat_dict['bFlatPercent'] = max(0,1 - (stat_dict['bFlat']/stat_dict['total_intervals']))
  stat_dict['constraint_performance'] = stat_dict['naturalNotesPercent'] + stat_dict['intervalsPercent'] + stat_dict['tritonOutlinesPercent'] + \
                                        stat_dict['tonicEndsPercent'] + stat_dict['stepwiseDescentToFinalPercent'] + stat_dict['noRepeatPercent'] + \
                                        stat_dict['coverModalRangePercent'] + stat_dict['characteristicModalSkipsPercent'] + stat_dict['skipsStepsRatioPercent'] + \
                                        stat_dict['avoidSixthsPercent'] + stat_dict['skipStepsSequencePercent'] + stat_dict['bFlatPercent']
  stat_dict['constraint_performance'] /= 12
  
def compose_and_evaluate_piece_counterpoint(rl_tuner,
                               stat_dict,
                               composition_length=32):
  """Composes a piece using the model, stores statistics about it in a dict.
  Args:
    rl_tuner: An RLTuner object.
    stat_dict: A dictionary storing statistics about a series of compositions.
    composition_length: The number of beats in the composition.
  Returns:
    A dictionary updated to include statistics about the composition just
    created.
  """
  rl_tuner.reset_composition()
  last_observation = rl_tuner.prime_internal_models()


  for _ in range(composition_length):
    action, new_observation, reward_scores = rl_tuner.action(
        last_observation,
        0,
        enable_random=False,
        sample_next_obs=True)

    note_rnn_reward = rl_tuner.reward_from_reward_rnn_scores(new_observation, reward_scores)

    obs_note = np.argmax(new_observation)
    notes = [str(note) for note in rl_tuner.composition]
    marginals = rl_tuner.compute_counterpoint_marginals(notes)
    marginal = marginals[obs_note]
    notes.append(str(obs_note))
    violations = rl_tuner.compute_counterpoint_violations(notes)
    stat_dict['naturalNotes'] += violations[0]
    stat_dict['intervals'] += violations[1]
    stat_dict['tritonOutlines'] += violations[2]
    stat_dict['tonicEnds'] += violations[3]
    stat_dict['stepwiseDescentToFinal'] += violations[4]
    stat_dict['noRepeat'] += violations[5]
    stat_dict['coverModalRange'] += violations[6]
    stat_dict['characteristicModalSkips'] += violations[7]
    stat_dict['skipsStepsRatio'] += violations[8]
    stat_dict['avoidSixths'] += violations[9]
    stat_dict['skipStepsSequence'] += violations[10]
    stat_dict['bFlat'] += violations[11]
    stat_dict['rnn_reward'] += note_rnn_reward
    stat_dict['marginals_reward'] += marginal
    stat_dict['violations_reward'] -= np.sum(violations)
    rl_tuner.composition.append(np.argmax(new_observation))
    rl_tuner.beat += 1
    last_observation = new_observation


  return stat_dict

def initialize_stat_dict_counterpoint():
  """Initializes a dictionary which will hold statistics about compositions.
  Returns:
    A dictionary containing the appropriate fields initialized to 0 or an
    empty list.
  """
  stat_dict = dict()

  stat_dict['naturalNotes'] = 0
  stat_dict['intervals'] = 0
  stat_dict['tritonOutlines'] = 0
  stat_dict['tonicEnds'] = 0
  stat_dict['stepwiseDescentToFinal'] = 0
  stat_dict['noRepeat'] = 0
  stat_dict['coverModalRange'] = 0
  stat_dict['characteristicModalSkips'] = 0
  stat_dict['skipsStepsRatio'] = 0
  stat_dict['avoidSixths'] = 0
  stat_dict['skipStepsSequence'] = 0
  stat_dict['bFlat'] = 0
  stat_dict['num_compositions'] = 0
  stat_dict['total_notes'] = 0
  stat_dict['constraint_performance'] = 0
  stat_dict['rnn_reward'] = 0
  stat_dict['marginals_reward'] = 0
  stat_dict['violations_reward'] = 0

  return stat_dict