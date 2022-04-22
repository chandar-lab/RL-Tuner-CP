# 1. Loads a quantized representation of midi files
# 2. Extracts melodic lines
# 3. Transposes to c major or a minor
# 4. Converts to range between 0 and 28
# 5. Converts back to note sequences
# 6. Creates inputs and labels
# 7. Creates sequence examples
# 8. Saves to file

import pickle
import note_seq
import numpy as np
import tensorflow as tf
from note_seq.protobuf import music_pb2
import random
from scipy.spatial.distance import jensenshannon

def extract_melodic_lines(melodies):
    return [[note for note in melody if note > 1] for melody in melodies]
    
def get_key(melody):
    maj_profile = np.asarray([0.223, 0.006, 0.12, 0.003, 0.154, 0.109, 0.019, 0.189, 0.007, 0.076, 0.005, 0.089])
    maj_profile /= np.sum(maj_profile)
    min_profile = np.asarray([0.189, 0.005, 0.125, 0.144, 0.014, 0.105, 0.021, 0.213, 0.068, 0.02, 0.023, 0.073])
    min_profile /= np.sum(min_profile)
    key_freq = np.zeros((12))
    key_score_major = np.zeros((12))
    key_score_minor = np.zeros((12))
    for note in melody:
        key_freq[(note - 5)%12] += 1
    key_freq /= len(melody)

    for i in range(12):
        key_test = [key_freq[(i + m)%12] for m in range(12)]
        key_score_major[i] = jensenshannon(maj_profile, key_test)
        key_score_minor[i] = jensenshannon(min_profile, key_test)
        #print(key_test, key_score_major[i], key_score_minor[i])
        
    #print(key_score_major, key_score_minor)
        
    best_score_major = np.min(key_score_major)
    best_score_minor = np.min(key_score_minor)
    
    return key_score_major, key_score_minor
    
def c_major(melodies, a_minor):
    c_major_melodies = []
    for melody in melodies:
        transposed_melody = []
        key_score_major, key_score_minor = get_key(melody)
        if np.min(key_score_major) < np.min(key_score_minor):
            key = np.argmin(key_score_major)
            for note in melody:
                transposed_melody.append(note - key)
        else:
            key = (np.argmin(key_score_minor) - 9) % 12
            if a_minor:
                for note in melody:
                    transposed_melody.append(note - key)
        
        
        c_major_melodies.append(transposed_melody)
    
    return c_major_melodies

def convert_to_range(melodies, mode):
    # For soprano voice
    # In the counterpoint CP model: between 0 (G) and 28 (C)
    # In the midi dataset: between 72 (C5) and 96 (C7)
    # In the NoteRNN dataset: between 2 (C5) and 37 (B7)
    # In the pickle dataset: 84 -> 65, 72 -> 53 (between 48 and 77) 65 -> C, 53 -> C, 41 -> C
    # G: 60 72 84...
    range_melodies = []
    g = np.array([6, 20, 34, 48, 60, 72, 84, 98])
    final_middle = (28 - 0)/2 + 0

    for i in range(len(melodies)):
        if len(melodies[i]) > 0:
            range_melody = []
            valid = True
            min = np.min(np.asarray(melodies[i]))
            max = np.max(np.asarray(melodies[i]))
            initial_middle = (max - min)/2 + min
            
            transposed_middle = initial_middle - g
            distances = np.abs(final_middle - transposed_middle)
            
            best_g = np.argmin(distances)
            
            
            transpose = g[best_g]
                
            offset = 0 if mode == 'melodic lines' else 1 if mode == 'no hold' else 2
            for j in range(len(melodies[i])):
                new_value = melodies[i][j] - transpose + offset
                if new_value < offset or new_value > (28 + offset):
                    new_value = -1
                    valid = False
                    print(melodies[i][j], transpose, min, max)

                range_melody.append(new_value)
            
            if (valid):
                range_melodies.append(range_melody)
            else:
                range_melodies.append([])
        else:
            range_melodies.append([])
    
    return range_melodies
    
def extract_note_sequences(melodic_lines, note_sequences, mode):
    for i in range(len(note_sequences)):
        if len(melodic_lines[i]) > 0:
            melodic_line_index = 0
            for j in range(len(note_sequences[i])):
                if j < len(note_sequences[i]) - 1 and note_sequences[i][j] == note_sequences[i][j + 1] and note_sequences[i][j] != 1:
                    print(note_sequences[i])
                    
                if note_sequences[i][j] > 1:
                    note_sequences[i][j] = melodic_lines[i][melodic_line_index]
                    melodic_line_index += 1
                
                if note_sequences[i][j] == 1 and mode == 'no hold':
                    note_sequences[i][j] = melodic_lines[i][melodic_line_index - 1]
                    
        else:
            note_sequences[i] = []
            
    return note_sequences
     
def make_inputs_labels(melodies, mode):
    n_note_values = 29 if mode == 'melodic lines' else 30 if mode == 'no hold' else 31
    inputs = []
    labels = []
    for melody in melodies:
        melody_inputs = []
        melody_labels = []
        for i in range(len(melody) - 1):
            input = np.zeros((n_note_values))
            input[melody[i]] = 1
            label = melody[i + 1]
            melody_inputs.append(input)
            melody_labels.append(label)
        inputs.append(melody_inputs)
        labels.append(melody_labels)
            
    return inputs, labels
     
def make_sequence_examples(inputs, labels):
  # Code from Magenta
  """Returns a SequenceExample for the given inputs and labels.
  Args:
    inputs: A list of input vectors for each melody. Each input vector is a list of floats.
    labels: A list of ints for each melody.
  Returns:
    A list of tf.train.SequenceExample containing inputs and labels.
  """
  sequence_examples = []
  for i in range(len(inputs)):
      melody_inputs = inputs[i]
      melody_labels = labels[i]
      input_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=input_))
          for input_ in melody_inputs]
      label_features = []
      for label in melody_labels:
        label = [label]
        label_features.append(
            tf.train.Feature(int64_list=tf.train.Int64List(value=label)))
      feature_list = {
          'inputs': tf.train.FeatureList(feature=input_features),
          'labels': tf.train.FeatureList(feature=label_features)
      }
      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      sequence_examples.append(tf.train.SequenceExample(feature_lists=feature_lists))
      
  return sequence_examples
  
def main():
  
    train = True
    granularity = 16
    mode = 'no hold'
    a_minor = True
    
    
    # MELODIC_LINES: No silence or hold, notes : 0 to 28, G = 0, A = 2, B = 4, C = 5, 29 classes
    # NOTE_SEQUENCES: Silence and hold, notes: 2 to 30, G = 2, A = 4, B = 6, C = 7, 31 classes
    # NO_HOLD: Silence, notes: 1 to 29, G = 1, A = 3, B = 5, C = 6, 30 classes
    
    input_file_name = 'train_' + str(granularity) + '.pkl' if train else 'test_' + str(granularity) + '.pkl'
    output_file_name = 'train.tfrecord' if train else 'valid.tfrecord'

    with open(input_file_name, 'rb') as file:
        dataset = pickle.load(file)

    all_voices = []
    #all_voices.extend(dataset[0])
    #all_voices.extend(dataset[1])
    #all_voices.extend(dataset[2])
    all_voices.extend(dataset[3])

    if train:
        with open('valid_' + str(granularity) + '.pkl', 'rb') as file:
            dataset_valid = pickle.load(file)
        
        #all_voices.extend(dataset_valid[0])
        #all_voices.extend(dataset_valid[1])
        #all_voices.extend(dataset_valid[2])
        all_voices.extend(dataset_valid[3])
        
    random.seed(42)
    random.shuffle(all_voices)
    #all_voices = all_voices[:1]
    print(all_voices[0])
    print("Note sequences.")
    melodic_lines = extract_melodic_lines(all_voices)
    print(melodic_lines[0])
    print("Transposition..")
    print(len(melodic_lines))
    melodic_lines = c_major(melodic_lines, a_minor)
    print(melodic_lines[0])
    print("Range conversion...")
    melodic_lines = convert_to_range(melodic_lines, mode)
    print(melodic_lines[0])
    
    if mode == 'melodic lines':
        note_sequences = melodic_lines
    else:
        note_sequences = extract_note_sequences(melodic_lines, all_voices, mode)
    
    print(note_sequences[0])
    print("Inputs labels....")
    inputs, labels = make_inputs_labels(note_sequences, mode)
    print("Sequence example.....")
    sequence_examples = make_sequence_examples(inputs, labels)
    print(len(inputs), len(labels))
    print("Writing......A")
    writer = tf.io.TFRecordWriter('updates/no_hold_minor_sixteenth/' + output_file_name)
    for sequence_example in sequence_examples:
        writer.write(sequence_example.SerializeToString())
        
main()
    





