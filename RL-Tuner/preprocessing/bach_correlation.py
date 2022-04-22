import pickle
import numpy as np
from bach_note_sequences import extract_melodic_lines, c_major, convert_to_range, extract_note_sequences
from melody_rnn_model import *
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from note_seq.protobuf import music_pb2
from scipy.spatial.distance import jensenshannon

def convert_notesequence_to_array(note_sequence):
    notes = []
    time = 0
    for note in note_sequence.notes:
        while note.start_time > time:
            notes.append(0)
            time += 0.5
            
        notes.append(note.pitch)
        time += 0.5
        
    return notes
            
            
granularity = 16
mode = 'no hold'
a_minor = True


with open('train_' + str(granularity) + '.pkl', 'rb') as file:
    dataset = pickle.load(file)

all_voices = []
#all_voices.extend(dataset[0])
#all_voices.extend(dataset[1])
#all_voices.extend(dataset[2])
all_voices.extend(dataset[3])

with open('valid_' + str(granularity) + '.pkl', 'rb') as file:
    dataset_valid = pickle.load(file)
    
#all_voices.extend(dataset_valid[0])
#all_voices.extend(dataset_valid[1])
#all_voices.extend(dataset_valid[2])
all_voices.extend(dataset_valid[3])

with open('test_' + str(granularity) + '.pkl', 'rb') as file:
    dataset_valid = pickle.load(file)
    
#all_voices.extend(dataset_valid[0])
#all_voices.extend(dataset_valid[1])
#all_voices.extend(dataset_valid[2])
all_voices.extend(dataset_valid[3])
    
#print(all_voices)
print("Note sequences.")
melodic_lines = extract_melodic_lines(all_voices)
#print(note_sequences)
print("Transposition..")
print(len(melodic_lines))
#print([get_key(m) for m in note_sequences])
melodic_lines = c_major(melodic_lines, a_minor)
#print(note_sequences)
#print([get_key(m) for m in note_sequences])
print("Range conversion...")
melodic_lines = convert_to_range(melodic_lines, mode)
if mode == 'melodic lines':
    note_sequences = melodic_lines
else:
    note_sequences = extract_note_sequences(melodic_lines, all_voices, mode)
    
max_note = 28 if mode == 'melodic lines' else 29 if mode == 'no hold' else 30

notes = [note for seq in note_sequences for note in seq]
key_freq_dataset = np.zeros((max_note + 1))
for note in notes:
    key_freq_dataset[note] += 1
key_freq_dataset /= len(notes)

config = MelodyRnnConfig(
            generator_pb2.GeneratorDetails(
                id='basic_rnn',
                description='Melody RNN with one-hot encoding.'),
            note_seq.OneHotEventSequenceEncoderDecoder(
                note_seq.MelodyOneHotEncoding(
                    min_note=0, max_note=max_note)),
            contrib_training.HParams(
                batch_size=16,
                rnn_layer_sizes=[256, 128],
                dropout_keep_prob=0.5,
                clip_norm=5,
                learning_rate=0.002),
            min_note = 0,
            max_note = max_note)
            
generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
      model=MelodyRnnModel(config),
      details=config.details,
      steps_per_quarter=1,
      checkpoint='updates/no_hold_minor_sixteenth/logdir/run1/train/model.ckpt-1571',
      bundle=None)

generator_options = generator_pb2.GeneratorOptions()
generate_section = generator_options.generate_sections.add(
        start_time=0.5,
        end_time=16)
generator_options.args['temperature'].float_value = 1.0
generator_options.args['beam_size'].int_value = 1
generator_options.args['branch_factor'].int_value = 1
generator_options.args['steps_per_iteration'].int_value = 1
 
primer_melody = note_seq.Melody([17])
input_sequence = primer_melody.to_sequence(qpm = 30)
sequences = []
for i in range(200):
    generated_sequence = generator.generate(input_sequence, generator_options)
    notes = convert_notesequence_to_array(generated_sequence)
    sequences.append(notes)
    #print(generated_sequence.notes)
    #print(notes)
    

notes_gen = [note for seq in sequences for note in seq]
key_freq_gen = np.zeros((max_note + 1))
for note in notes_gen:
    key_freq_gen[note] += 1
key_freq_gen /= len(notes_gen)

print(key_freq_dataset)
print(key_freq_gen)
corr = jensenshannon(key_freq_dataset, key_freq_gen)
print(corr)
    

