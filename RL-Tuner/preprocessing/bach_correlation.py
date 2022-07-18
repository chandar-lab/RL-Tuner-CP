import pickle
import numpy as np
from bach_note_sequences import extract_melodic_lines, c_major, convert_to_range, extract_note_sequences
from melody_rnn_model import *
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from note_seq.protobuf import music_pb2
from scipy.spatial.distance import jensenshannon

def convert_notesequence_to_array(note_sequence):
    """Returns an array containing the note playing at each timestep.
      Args:
        note_sequence: A NoteSequence containing a list of notes with the start time and end time of each note
      Returns:
        An array of notes
      """
    notes = []
    time = 0
    for note in note_sequence.notes:
        while note.start_time > time:
            # Means that the next note has not started yet, so we add a 0 indicating that no note is playing
            notes.append(0)
            time += 0.5
            
        notes.append(note.pitch)
        time += 0.5
        
    return notes

def load_pickle_file(path):
    """Returns content of pickle file
      Args:
        path: path of the pickle file
      Returns:
        Data contained in the file
      """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def main():
    """Computes divergence between generated data and dataset"""
    granularity = 16
    mode = 'no hold'
    a_minor = True

    dataset = load_pickle_file('train_' + str(granularity) + '.pkl')

    all_voices = []
    all_voices.extend(dataset[3])

    dataset_valid = load_pickle_file('valid_' + str(granularity) + '.pkl')

    all_voices.extend(dataset_valid[3])

    dataset_test = load_pickle_file('test_' + str(granularity) + '.pkl')

    all_voices.extend(dataset_test[3])

    print("Note sequences.")
    melodic_lines = extract_melodic_lines(all_voices)
    print("Transposition..")
    print(len(melodic_lines))
    melodic_lines = c_major(melodic_lines, a_minor)
    print("Range conversion...")
    melodic_lines = convert_to_range(melodic_lines, mode)
    if mode == 'melodic lines':
        note_sequences = melodic_lines
    else:
        note_sequences = extract_note_sequences(melodic_lines, all_voices, mode)

    # Melodic lines: no rest and no hold token, pitch values go from 0 to 28
    # Note sequences: rest and hold tokens, pitch values go from 2 to 30 (0 for rest and 1 for hold)
    # No hold: rest tokens but no hold tokens, pitch values go from 1 to 29 (0 for rest)
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


    notes_gen = [note for seq in sequences for note in seq]
    key_freq_gen = np.zeros((max_note + 1))
    for note in notes_gen:
        key_freq_gen[note] += 1
    key_freq_gen /= len(notes_gen)

    print(key_freq_dataset)
    print(key_freq_gen)
    corr = jensenshannon(key_freq_dataset, key_freq_gen)
    print(corr)

main()
    

