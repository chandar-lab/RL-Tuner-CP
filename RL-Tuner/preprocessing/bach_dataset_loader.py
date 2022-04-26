# Converts midi files into note sequences

# # Midi note numbers
# # C1: 24
# # C2: 36
# # C3: 48
# # C4: 60
# # C5: 72
# # C6: 84
# # C7: 96
# # C8: 108

# # RL-Tuner note numbers: 2, 14, 26

# # Soprano: between 60 and 81 = 21   60 -> 2, 72 -> 14
# # Alto: between 55 and 74 = 19      60 -> 14, 72 -> 26
# # Tenor: between 48 and 69 = 21     48 -> 2, 60 -> 14
# # Bass: between 36 and 64 = 28      36 -> 2, 48 -> 14, 60 -> 26
        
import mido
import string
import numpy as np
import copy
from itertools import combinations
from os import listdir
import pickle

def switch_note(last_state, note, velocity, on_=True):
    # https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """Converts note to a piano note value (between 0 and 88)
      Args:
        last_state: State indicating notes already playing
        note: Pitch of current note
        velocity: Loudness of current note
        on_: If the note is playing
      Returns:
        An array indicating which notes are currently playing
      """

    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

def msg2dict(msg):
    # https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """Converts midi message to a dictionary containing all useful informations
      Args:
        msg: Midi message
      Returns:
        Dictionary containing all useful information from message
    """
    result = dict()
    # Checking if the note is starting or ending
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None

    # Extracting time of message
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        # Extracting pitch value and loudness
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def get_new_state(new_msg, last_state):
    # https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """Updates state with new message
          Args:
            new_msg: New message
            last_state: Current state
          Returns:
            Updated state and current time
        """
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]
    
def track2seq(track):
    # https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """Converts track to array
          Args:
            track: Midi track
          Returns:
            Array indicating all notes in the track
        """
    result = []
    # Initialization of track
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        # Updating with every message in the track
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

def midi2array(mid, min_msg_pct=0.1):
    # https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    """Converts midi file to a set of arrays (one array for each track)
          Args:
            mid: Midi file content
            min_msg_pct: Minimum number of messages in a track
          Returns:
            Set of piano-roll arrays representing the midi file
        """
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]
    
    
def compute_distance_note_voice(voice, note):
    """Compute the distance between the middle of the range of the voice and the note
          Args:
            voice: Voice index (0 for bass, 1 for tenor, 2 for alto, 3 for soprano)
            note: Pitch value (between and 88)
          Returns:
            Distance between the note and the voice
        """
    # Soprano: 58 to 68
    # Alto: 50 to 65
    # Tenor: 34 to 58
    # Bass: 34 to 48

    # Indicates the minimum value for each voice, maximum value without overlap (-1 if all values overlap with another voice), middle value
    # Example: since bass is the lowest voice and tenor usually starts at 34, every note below 34 belows to the bass voice
    # Example no 2: tenor fully overlaps with bass and alto voices so we cannot say with certainty that a note belongs to the tenor voice
    important_notes = [[0, 33, (48 + 34)/2], [0, -1, (58 + 34)/2], [0, -1, (65 + 50)/2], [66, 100, (68 + 58)/2]] # bass, tenor, alto, soprano
    
    v = important_notes[voice]
    # If the note is in the interval exclusive to the voice, we return a very low distance (this note belongs to that voice)
    if note >= v[0] and note <= v[1]:
        return -1000

    # Otherwise we return the distance to the middle value of the voice
    return abs(note - v[2])
    
def replace_held_notes(voice):
    """Replaces held notes with a 1 indicating that the note is held
              Args:
                voice: List of notes in voice
              Returns:
                Voice with repeated notes replaced with held token
            """
    held_notes = list()
    for i in range(1, len(voice)):
        if voice[i] == voice[i - 1]:
            held_notes.append(i)
            
    held_notes = np.asarray(held_notes)
    voice = np.array(voice)
    voice[held_notes] = 1
    
    return voice
    
def change_granularity(voice, factor):
    """Changes precision of the time information
              Args:
                voice: Liste of notes in voice
                factor: Indicates how big the change is
              Returns:
                New list of notes with specified granularity
            """
    new_voice = []
    for i in range(len(voice)):
        if i % factor == 0:
            note = voice[i]
            
            # If we have a held note, we need to find if the note is actually held in the new granularity
            if note == 1:
                for j in range(i, max(0, i - factor), -1):
                    # We select the last note in the time interval
                    if voice[j] != 1:
                        note = voice[j]
                        break

            new_voice.append(note)
            
    return np.asarray(new_voice)


def load_all_midi_in_folder(path, resolution):
    """Reads all midi files in folder and extracts a list of each processed soprano, alto, tenor, bass voices
          Args:
            path: Path of the folder
            resolution: Granularity to apply
          Returns:
            Distance between the note and the voice
            """
    files = listdir(path)
    soprano_voices = list()
    alto_voices = list()
    bass_voices = list()
    tenor_voices = list()
    for file in files[:1]:
        midi = mido.MidiFile(filename=path + file)

        # Obtains piano roll
        result_array = midi2array(midi)

        soprano_voice = list()
        alto_voice = list()
        bass_voice = list()
        tenor_voice = list()

        for k in range(len(result_array)):
            t = copy.deepcopy(result_array[k])
            notes_on = np.where(t > 0)

            # If all voices are present
            if len(notes_on[0]) == 4:
                bass_voice.append(notes_on[0][0] + 2)
                tenor_voice.append(notes_on[0][1] + 2)
                alto_voice.append(notes_on[0][2] + 2)
                soprano_voice.append(notes_on[0][3] + 2)
                
            elif len(notes_on[0]) == 0:
                # Four voices missing:
                bass_voice.append(0)
                tenor_voice.append(0)
                alto_voice.append(0)
                soprano_voice.append(0)
               
            else:
                # We need to find which voice(s) is/are missing
                distances_to_voices = np.zeros((4, len(notes_on[0])))
                for i in range(4):
                    for j in range(len(notes_on[0])):
                        distances_to_voices[i, j] = compute_distance_note_voice(i, notes_on[0][j])

                # Compute all the possible combinations for the missing voices (0 for bass, 1 for tenor, 2 for alto, 3 for soprano)
                missing_voices = list(combinations([0, 1, 2, 3], 4 - len(notes_on[0])))
                total_distance = np.zeros(len(missing_voices))

                # For each combination, we compute the distance between the voices that are not missing
                for i in range(len(missing_voices)):
                    print("i: ", i)
                    current_note = 0
                    for j in range(4):
                        # If the voice is not missing
                        if j not in missing_voices[i]:
                            total_distance[i] += distances_to_voices[j, current_note]
                            current_note += 1

                # We select the tuple of missing voices with the lowest distance
                missing_voice = missing_voices[np.argmin(total_distance)]

                # We add 0 if the bass voice is missing, the lowest value if the note is not missing
                current_note = 0
                if 0 in missing_voice:
                    bass_voice.append(0)
                else:
                    bass_voice.append(notes_on[0][current_note] + 2)
                    current_note += 1
                    
                if 1 in missing_voice:
                    tenor_voice.append(0)
                else:
                    tenor_voice.append(notes_on[0][current_note] + 2)
                    current_note += 1
                    
                if 2 in missing_voice:
                    alto_voice.append(0)
                else:
                    alto_voice.append(notes_on[0][current_note] + 2)
                    current_note += 1
                   
                if 3 in missing_voice:
                    soprano_voice.append(0)
                else:
                    soprano_voice.append(notes_on[0][current_note] + 2)
                    current_note += 1

        factor = int(resolution)
        bass_voice = replace_held_notes(bass_voice)
        tenor_voice = replace_held_notes(tenor_voice)
        alto_voice = replace_held_notes(alto_voice)
        soprano_voice = replace_held_notes(soprano_voice)
        bass_voice = change_granularity(bass_voice, factor)
        tenor_voice = change_granularity(tenor_voice, factor)
        alto_voice = change_granularity(alto_voice, factor)
        soprano_voice = change_granularity(soprano_voice, factor)
        
        bass_voices.append(bass_voice)
        tenor_voices.append(tenor_voice)
        alto_voices.append(alto_voice)
        soprano_voices.append(soprano_voice)
        
    return [bass_voices, tenor_voices, alto_voices, soprano_voices]

def main():
    """Reads midi files and writes the processed arrays in a pickle file"""
    quarter_note = 120
    eight_note = quarter_note/2
    sixteenth_note = quarter_note/4
    resolution = quarter_note
    path_root = 'JSB Chorales/JSB Chorales/'
    dataset = load_all_midi_in_folder(path_root + 'train/', resolution)
    with open('train_4.pkl', 'wb') as file:
        pickle.dump(dataset, file)

main()

