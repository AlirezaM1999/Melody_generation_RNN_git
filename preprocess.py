import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np

#pass different formats kern, MIDI and convert them to 21 and convert them back
#we can use m21 to represent music in a object oriented manner

KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [
    0.25,  #sixthteen note (current time step rate)
    0.5,
    0.75,
    1.0,   #quarter note
    1.5,
    2,
    3,
    4,
]
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"






def load_songs_in_kern(dataset_path):
    songs = []

    # go through all the files in the dataset and load them with music21 (m21 objects)
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            # load only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs



#if a song does not have acceptable duration(accetable notes) not time signature, we do not include them
def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:      #gets rid of everything other than notes and rests
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)    #get all the part objects
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) #get the first part with all the info
    key = measures_part0[0][4]   #4th element in the first measure of the first part

    # if the key is not noted in the song, estimate key using music 21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # calculate interval for transposation
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))


    # transpose by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


#encodes the song as a time series representation
def encode_song(song, time_step=0.25): #our time step is 16th note by default
    # p = 60, d = 1.0 -> [60,_,_,_] is C4 quarter note
    encoded_song = []

    for event in song.flat.notesAndRests:  #we either have notes or rests
        #handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi    #a midi note like 60 (C4)
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest to time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast the encoded song to string
    encoded_song = " ".join(map(str, encoded_song)) # make all the items strings and then join them

    return encoded_song





def preprocess(dataset_path):
    # load the folk songs
    print("loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"loaded {len(songs)} songs")

    for i, song in enumerate(songs):        #go through each song individually and apply the stuff

        # filter out songs that have non-accetable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue #ignore the song

        # transpose songs to Cmaj or Amin
        song = transpose(song)

        # Encode songs with music time series representation i.e [60,_,_,_] <=== one quarter note
        encoded_song = encode_song(song)

        #save the songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


#puts all the generated song files in one sequence in one file
def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter =  "/ " * sequence_length           # number of slashes that represent the length of a song (64)
    songs = ""   #contains all the songs' symbols togather

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)  # a string that contains a whole sequence in a single song. i.e all symbols in "1" in test dataset
            songs += song  + " " + new_song_delimiter
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs



#our model is going to only train on numbers
# we are going to map all the symbols to numbers to be able to train our model with data
def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary (all the symbyols we have in the dataset)
    songs = songs.split()
    vocabulary = list(set(songs))

    # create the mapppings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save the vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load the mapping json file as a dict
    with open(MAPPING_PATH, "r")as fp:
        mappings = json.load(fp)

    #cast songs string to a list
    songs = songs.split()

    # map songs to int  - everytime we see a midi number, we add its corresponding key to the list
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs



def generate_training_sequences(sequence_length):
    # given a note sequence [11, 12, 13, 14, ...] -> input: [11,12]   , target: 13
    # we shift continuesly until we reach the end of the sequence so the next input = [12, 13]


    # load the songs and map them to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    # lets say we have 100 and our sequence length is 64, we can generate 100-64=36 sequences
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length]) #takes a slice of int songs and shifts right by one step
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences - easiest way to deal with categorical data
    # inputs: (#number of sequences, sequence length, vocabulary size)
    # one hot encoding -- [0,1,2] = [[1,0,0],[0,1,0],[0,0,1]], each position represents a class
    # in our case, we have 18 different values , vocab size  = 18
    vocabulary_size = len(set(int_songs)) # 18
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()