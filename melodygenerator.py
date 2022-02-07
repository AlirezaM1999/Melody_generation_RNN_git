import numpy as np
import tensorflow.keras as keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import music21 as m21


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:   #load the mapping json file
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # seed is a sequence of encoded notes and asking the model to continue the rest
        # num_steps the number of steps we want the network to output
        # max_sequence is the max number of steps we want to consider in the seed for the network- this is gonna equal 64
        # temperature represent the predictibility of the output

        # Create a seed with start symbol
        seed = seed.split()   # [64, _, _, _]
        melody = seed
        seed = self._start_symbols + seed


        #map seed to int
        seed = [self._mappings[symbol] for symbol in seed ]  #give me the corresponding int for every symbol using mapping.json

        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # convert the ints to one hot encoded format - 2 dimentional array with the size of (max_sequence_length, number of symbols)
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings)) # 38

            #keras expects a 3 dimentional input so we have to expand the dims such that (1, max_sequence_length, number of symbols)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0] #only predict one sample
            # i.e [0.1, 0.2,...., 0.6] for each symbol
            # we dont always wanna choose the highest probability as it will make our model too rigid, that why we are using temperature sampling
            output_int  = self._sample_with_temperature(probabilities, temperature)


            # update seed - append the output_int the seed so we that we can feed it to the network
            seed.append(output_int)

            # map int back to our encoding - returns a list of symbols
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0] #only want the keys

            # check whether we are at the end of a melody - if we see "/", we are at the end of a song
            if output_symbol == "/":
                break


            # otherwise we update the melody
            melody.append(output_symbol)

        return  melody



    def _sample_with_temperature(self, probabilities, temperature):
        # temperature ->  infinity - complete random value
        # temperature -> 0 - the highest probabilty gets picked
        # temperature = 1 - we just use a normal sampling, nothing changes

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)

        return index



    #Converting the final list melody to MIDI
    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.midi"):

        # Create a music21 stream
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        # [60, _, _, _, r, _, 62], iterate through all the items and whenver we hit an event (midi or rest), we keep track of that information
        start_symbol = None  #needs to be an event (note or rest)
        step_counter = 1 # keeps track of all events, i.e stepcounter of 4 => a quarter note

        for i, symbol in enumerate(melody): # or if we are at the end of a melody
            # handle note/rest case

            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter  #0.25 * 4 = 1 beat

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter - reset everytime we have a new event
                    step_counter = 1

                start_symbol = symbol

            # handle prolongation sign "_" case
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == '__main__':
    mg = MelodyGenerator()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    seed3 = "67 _ _ _"
    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGTH, 0.2) #melody in time series representation
    print(melody)
    mg.save_melody(melody)