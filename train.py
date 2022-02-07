import tensorflow.keras

from preprocess import generate_training_sequences, SEQUENCE_LENGTH
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import os



OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNTIS = [256]
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"





def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture - using the functional API
    input = Input(shape=(None, output_units))  #fist arguement represents the number of time steps, it can be any number, second arguement represnts how many values we have at each times step
    x = LSTM(num_units[0])(input)
    x = Dropout(0.2)(x)
    output = Dense(output_units, activation='softmax')(x)
    model = Model(input, output)

    # compile the model
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=['accuracy'])

    #print some basic information about the model
    model.summary()

    return model



def train(output_units=OUTPUT_UNITS, num_units=NUM_UNTIS, loss=LOSS, learning_rate=LEARNING_RATE):

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the nueral network model
    model = build_model(output_units, num_units, loss, learning_rate)

    #train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)





if __name__ == "__main__":
    if not os.path.isfile('model.h5'):
        train()
