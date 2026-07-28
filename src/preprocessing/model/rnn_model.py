#import 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,SimpleRNN,Embedding

#configuration
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
MAX_LENGTH = 40

RNN_UNITS = 64
DROPOUT_RATE = 0.5

NUM_CLASSES = 3


#building the RNN model
def build_model():
    model=Sequential()
    #embedding layer
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=EMBEDDING_DIM,
                        input_length=MAX_LENGTH))

    #simple rnn layer
    model.add(
        SimpleRNN(
            units=RNN_UNITS)
    )
    #dropout layer
    model.add(
        Dropout(
            rate=DROPOUT_RATE))

    #hidden dense layer 
    model.add(
        Dense(
            64,
            activation='relu'
        )
    )
    #output layer
    model.add(
        Dense(
            NUM_CLASSES,
            activation='softmax'
        )
    )
    return model

#test model
if __name__=="__main__":
    model=build_model()
    model.summary()


