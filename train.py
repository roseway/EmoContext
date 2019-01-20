from utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten, TimeDistributed
import io
from keras.callbacks import ModelCheckpoint

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def lstm(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    model_lstm = Sequential()
    model_lstm.add(embeddingLayer)
    model_lstm.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model_lstm.add(Dense(NUM_CLASSES, activation='softmax'))
    model_lstm.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['acc'])
    model_lstm.summary()
    return model_lstm


def cnn(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    model_conv = Sequential()
    model_conv.add(embeddingLayer)
    model_conv.add(Dropout(DROPOUT))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(Dropout(DROPOUT))
    model_conv.add(MaxPooling1D(pool_size=4))
    #model_conv.add(Flatten())
    model_conv.add(LSTM(64))
    model_conv.add(Dense(NUM_CLASSES, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_conv.summary()
    return model_conv


def main():
    global trainDataPath, devDataPath, testDataPath, solutionPath
    global NUM_CLASSES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, wordIndex

    trainDataPath = "data/train.txt"
    devDataPath = "data/dev.txt"
    testDataPath = "data/test.txt"
    solutionPath = "test.txt"

    NUM_CLASSES = 4
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 300
    BATCH_SIZE = 200
    LSTM_DIM = 128
    DROPOUT = 0.2
    NUM_EPOCHS = 30

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")

    print("Processing dev data...")
    devIndices, devTexts, devlabels = preprocessData(devDataPath, mode="train")

    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")

    print("Extracting tokens...")
    tokenizer = Tokenizer(oov_token="unk")
    tokenizer.fit_on_texts(trainTexts)
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    devSequences = tokenizer.texts_to_sequences(devTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)
    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    devData = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    devlabels = to_categorical(np.asarray(devlabels))
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))
    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    print(embeddingMatrix.shape)

    model = cnn(embeddingMatrix)
    checkpoint = ModelCheckpoint('cnn+lstm.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(devData, devlabels),
              callbacks=callbacks_list)

    model = load_model("cnn+lstm.h5")
    devpredictions = model.predict(devData, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(devpredictions, devlabels)

    pred = model.predict(testData, batch_size=BATCH_SIZE)
    pred = pred.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[pred[lineNum]] + '\n')
    print("Completed.")


if __name__ == '__main__':
    main()
