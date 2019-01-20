from utils import getMetrics, getEmbeddingMatrix
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import re
import io
import emoji
import contractions

label2emotion = {0: "others", 1: "happy"}
emotion2label = {"others": 0, "happy": 1, "sad": 0, "angry": 0}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            line = line.replace('😂', '😁')
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)
            # Add whitespace around emojis
            emojis = [c for c in line if c in emoji.UNICODE_EMOJI]
            for e in emojis:
                lineSplit = line.split(e)
                cSpace = ' ' + e + ' '
                line = cSpace.join(lineSplit)
            # Expand contractions
            line = contractions.fix(line)

            line = line.strip().split('\t')

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def buildModel(embeddingMatrix):
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
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    return model


def main():
    global trainDataPath, devDataPath, testDataPath, solutionPath
    global NUM_CLASSES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, wordIndex

    trainDataPath = "data/happy.txt"
    devDataPath = "data/dev.txt"
    testDataPath = "data/test.txt"
    solutionPath = "test.txt"

    NUM_CLASSES = 2
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

    model = buildModel(embeddingMatrix)
    checkpoint = ModelCheckpoint('happy.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(devData, devlabels),
              callbacks=callbacks_list)

    model = load_model("happy.h5")
    devpredictions = model.predict(devData, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(devpredictions, devlabels)
    
    predictions = devpredictions.argmax(axis=1)

    wrong=[]
    with io.open('happy.txt', "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label", "prediction"]) + '\n')        
        with io.open(devDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line=line.strip().split('\t')
                if emotion2label[line[-1]]!=predictions[lineNum]:
                    fout.write('\t'.join(line) + '\t')
                    fout.write(label2emotion[predictions[lineNum]] + '\n')


if __name__ == '__main__':
    main()
