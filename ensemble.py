import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from utils import getMetrics
from keras.models import load_model
import io
from keras.callbacks import ModelCheckpoint

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def preprocessData(dataFilePath):
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            label = emotion2label[line[4]]
            labels.append(label)
    return labels


def main():
    trainDataPath = "data/train.txt"
    devDataPath = "data/dev.txt"
    testDataPath = "data/test.txt"
    solutionPath = "solution.txt"

    BATCH_SIZE = 200
    NUM_EPOCHS = 50

    print("Processing training data...")
    data = np.loadtxt('train.txt')
    labels = preprocessData(trainDataPath)
    labels = to_categorical(np.asarray(labels))

    print("Processing dev data...")
    devdata = np.loadtxt('dev.txt')
    devlabels = preprocessData(devDataPath)
    devlabels = to_categorical(np.asarray(devlabels))

    print("Processing test data...")
    testdata = np.loadtxt('test.txt')

    model = Sequential()
    model.add(Dense(8, input_dim=15, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()

    print("Training...")
    checkpoint = ModelCheckpoint('ensemble.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(devdata, devlabels),
              callbacks=callbacks_list)

    model = load_model('ensemble.h5')

    devpredictions = model.predict(devdata, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(devpredictions, devlabels)

    pred = model.predict(testdata, batch_size=BATCH_SIZE)
    pred = pred.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[pred[lineNum]] + '\n')
    print("Completed")


if __name__ == '__main__':
    main()
