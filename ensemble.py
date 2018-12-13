#Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.core import*
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import io
import sys
import emoji
import contractions
from keras_self_attention import SeqSelfAttention
from keras.callbacks import ModelCheckpoint

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def preprocessData(dataFilePath):
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line=line.strip().split('\t')
            label = emotion2label[line[4]]
            labels.append(label)
    return labels
        
def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1
   
    
def main():
        
    global trainDataPath,devDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    
    trainDataPath = "data/train0.txt"
    devDataPath = "data/dev.txt"
    testDataPath="data/test.txt"
    solutionPath = "test.txt"
    gloveDir = "data"
    
    NUM_CLASSES = 4
    MAX_NB_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 300
    BATCH_SIZE = 200
    LSTM_DIM = 128
    DROPOUT = 0.2
    LEARNING_RATE = 0.002
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
    testdata = np.loadtxt('test0.txt')

    model=Sequential()
    model.add(Dense(8, input_dim=19, activation='tanh'))
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
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(devdata, devlabels), callbacks=callbacks_list)
    
    model=load_model('ensemble.h5')
    devpredictions = model.predict(devdata, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(devpredictions, devlabels)
    devpredictions=devpredictions.argmax(axis=1)
    wrong=[]
    with io.open('ensemble.txt', "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label", "prediction"]) + '\n')        
        with io.open("data/dev.txt", encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line=line.strip().split('\t')
                if emotion2label[line[-1]]!=devpredictions[lineNum]:
                    fout.write('\t'.join(line) + '\t')
                    fout.write(label2emotion[devpredictions[lineNum]] + '\n')
   
    pred = model.predict(testdata, batch_size=BATCH_SIZE)
    pred = pred.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[pred[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
    

               
if __name__ == '__main__':
    main()
