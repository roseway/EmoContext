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
            emojis=[c for c in line if c in emoji.UNICODE_EMOJI]
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
    NUM_EPOCHS = 100
        

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")

    print("Processing dev data...")
    devIndices, devTexts, yVal = preprocessData(devDataPath, mode="train")
    
    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    
    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    devSequences = tokenizer.texts_to_sequences(devTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)
    
    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
    
    model1 = load_model('happy.h5')
    model2 = load_model('sad.h5')
    model3 = load_model('angry.h5')
    model4 = load_model('best.h5')
    model5 = load_model('cnn.h5')
    model6 = load_model('cnn+lstm.h5')
    model7 = load_model('noemoji.h5')

    pred1 = model1.predict(data, batch_size=BATCH_SIZE)
    pred2 = model2.predict(data, batch_size=BATCH_SIZE)
    pred3 = model3.predict(data, batch_size=BATCH_SIZE)
    pred4 = model4.predict(data, batch_size=BATCH_SIZE)
    pred5 = model5.predict(data, batch_size=BATCH_SIZE)
    pred6 = model6.predict(data, batch_size=BATCH_SIZE)
    pred7 = model7.predict(data, batch_size=BATCH_SIZE)
    data=np.hstack((pred1[:,1].reshape(-1,1),pred2[:,1].reshape(-1,1),pred3[:,1].reshape(-1,1),pred4,pred5,pred6,pred7))

    np.savetxt('train.txt',data)

    devData = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    yVal = to_categorical(np.asarray(yVal))
    pred1 = model1.predict(devData, batch_size=BATCH_SIZE)
    pred2 = model2.predict(devData, batch_size=BATCH_SIZE)
    pred3 = model3.predict(devData, batch_size=BATCH_SIZE)
    pred4 = model4.predict(devData, batch_size=BATCH_SIZE)
    pred5 = model5.predict(devData, batch_size=BATCH_SIZE)
    pred6 = model6.predict(devData, batch_size=BATCH_SIZE)
    pred7 = model7.predict(devData, batch_size=BATCH_SIZE)
    data=np.hstack((pred1[:,1].reshape(-1,1),pred2[:,1].reshape(-1,1),pred3[:,1].reshape(-1,1),pred4,pred5,pred6,pred7))

    np.savetxt('dev.txt',data)


    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    pred1 = model1.predict(testData, batch_size=BATCH_SIZE)
    pred2 = model2.predict(testData, batch_size=BATCH_SIZE)
    pred3 = model3.predict(testData, batch_size=BATCH_SIZE)
    pred4 = model4.predict(testData, batch_size=BATCH_SIZE)
    pred5 = model5.predict(testData, batch_size=BATCH_SIZE)
    pred6 = model6.predict(testData, batch_size=BATCH_SIZE)
    pred7 = model7.predict(testData, batch_size=BATCH_SIZE)
    data=np.hstack((pred1[:,1].reshape(-1,1),pred2[:,1].reshape(-1,1),pred3[:,1].reshape(-1,1),pred4,pred5,pred6,pred7))

    np.savetxt('test0.txt',data)

               
if __name__ == '__main__':
    main()
