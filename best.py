#Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.core import*
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout, Flatten,TimeDistributed
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


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}

    # Load the embedding vectors from the GloVe file
    with io.open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    # Load the embedding vectors from emoji2vec file
    with open('data/emoji2vec.txt') as f:
	    f.readline()
	    for line in f:
		    values = line.split()
		    word = values[0]
		    embeddingVector = np.asarray(values[1:], dtype='float32')
		    embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    #ttt=0
    #rrrsss=[]
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
        	# words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
            #ttt+=1
        #else:
            #rrrsss.append(word)
    #print (ttt)
    #with open("notgood.txt", "w", encoding='utf-8') as f:
        #f.write('\n'.join(rrrsss))
    return embeddingMatrix
            

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
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model
    
#    model_conv = Sequential()
#    model_conv.add(embeddingLayer)
#    model_conv.add(Dropout(0.2))
#    model_conv.add(Conv1D(128, 5, activation='relu'))
#    model_conv.add(Dropout(0.2))
#    model_conv.add(MaxPooling1D(pool_size=4))
#    #model_conv.add(Flatten())
#    model_conv.add(LSTM(128))
#    model_conv.add(Dense(4, activation='softmax'))
#    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    model_conv.summary()
#    return model_conv

def main():
        
    global trainDataPath,devDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    
    trainDataPath = "data/train0.txt"
    devDataPath = "data/dev.txt"
    testDataPath="data/test.txt"
    solutionPath = "test.txt"
    gloveDir = "data"
    
    NUM_FOLDS = 5
    NUM_CLASSES = 4
    MAX_NB_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 300
    BATCH_SIZE = 200
    LSTM_DIM = 128
    DROPOUT = 0.2
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 75
        
    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")

    print("Processing dev data...")
    devIndices, devTexts, devlabels = preprocessData(devDataPath, mode="train")
    
    # writeNormalisedData(testDataPath, testTexts)
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
    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    devData = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)

    devlabels = to_categorical(np.asarray(devlabels))
    
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
    
    
    
    model = buildModel(embeddingMatrix)
    checkpoint = ModelCheckpoint('best.h5', monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(testData, yVal), callbacks=callbacks_list)

    model=load_model("best.h5")
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
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
#    predictions = model.predict(testData, batch_size=BATCH_SIZE)
#    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
#    predictions = predictions.argmax(axis=1)
#    print("Creating solution file...")
#    wrong=[]
#    with io.open('best.txt', "w", encoding="utf8") as fout:
#        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label", "prediction"]) + '\n')        
#        with io.open(testDataPath, encoding="utf8") as fin:
#            fin.readline()
#            for lineNum, line in enumerate(fin):
#                line=line.strip().split('\t')
#                if emotion2label[line[-1]]!=predictions[lineNum]:
#                    fout.write('\t'.join(line) + '\t')
#                    fout.write(label2emotion[predictions[lineNum]] + '\n')

               
if __name__ == '__main__':
    main()