from utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def main():
    global trainDataPath, devDataPath, testDataPath
    global MAX_SEQUENCE_LENGTH
    trainDataPath = "data/train.txt"
    devDataPath = "data/dev.txt"
    testDataPath = "data/test.txt"

    MAX_SEQUENCE_LENGTH = 100

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
    traindata = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    devData = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)

    model1 = load_model('happy.h5')
    model2 = load_model('sad.h5')
    model3 = load_model('angry.h5')
    model4 = load_model('lstm.h5')
    model5 = load_model('cnn.h5')
    model6 = load_model('cnn+lstm.h5')

    pred1 = model1.predict(traindata)
    pred2 = model2.predict(traindata)
    pred3 = model3.predict(traindata)
    pred4 = model4.predict(traindata)
    pred5 = model5.predict(traindata)
    pred6 = model6.predict(traindata)
    temp = np.hstack(
        (pred1[:, 1].reshape(-1, 1), pred2[:, 1].reshape(-1, 1), pred3[:, 1].reshape(-1, 1), pred4, pred5, pred6))

    np.savetxt('train.txt', temp)

    pred1 = model1.predict(devData)
    pred2 = model2.predict(devData)
    pred3 = model3.predict(devData)
    pred4 = model4.predict(devData)
    pred5 = model5.predict(devData)
    pred6 = model6.predict(devData)
    temp = np.hstack(
        (pred1[:, 1].reshape(-1, 1), pred2[:, 1].reshape(-1, 1), pred3[:, 1].reshape(-1, 1), pred4, pred5, pred6))

    np.savetxt('dev.txt', temp)

    pred1 = model1.predict(testData)
    pred2 = model2.predict(testData)
    pred3 = model3.predict(testData)
    pred4 = model4.predict(testData)
    pred5 = model5.predict(testData)
    pred6 = model6.predict(testData)
    temp = np.hstack(
        (pred1[:, 1].reshape(-1, 1), pred2[:, 1].reshape(-1, 1), pred3[:, 1].reshape(-1, 1), pred4, pred5, pred6))

    np.savetxt('test.txt', temp)


if __name__ == '__main__':
    main()
