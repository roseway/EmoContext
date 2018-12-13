# README 
Download Instructions -
1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
2. Add glove.6B.300d.txt to `data/`.

### How to run
1. To run the baseline:
   ```bash
   python3 try.py -config testBaseline.config
   ```
2. To train the single model that predicts the probability of class angry:
   ```bash
   python3 angry.py
   ```
3. To train the single model that predicts the probability of class angry:
   ```bash
   python3 angry.py
   ```
4. To train the single model that predicts the probability of class happy:
   ```bash
   python3 happy.py
   ```
5. To train the single model that predicts the probability of class sad:
   ```bash
   python3 sad.py
   ```
6. To train the CNN or CNN+LSTM single model that predicts the probability distribution:
   ```bash
   python3 cnn.py
   ```
   Note: you may need to modify the file to train different models.
7. To train the CNN single model that predicts the probability distribution:
   ```bash
   python3 cnn.py
   ```
8. To generate outputs of the single models and concatenate them:
   ```bash
   python3 write.py
   ```
9. To train the ensemble model"
   ```bash
   python3 ensemble.py
   ```
