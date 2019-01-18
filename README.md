# README 
Download Instructions -
1. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
2. Add glove.6B.300d.txt to `data/`.
3. Add Emocontext train, dev, test files to `data/`.

### How to run
1. To get a subset of training data to train the binary classifiers:

   ```bash
   python3 resample.py
   ```

2. To train the single model that predicts the probability of class sad:

   ```bash
   python3 sad.py
   ```

3. To train the single model that predicts the probability of class angry:
   ```bash
   python3 angry.py
   ```

4. To train the single model that predicts the probability of class happy:
   ```bash
   python3 happy.py
   ```

5. To train the LSTM, CNN or CNN+LSTM single model that predicts the probability distribution:
   ```bash
   python3 train.py
   ```
   Note: you may need to modify the file to train different models.

6. To generate outputs of the single models and concatenate them:
   ```bash
   python3 write.py
   ```

7. To train the ensemble model:
   ```bash
   python3 ensemble.py
   ```
