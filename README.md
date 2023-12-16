```markdown
# Stock Market Sentiment Analysis

## Overview

This project conducts sentiment analysis on stock market news headlines to predict market trends. The dataset is loaded into a pandas DataFrame using the provided 'Data.csv' file. A training-testing split is performed, and headlines undergo preprocessing, including punctuation removal and conversion to lowercase.

## Implementation Steps

1. **Data Loading:**

   ```python
   import pandas as pd
   df = pd.read_csv('Data.csv', encoding='ISO-8859-1')
   df.head()
   ```

   Make sure to download the 'Data.csv' file from the [Kaggle Stock Sentiment Analysis Dataset](https://www.kaggle.com/code/siddharthtyagi/stock-sentiment-analysis-using-news-headlines/input).

2. **Data Preprocessing:**

   ```python
   train = df[df['Date'] < '20150101']
   data = train.iloc[:, 2:27]
   data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
   data.columns = [str(i) for i in range(25)]
   for index in data.columns:
       data[index] = data[index].str.lower()
   ```

3. **Bag of Words Model:**

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   countvector = CountVectorizer(ngram_range=(2, 2))
   traindataset = countvector.fit_transform(headlines)
   ```

4. **Random Forest Classifier:**

   ```python
   from sklearn.ensemble import RandomForestClassifier
   randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
   randomclassifier.fit(traindataset, train['Label'])
   ```

5. **Model Evaluation:**

   ```python
   test_transform = [' '.join(str(x) for x in test.iloc[row, 2:27]) for row in range(len(test.index))]
   test_dataset = countvector.transform(test_transform)
   predictions = randomclassifier.predict(test_dataset)

   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

   matrix = confusion_matrix(test['Label'], predictions)
   print(matrix)
   score = accuracy_score(test['Label'], predictions)
   print(score)
   report = classification_report(test['Label'], predictions)
   print(report)
   ```

## Results

The Random Forest Classifier achieves an accuracy of 86.77% on the test dataset. The 'Data.csv' file is available on Kaggle. You can download it using the following link:

[Kaggle Stock Sentiment Analysis Dataset](https://www.kaggle.com/code/siddharthtyagi/stock-sentiment-analysis-using-news-headlines/input)

Please make sure to comply with Kaggle's terms and conditions when using the dataset. If you find this dataset useful, consider giving credit to the original contributor on Kaggle.
```
