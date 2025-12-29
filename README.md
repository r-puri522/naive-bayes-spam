
# Naive Bayes Spam Detection (From Scratch)

**View / Run in Google Colab:**  
[https://colab.research.google.com/drive/1POH-EnuZkRKqLOi1CksWwehKV2sAM6tG#scrollTo=x3OFl47z_5bf]

A spam classification project that implements a **Naive Bayes classifier from scratch** to distinguish between `spam` and `ham` SMS messages.  
The model is trained on labeled text data, evaluated on a held-out test set, and analyzed using a **confusion matrix** and standard classification metrics.

A **scikit-learn `MultinomialNB`** model is also included as a baseline comparison.

---

## Project motivation

Spam detection is a classic natural language processing problem with real-world impact.  
This project focuses on **understanding and implementing the Naive Bayes algorithm manually**, rather than relying solely on library abstractions.

Key learning goals:
- Build a text classification pipeline end-to-end
- Apply probabilistic reasoning with Naive Bayes
- Evaluate classification performance beyond simple accuracy

---

## What this project demonstrates

- Text preprocessing and bag-of-words feature construction
- Word frequency counting using **pandas**
- Class prior and conditional probability estimation
- **Laplace smoothing** to handle unseen words
- Log-probability scoring to avoid numerical underflow
- Model evaluation using:
  - confusion matrix
  - accuracy
  - precision
  - recall
  - F1 score
- Error analysis via misclassified examples

---

## Files in this repository

- `NaiveBayesSpam.ipynb`  
  Main notebook containing data loading, model implementation, evaluation, and analysis.

- `SMS.txt`  
  Tab-separated dataset of SMS messages with labels (`ham` or `spam`).

- `README.md`  
  Project overview and instructions.

---

## Dataset format

`SMS.txt` contains one message per line with the following format:

<label>\t<message text>

Example:

ham\tHey, are we still meeting later?  
spam\tCongratulations! You've won a free prize.

---

## Model overview

For a given message, the classifier compares:

- **P(spam | message)** vs **P(ham | message)**

Using the Naive Bayes assumption of word independence:

P(class | message) ∝ P(class) × ∏ P(word | class)

To improve numerical stability, all probabilities are computed in **log space** during prediction.

---

## Results

After training, the notebook outputs:

- Confusion matrix (TP, TN, FP, FN)
- Accuracy
- Precision
- Recall
- F1 score

Example:
- Accuracy: `0.9883303411131059`
- Precision: `0.9652777777777778`
- Recall: `0.9455782312925171`
- F1: `0.9553264604810997`

---

## Baseline comparison

To validate the custom implementation, the project also includes a comparison against:

- `sklearn.naive_bayes.MultinomialNB`

This helps confirm correctness and provides a performance reference using a standard library implementation.

---


