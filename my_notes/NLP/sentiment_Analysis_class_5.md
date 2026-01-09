# 📘 Class 5 — Sentiment Analysis (NLP)


---

## 1. Where This Class Fits in NLP

Up to this point, the course focused on **NLP fundamentals**:
- Tokenization
- Stop word removal
- Stemming vs lemmatization
- POS tagging
- Named Entity Recognition (NER)

**Class 5 transitions from theory to application**, using **Sentiment Analysis** as the primary real-world use case.

---

## 2. Why Sentiment Analysis Matters

Most customer feedback is **text**, not numeric ratings:

- Product reviews
- Social media posts
- App store feedback
- Customer surveys
- Support tickets

Sentiment analysis enables organizations to:
- Detect negative trends early
- Respond publicly to complaints
- Improve products proactively
- Protect brand reputation

This is why sentiment analysis is used across nearly every industry.

---

## 3. NLP Preprocessing Recap

Machines do not understand raw text — they only understand **numbers**.

Therefore, text must be converted into numerical representations.

### 3.1 Tokenization
Splitting sentences into words/tokens.

"I love this phone" → ["I", "love", "this", "phone"]


---

### 3.2 Lowercasing
Avoids treating `Good` and `good` as different words.

---

### 3.3 Removing Punctuation and Numbers
Traditional ML pipelines treat punctuation and digits as noise.

> ⚠️ Removing punctuation may lose nuance (e.g., “!!!”), but is common in TF-IDF pipelines.

---

### 3.4 Stop Word Removal
Words such as:
- the, is, to, and, of

These appear frequently but add little semantic value.

---

### 3.5 Lemmatization
Converts words to their root/base form.

| Word | Lemma |
|----|----|
| eating | eat |
| ate | eat |
| better | good |

Lemmatization is preferred over stemming because it preserves meaning.

---

### 3.6 POS Tagging (Conceptual)
POS tagging helps lemmatization choose the correct root:
- noun
- verb
- adjective

Covered in earlier classes and conceptually important here.

---

## 4. What Is Sentiment Analysis?

Sentiment analysis determines the **emotional tone** of text.

---

### 4.1 Basic Sentiment Classification
Most common form:
- Positive
- Negative
- Neutral

Example:

"I love this product" → Positive


---

### 4.2 Rating-Based Sentiment
Numeric representations:
- ⭐⭐⭐⭐⭐ → very positive
- ⭐ → very negative

Used heavily by Amazon, Google Reviews, and app stores.

---

## 5. Aspect-Based Sentiment Analysis

Example:

"The screen is awesome but the sound quality is bad."


Instead of one sentiment:
- Screen → Positive
- Sound → Negative

### Why this matters:
- Different teams own different features
- Enables targeted improvements
- Much more actionable than a single sentiment label

---

## 6. Traditional Approaches to Sentiment Analysis

### 6.1 Rule-Based (Oldest)
- Keyword matching (good, bad, terrible)
- Lexicon scoring (e.g., VADER)

Limitations:
- Does not scale well
- Poor semantic understanding

---

### 6.2 Applied Machine Learning (This Class)

Pipeline:
1. Text preprocessing
2. Vectorization (TF-IDF)
3. Classification model

Common models:
- Logistic Regression
- Naive Bayes
- Support Vector Machine
- Random Forest

---

### 6.3 Transformers / LLMs (Upcoming Classes)
- Minimal preprocessing
- Better semantic understanding
- Handles context and sarcasm

---

## 7. TF-IDF Explained

TF-IDF = **Term Frequency × Inverse Document Frequency**

### Term Frequency (TF)
Measures how often a word appears in a document.

### Inverse Document Frequency (IDF)
Penalizes words that appear across many documents.

Result:
- Rare but meaningful words get higher weight
- Common words get lower weight

Output is a **sparse matrix**:
- Rows → documents
- Columns → vocabulary
- Values → importance weights

---

## 8. End-to-End ML Workflow Built in Class

### 8.1 Dataset
- ~27,000 tweets
- Labels:
  - positive
  - neutral
  - negative

---

### 8.2 Cleaning and Preprocessing
- Lowercase text
- Remove punctuation and digits
- Remove stop words
- Lemmatize
- Rejoin tokens into sentences

---

### 8.3 Train / Validation Split
Typical split:
- 80% training
- 20% validation

---

### 8.4 Vectorization
- TF-IDF `fit_transform()` on training data
- TF-IDF `transform()` on validation/test data

⚠️ Never fit on validation or test data.

---

### 8.5 Models Trained
- Random Forest (best performance in demo ~70%)
- Multinomial Naive Bayes
- Linear SVM

---

### 8.6 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

> Accuracy alone is not sufficient; always check F1-score per class.

---

### 8.7 Saving the Model
- Best model saved using `pickle`
- Reloaded later for predictions
- Production-ready workflow

---

## 9. Aspect-Based Sentiment Using Clustering

When aspects are unknown:

1. Convert sentences to vectors
2. Apply clustering (unsupervised)
3. Clusters emerge based on:
   - Topic (screen, sound)
   - Sentiment (positive, negative)

Techniques discussed:
- K-Means
- PCA
- NMF
- LDA (topic modeling)

---

## 10. Limitations of TF-IDF + Classic ML

- Loses word order
- Limited semantic understanding
- Sensitive to preprocessing choices
- Poor handling of sarcasm

➡️ These limitations motivate the use of transformers.

---

## 11. What to Know Before the Next Class

✔️ Explain TF-IDF in plain English  
✔️ Understand the classic ML sentiment pipeline  
✔️ Know sentiment types (binary, multi-class, aspect-based)  
✔️ Understand why clustering helps with aspects  
✔️ Be ready to compare ML vs transformers  

---

## 12. Learning Resources

### Blogs
- https://jalammar.github.io/illustrated-bert/
- https://machinelearningmastery.com/tfidf-for-machine-learning/
- https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

### YouTube
- TF-IDF intuition: https://www.youtube.com/watch?v=4vGJ6BIb4rM
- Sentiment analysis intro: https://www.youtube.com/watch?v=8oF1-5lZsA8
- BERT for sentiment: https://www.youtube.com/watch?v=XFh_S5o3slM

---

## 13. What’s Coming Next

Upcoming sessions will cover:
- BERT
- RoBERTa
- DistilBERT
- LLM-based sentiment analysis
- Aspect-based sentiment using transformers
