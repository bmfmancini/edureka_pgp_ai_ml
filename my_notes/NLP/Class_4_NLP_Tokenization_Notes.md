# Class 4 — NLP Tokenization & Text Processing (Catch‑Up Notes)

## 📌 Purpose of This File
These notes are designed to fully replace attendance for **Class 4**.
They include:
- A clear summary
- Detailed explanations
- Practical examples
- Learning resources (blogs & videos)

---

## 🧠 High‑Level Summary

This class focused on **tokenization**, the process of breaking raw text into smaller units (tokens) so machines can understand and process language.
Tokenization is the **foundation of all NLP pipelines**, including sentiment analysis, topic modeling, and large language models.

Key ideas:
- Why tokenization matters
- Types of tokenization
- Text normalization steps
- Common NLP pitfalls
- How tokenization impacts ML models

---

## 🔤 What Is Tokenization?

**Tokenization** is the process of splitting text into:
- Words
- Sub‑words
- Characters

Example:
```
"I love NLP!" → ["I", "love", "NLP", "!"]
```

Why this matters:
- ML models cannot process raw text
- Tokens are mapped to numbers (vectors)
- Token quality affects model accuracy

---

## 🧹 Text Pre‑Processing Steps

### 1. Lowercasing
```
"Hello World" → "hello world"
```

### 2. Removing punctuation
```
"Wow!!!" → "wow"
```

### 3. Removing stopwords
Common words like:
- the, is, and, but

⚠️ Sometimes stopwords matter (e.g. **sentiment analysis**)

### 4. Normalization
- Removing extra spaces
- Handling accents
- Standardizing text

---

## ✂️ Types of Tokenization

### 1. Word Tokenization
Splits by spaces and punctuation.

Pros:
- Simple
- Human‑readable

Cons:
- Struggles with slang & typos

---

### 2. Subword Tokenization
Breaks words into smaller meaningful pieces.

Example:
```
"unhappiness" → ["un", "happy", "ness"]
```

Used by:
- BERT
- GPT models

Benefits:
- Handles unknown words
- Reduces vocabulary size

---

### 3. Character Tokenization
Each character becomes a token.

Pros:
- Language‑agnostic

Cons:
- Very long sequences
- Less semantic meaning

---

## ⚠️ Common Tokenization Challenges

- Emojis 😃
- Hashtags (#NLP)
- URLs
- Contractions ("don't")
- Domain‑specific language

Poor tokenization = poor model performance

---

## 🧪 Why Tokenization Impacts Models

Tokenization affects:
- Vocabulary size
- Model memory usage
- Training time
- Accuracy

Example:
Bad tokens → noisy features → weak predictions

---

## 🧰 Common NLP Libraries

### Python Libraries
- **NLTK**
- **spaCy**
- **scikit‑learn**
- **HuggingFace Tokenizers**

---

## 🧪 Example (Python)

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love NLP", "NLP loves data"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
```

---

## 📚 Recommended Reading (Blogs)

- https://towardsdatascience.com/tokenization-in-nlp
- https://machinelearningmastery.com/natural-language-processing/
- https://huggingface.co/docs/tokenizers

---

## 🎥 Recommended YouTube Videos

- "Tokenization Explained Simply" – StatQuest
- "NLP Preprocessing Tutorial" – freeCodeCamp
- "How BERT Tokenization Works" – HuggingFace

---

## ✅ What You Should Know for Exams / Assignments

You should be able to:
- Define tokenization
- Explain different token types
- Describe preprocessing steps
- Explain why tokenization matters
- Identify tokenization challenges

---

## 📝 Key Takeaway

Tokenization is **not just a technical step** — it fundamentally shapes how machines understand language.
Good tokenization leads to better features, better models, and better results.

---

*Prepared as a complete catch‑up reference.*
