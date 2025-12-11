# **1. Single-Label Text Classification**

_(Logistic Regression, Softmax Regression, Linear Classifiers, BERT Fine-Tuning)_

---

## **Data Shape**

- Input: **fixed-size vector** (from TF-IDF, averaged embeddings, or CLS embedding).
- Output: **one label** from K classes → **mutual exclusivity enforced**.
- Target: **one-hot vector**.

---

## **Objective: Softmax Cross-Entropy**

Softmax converts logits to a probability distribution:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Loss for gold class:

$$
L = -\log \hat{y}_{\text{gold}}
$$

Equivalent to **maximum likelihood** under a categorical distribution.

---

## **Evaluation Metrics**

**Accuracy:** best when label distribution is balanced.

$$
\text{acc}=\frac{\text{correct}}{N}
$$

**Precision / Recall / F1:** needed when classes differ in frequency.

$$
P=\frac{TP}{TP+FP},\ R=\frac{TP}{TP+FN},\
F_1=\frac{2PR}{P+R}
$$

**Micro vs Macro**

- **Micro** = treats all predictions equally → sensitive to majority class.
- **Macro** = averages class-wise → detects poor minority-class performance.

---

## **Pros**

- **Convex** → guaranteed global optimum.
- Strong, interpretable **baseline** for many tasks.
- Works well with **sparse features** (n-grams).

## **Cons**

- Only **linear** decision boundaries.
- Cannot capture **order**, **syntax**, or long-range dependencies.

---

## **Where used**

- Sentiment analysis
- Topic classification
- POS tagging baselines
- Any “bag-of-words → label” task

---

# **2. Multi-Label Text Classification**

_(Multi-topic news labels, toxicity categories, emotion tagging)_

### **Example Models**

- **TF-IDF + One-vs-Rest Logistic Regression**
- **CNN/RNN classifiers with sigmoid output**
- **BERT sigmoid multi-label head**

---

## **Data Shape**

- Output: vector **[y₁,…,y_K]**, each ∈ {0,1}.
- **Multiple labels can be active** at once.
- No mutual exclusion → different objective.

---

## **Objective: Sigmoid + Binary Cross-Entropy**

$$
\hat{y}_k = \sigma(z_k)
$$

$$
L = -\sum\_{k=1}^K \big[y_k \log \hat{y}_k + (1-y_k)\log(1-\hat{y}_k)\big]
$$

Each label is an **independent binary classification task**.

---

## **Evaluation**

- **Micro-F1**: handles label imbalance well.
- **Macro-F1**: penalises poor minority-label performance.

Multi-label F1 is computed **after thresholding** probabilities (commonly 0.5, but tuned).

---

## **Pros**

- Handles overlapping, correlated topics.
- Produces **per-label confidence scores**.

## **Cons**

- Choosing a threshold strongly affects metrics.
- Many rare-label datasets suffer from **sparsity**.

---

# **3. Retrieval / Similarity Models**

_(Information retrieval, semantic search, embeddings, word vectors)_

### **Example Models**

- **Word2Vec Skip-Gram with Negative Sampling (SGNS)**
- **Doc2Vec**
- **Sentence-BERT**
- **Dense Passage Retrieval (DPR)**

---

## **Similarity Measures**

**Dot product:** captures **overlap + magnitude**.

$$
u \cdot v = \sum_i u_i v_i
$$

**Cosine similarity:** magnitude-invariant → compares **direction**.

$$
\cos(u,v)=\frac{u\cdot v}{|u||v|}
$$

**Euclidean distance:** geometric distance.

$$
d(u,v)=\sqrt{\sum_i (u_i-v_i)^2}
$$

---

## **SGNS Objective: Binary Classification Over Pairs**

### **Positive pair probability**

$$
P(+|w_t,w_k)=\sigma\left(v(w_t)\cdot c(w_k)\right)
$$

### **Negative samples**

$$
P(-|w_t,w_i^-)=\sigma\left(-,v(w_t)\cdot c(w_i^-)\right)
$$

### **Loss**

$$
L = -\log \sigma(v_t\cdot c_{+}) - \sum_{i=1}^K \log \sigma(-v_t\cdot c_i^{-})
$$

Interpretation:
**Increase (t, positive) similarity, decrease (t, negative) similarity.**

---

## **Evaluation**

- **Precision@1 / Recall@k** for ranking tasks.
- For embeddings: **cosine similarity quality**, clustering coherence, intrinsic tests.

---

## **Pros**

- Very efficient compared to full softmax.
- Learns high-quality **semantic embeddings**.

## **Cons**

- Sensitive to the sampling distribution.
- Embedding norms + anisotropy issues without normalisation.

---

# **4. Language Modelling (Generative Models)**

_(N-gram LMs, RNN LMs, GPT-style Transformers)_

### **Example Models**

- **n-gram language models** with smoothing
- **LSTM / GRU LMs**
- **GPT-2 / GPT-3 / LLaMA**

---

## **Objective: Next-Token Cross-Entropy**

$$
L = -\sum_t \log P(w_t \mid w_{<t})
$$

This is **MLE** under an autoregressive factorisation of the sequence.

One-hot view:

$$
L = -\log q(\text{gold token})
$$

---

## **Perplexity**

$$
PP = 2^{H_M}
$$

Lower PP → better model fit to distribution.

---

## **Pros**

- Captures sequential structure.
- Forms basis of **all modern LLMs**.

## **Cons**

- Perplexity may not correlate with generation quality.
- Predict-next-token objective doesn’t naturally align with tasks like summarisation.

---

# **5. Masked Language Modelling (BERT)**

_(BERT, RoBERTa, XLM-R — bidirectional encoders)_

### **Example Models**

- **BERT-base**
- **RoBERTa** (no NSP, dynamic masking)
- **DeBERTa**

---

## **Objective**

Predict masked tokens (≈15%):

$$
L = -\sum_{\text{masked } t} \log P(w_t | x_{\text{masked}})
$$

This teaches **deep bidirectional contextual understanding**.

---

## **Evaluation**

- Masked-token accuracy
- Cross-entropy on masked positions

---

## **Pros**

- Extremely strong for **classification**, **NER**, **QA (extractive)**.
- Learns rich **contextual** representations.

## **Cons**

- Not generative.
- Masking is artificial; may mismatch natural text.

---

# **6. Sequence-to-Sequence (Seq2Seq)**

_(MT, summarisation, dialogue, data-to-text)_

### **Example Models**

- **RNN Encoder–Decoder** with Bahdanau or Luong attention
- **Transformer encoder–decoder** (T5, BART, mBART)
- **Classic NMT systems**

---

## **Objective**

$$
L = -\sum_t \log P(y_t \mid y_{<t}, x)
$$

Decoder is autoregressive over output tokens.

---

## **Teacher Forcing**

During training:

- Decoder receives **gold previous token** rather than its own prediction.
- **Exposure bias:** at inference, model receives its own disturbed outputs, causing drift.

---

## **Evaluation Metrics**

- **BLEU** (n-gram precision + brevity penalty)
- **ROUGE-N / ROUGE-L** (recall, LCS)
- **METEOR**, **BERTScore**, **chrF** (semantic/character-level)

---

## **Pros**

- Captures input → output conditional structure.
- Excellent for MT and summarisation.

## **Cons**

- N-gram metrics ignore meaning.
- Exposure bias unless mitigated (scheduled sampling, etc.).

---

# **7. LLM-as-a-Judge + Elo / Win Rate**

### **Win Rate**

$$
\text{WinRate}(A)=P(A \text{ beats } B)
$$

Judge outputs preferences across many examples, yielding a probability.

### **Elo Ranking**

Repeated pairwise comparisons → estimate a **global latent quality score**.

---

## **Pros**

- Works for open-ended tasks: summarisation, reasoning, QA.
- Much cheaper than human annotation.

## **Cons**

- Biased by judge model (style, verbosity, hallucination tolerance).
- Produces **relative**, not absolute, correctness.

---

# **8. Regularisation & Optimisation**

## **L2 Regularisation (Weight Decay)**

$$
L' = L + \lambda \sum_j w_j^2
$$

Encourages **small weights**, reduces overfitting, smooths decision boundaries.

---

## **SGD Update**

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

Stochastic updates approximate the global optimum over time.

---

## **Backpropagation**

Uses the **chain rule** to compute gradients layer-by-layer:

$$
\delta = \frac{\partial L}{\partial z}
$$

Then propagate backward through weights and activations.

---

# **TLDR: Exam Flash Sheet**

### **Single-Label**

- **Softmax CE**, accuracy/F1
- Mutually exclusive classes

### **Multi-Label**

- **Sigmoid + BCE**, micro-F1
- Independent labels

### **Retrieval**

- SGNS objective
- Cosine + dot product
- P@1 for ranking

### **Language Modelling**

- Next-token CE
- Perplexity
- GPT, RNN LM, n-gram LM

### **Masked LM**

- Masked CE
- Bidirectional; good for encoders

### **Seq2Seq**

- Decoder CE
- BLEU/ROUGE
- Attention; exposure bias

---

If you want, I can also **compress this into a single-page exam cheat-sheet**, or **add worked examples** for any section.
