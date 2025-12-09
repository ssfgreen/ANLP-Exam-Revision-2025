# **Task → Model Type Cheat Sheet**

## **1. Classification (choose _one_ label)**

**Models:**

- Logistic Regression (binary → sigmoid)
- Softmax Regression / Multinomial Logistic Regression (multi-class → softmax)
- MLP (classification head)
- BERT encoder + classification head

**Key idea:**  
👉 Exactly **one** label → **softmax** (or **sigmoid** if binary).

---

## **2. Multi-Label Classification (choose _multiple_ labels)**

**Models:**

- Sigmoid layer (one per label)
- BCE loss

**Key idea:**  
👉 Labels are **independent**, can all be 0/1 → **sigmoid for each label**, **not softmax**.

---

## **3. Similarity / Retrieval**

**Models & Measures:**

- Cosine similarity
- Dot product
- Euclidean distance
- SGNS / contrastive objectives

**Key idea:**  
👉 Score how close vectors are, not classify.

---

## **4. Language Modelling (next-token prediction)**

**Models:**

- N-grams
- RNN LMs
- GPT (Transformer decoder-only)

**Key idea:**  
👉 Predict **P(wₜ | w<ₜ)** → autoregressive cross-entropy.

---

## **5. Sequence-to-Sequence (conditional generation)**

**Models:**

- RNN + Attention
- Transformer Encoder–Decoder (e.g., T5)

**Key idea:**  
👉 Map input sequence → output sequence using encoder + decoder.

---

## **6. Representation Learning**

**Models:**

- SGNS (word2vec / negative sampling)
- BERT encoders (masked LM)

**Key idea:**  
👉 Learn **embeddings** for downstream tasks.

---

## **7. Generative Modelling**

**Models:**

- GPT (decoder-only Transformers)
- Seq2seq Transformers

**Key idea:**  
👉 Generate coherent text via autoregressive or encoder–decoder decoding.

---

# **TLDR**

- **Softmax = pick ONE class.**
- **Sigmoid = pick ANY subset of labels.**
- **LMs = next token.**
- **Seq2seq = transform input → output.**
- **Retrieval = similarity, not classification.**
- **Representation learning = embeddings, not predictions.**
