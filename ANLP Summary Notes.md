## 1️⃣ MODELS

---

### **N-gram Models — Local Context LM**

**Idea**

- Approximate next-word probability using only the last **N–1 words**  
   $$P(w_i \mid w_1^{i-1}) \approx P(w_i \mid w_{i-N+1}^{i-1})$$

**Key formulas**

- Sentence probability:  
   $$P(w_1^T) = \prod_{i=1}^T P(w_i \mid w_{i-N+1}^{i-1})$$
- MLE estimate:  
   $$P(w \mid h) = \frac{C(h,w)}{C(h)}$$

**Why smoothing?**

- Many N-grams never appear → **zero probabilities** → catastrophic under LM / MT.
- Add-α, Kneser–Ney, backoff/interpolation.

**Tasks**

- Classical **language modelling**, ASR, spelling correction, early MT baselines.

**Pros / Cons (1 each)**

- **+** Extremely simple, interpretable, good for showing sparsity issues.
- **–** Fails badly on **long-distance dependencies**; data-hungry and sparse.

---

### **Binary Logistic Regression — Linear Classifier**

**Idea**

- Map feature vector **x** to probability of class **1** using sigmoid.

**Key formula**

- $$
  P(y=1 \mid x) = \sigma(w^\top x + b)
    = \frac{1}{1+e^{-(w^\top x + b)}}
  $$

**Training**

- Data: $(x^{(i)}, y^{(i)})$ with $y\in{0,1}$.
- Loss (per example):  
   $$L = -\big[y \log \hat y + (1-y)\log(1-\hat y)\big]$$
- Optimised with (stochastic) gradient descent; often L2 regularised.

**Typical tasks**

- Binary **sentiment** (pos/neg), **spam** vs ham, toxicity yes/no.

**Pros / Cons (1 each)**

- **+** Convex objective → **unique optimum**, stable and interpretable.
- **–** Only **linear** decision boundaries; relies heavily on feature engineering.

---

### **Softmax Regression — Multiclass Classifier**

**Idea**

- Generalisation of logistic regression to **K classes** with softmax.

**Key formulas**

- Scores: $s_k = w_k^\top x + b_k$
- Softmax:
  $$
  P(y=k\mid x) =
  \frac{\exp(s_k)}{\sum_{j=1}^K \exp(s_j)}
  $$

**Training**

- Loss (cross-entropy):  
   $$L = -\log P(y_{\text{true}} \mid x)$$

**Typical tasks**

- **POS tagging** (independent per token), **topic classification**, intent classification.

**Pros / Cons (1 each)**

- **+** Direct, simple way to handle **multiclass** outputs.
- **–** Still **linear in features**; cannot capture complex interactions without extra engineering.

---

### **Skip-Gram with Negative Sampling (SGNS) — Static Word Embeddings**

**Idea**

- Uses a **target word** to predict **context words**; neural model learns vector representations by casting prediction as **binary classification**:  
   “Is this context word a real neighbour or a negative sample?”

**Key loss**

- For target word $v_t$, positive context $u_{pos}$ and negatives $u_{neg,i}$:
  $$
  L = -\log\sigma(u_{pos}^\top v_t)
  - \sum_i \log \sigma(-u_{neg,i}^\top v_t)
  $$

**Interpretation**

- **Positive term:** maximise probability that **true context** is labelled positive.
- **Negative term:** minimise probability that sampled **noise words** are labelled positive.
- Result: words with similar contexts get similar embeddings.

**Tasks**

- Unsupervised **word embeddings** used in downstream models: classification, NER, similarity, etc.

**Pros / Cons (1 each)**

- **+** Learns high-quality **distributional embeddings** efficiently on large corpora.
- **–** One vector per **word type** → cannot represent **polysemy / sense shifts**.

---

### **Multilayer Perceptron (MLP) — Feed-forward Network**

**Idea**

- Stack of fully connected layers with **nonlinearity**:  
   $$h = f(Wx + b), \quad y = g(Uh + c)$$

**Role**

- Learns **nonlinear mappings** from input features (e.g. sentence embeddings) to outputs (labels).

**Tasks**

- Sentence-level classification: sentiment, topic, NLI (on top of embeddings).

**Pros / Cons (1 each)**

- **+** Captures **nonlinear feature interactions**; universal approximator with enough width.
- **–** No built-in handling of **sequence structure** or order; needs pre-computed fixed-size features.

---

### **RNN — Sequential Model**

**Idea**

- Process tokens one by one, maintain **hidden state** as summary of the past.

**Key update**

- $$h_t = f(W_x x_t + W_h h_{t-1} + b)$$

**Usage**

- At each time step, $h_t$ can be used to predict next token (LM) or token label (tagging).

**Typical tasks**

- **Language modelling**, sequence classification, tagging, early seq2seq.

**Pros / Cons (1 each)**

- **+** Naturally handles **variable-length sequences** and order.
- **–** Suffers from **vanishing/exploding gradients**, struggles with very long dependencies.

---

### **RNN with Attention — Seq2Seq + Alignment**

**Idea**

- Encoder RNN → sequence of hidden states.
- Decoder RNN → generates output; at each step, uses **attention** over encoder states.

**Key attention steps**

- Score: $e_{t,i} = \text{score}(s_t, h_i)$ (e.g. dot or small MLP)
- Weights: $\alpha_{t,i} = \text{softmax}_i(e_{t,i})$
- Context: $c_t = \sum_i \alpha_{t,i} h_i$

**Tasks**

- **MT**, summarisation, seq2seq tasks before Transformers.

**Pros / Cons (1 each)**

- **+** Explicit **alignment mechanism**; lets decoder focus on relevant source tokens.
- **–** Still **sequential** on encoder/decoder; slower than parallel Transformer and still limited on very long sequences.

---

### **Transformer Encoder-Only (BERT-style)**

**Idea**

- Stack of **self-attention + FFN** with residuals & LayerNorm; outputs **contextual embeddings** for each token using **bidirectional context**.

**Key self-attention**

- $$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$
- $$\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Training**

- **Masked LM**: randomly mask some tokens, predict them from both left and right context.

**Tasks**

- **Classification** (use [CLS]), **NER**, **QA**, semantic similarity.

**Pros / Cons (1 each)**

- **+** Strong **bidirectional representations**, great for understanding tasks.
- **–** Not natively **generative**; attention is **$O(n^2)$** in sequence length.

---

### **Transformer Decoder-Only (GPT-style)**

**Idea**

- Same blocks as encoder, but with **causal mask** so position _t_ only attends to $\le t$.
- Trained as **autoregressive LM**.

**Objective**

- $$\mathcal{L}_{LM} = -\sum_t \log P(x_t \mid x_{<t})$$

**Tasks**

- **Text generation**, code, reasoning, few-shot / zero-shot prompting.

**Pros / Cons (1 each)**

- **+** Excellent **generative** modelling and in-context learning.
- **–** No direct access to **future context** in a single pass; same $O(n^2)$ attention cost.

---

### **Transformer Encoder–Decoder (T5-style)**

**Idea**

- **Encoder**: BERT-like; **decoder**: GPT-like + **cross-attention** to encoder outputs.

**Cross-attention**

- Queries from decoder state, keys/values from encoder hidden states.
- Allows decoder to “look back” at encoded input each step.

**Tasks**

- **Machine translation**, **abstractive summarisation**, generic text-to-text.

**Pros / Cons (1 each)**

- **+** Very powerful for **input→output** mapping (seq2seq).
- **–** Heavier: two stacks of layers → **more parameters & compute**.

---

## 2️⃣ LEARNING / PROBABILITY ESTIMATION

---

### **Maximum Likelihood Estimation (MLE)**

**Idea**

- Choose parameters $\theta$ that **maximise** likelihood of data:  
   $$\theta^* = \arg\max_\theta \prod_i P_\theta(x_i)$$

**Characteristic errors**

- Overfits rare patterns; zeros for unseen events.

**Pros / Cons (1 each)**

- **+** Statistically principled; simple to compute with counts.
- **–** Fails badly with **sparse data**; needs smoothing/regularisation.

---

### **Add-α Smoothing**

**Idea**

- Fix zero counts by adding α pseudo-counts.

**Formula**

- $$P(w \mid h)=\frac{C(h,w)+\alpha}{C(h)+\alpha |V|}$$

**Pros / Cons (1 each)**

- **+** Very simple way to avoid zero probabilities.
- **–** Over-smooths, especially frequent events; not state-of-the-art.

---

### **Cross-Entropy / Negative Log-Likelihood**

**Definition**

- $$H(p,q) = -\sum_w p(w)\log q(w)$$
- For one-hot label $y$:  
   $$L = -\log q(y)$$

**Meaning**

- Expected “surprise” of true labels under model distribution.

**Pros / Cons (1 each)**

- **+** Proper scoring rule; aligns with **MLE**.
- **–** Highly penalises **over-confident wrong** predictions.

---

### **Teacher Forcing**

**Idea**

- During training, decoder sees **gold previous token**, not its own prediction.

**Benefit / harm**

- Faster convergence; but at test time model conditions on its **own outputs** → **exposure bias**.

**Pros / Cons (1 each)**

- **+** Stable, efficient training for seq2seq.
- **–** Gap between train/test conditions → exposure bias errors.

---

### **Stochastic Gradient Descent (SGD) + Backprop**

**Idea**

- Update parameters using noisy gradient estimates from mini-batches.
- **Backprop** uses chain rule to propagate gradients layer by layer.

**Pros / Cons (1 each)**

- **+** Scales to large datasets and deep models.
- **–** Non-convex optimisation; sensitive to learning rate, can get stuck in bad minima / saddle points.

---

### **Negative Sampling (as in SGNS)**

**Idea**

- Replace expensive softmax over V with **multiple binary logistic tasks** (pos vs sampled negs).

**Pros / Cons (1 each)**

- **+** Huge speedup for large vocabularies.
- **–** Approximate; quality depends on **negative sampling distribution**.

---

### **Contrastive Learning**

**Idea**

- Learn representations by **pulling together positives** and **pushing apart negatives** in embedding space.

**Generic InfoNCE-style loss**

- $$
  L = -\log \frac{\exp(\text{sim}(x,x^+)/\tau)}
    {\exp(\text{sim}(x,x^+)/\tau) + \sum_{neg}\exp(\text{sim}(x,x^-)/\tau)}
  $$

**Pros / Cons (1 each)**

- **+** Strong representations without labels.
- **–** Requires many negatives and careful temperature/negative design.

---

### **Transfer Learning / In-Context / Few-/Zero-shot**

**Transfer learning**

- Pretrain on large generic data → **finetune** on task-specific labeled data.

**In-context learning**

- Provide examples in prompt; model **conditions** on them, no weight updates.

**Few-/Zero-shot**

- Evaluate with few or no task-specific labels; rely on pretrained knowledge & prompts.

**Pros / Cons (1 each)**

- **+** Huge performance gains in low-data regimes.
- **–** Domain shift / prompt sensitivity can give brittle behaviour.

---

### **Pretraining Objectives**

- **Causal LM:** predict next token given past.
  - **+** Simple, aligns with generative use.
  - **–** Only left-to-right context.
- **Masked LM:** mask tokens, predict them from both sides.
  - **+** Bidirectional context → strong encoders.
  - **–** Objective mismatched with left-to-right decoding.
- **Denoising LM:** corrupt input, reconstruct full sequence.
  - **+** Robust representations.
  - **–** More complex training pipeline.

---

### **Post-Training: SFT, RLHF, RLVR**

- **SFT (Supervised Fine-Tuning)**
  - Train on human-written demonstrations.
  - **+** Good controllability & style.
  - **–** Limited by dataset quality/bias.
- **RLHF**
  - Train reward model on human preferences; use RL (e.g. PPO) to update policy.
  - **+** Aligns with human preferences, handles “no single correct answer” tasks.
  - **–** Reward hacking; optimises for reward model quirks, not true quality.
- **RLVR / Direct Preference Optimisation**
  - Optimise directly on pairwise preferences, often no explicit reward model.
  - **+** Simpler pipeline than full RLHF; more stable.
  - **–** Still inherits annotator bias; risk of mode collapse.

---

## 3️⃣ DECODING / INFERENCE STRATEGIES

---

### **Greedy Decoding**

- Always choose highest-prob token.
- **+** Very fast and simple.
- **–** Easily gets stuck in **local optima**, missing better sequences.

---

### **Beam Search**

- Keep top **B beams** (partial sequences), expand them each step.
- **+** Better global search, higher BLEU in MT.
- **–** Can produce bland / repetitive text; computationally heavier.

---

### **Sampling, Top-k, Top-p**

- **Sampling:** directly sample from full distribution.
- **Top-k:** truncate to top k tokens, renormalise, sample.
- **Top-p (nucleus):** smallest prefix with cumulative prob ≥ p, renormalise, sample.
- **+** Enable **diverse** and creative generation.
- **–** If tuned badly, can be incoherent (too random) or dull (too deterministic).

---

## 4️⃣ KEY MATH / SIMILARITY

---

### **Dot Product**

- $$u\cdot v = \sum_i u_i v_i$$
- Mixed measure of **magnitude + alignment**.
- **+** Very cheap to compute; natural in linear models.
- **–** Confuses similarity with vector length.

---

### **Cosine Similarity**

- $$\cos(u,v) = \frac{u\cdot v}{|u|,|v|}$$
- **+** Captures **pure directional** similarity, invariant to norm.
- **–** Ignores magnitude, can be misleading in anisotropic spaces.

---

### **Euclidean Distance**

- $$d(u,v)=\sqrt{\sum_i (u_i - v_i)^2}$$
- **+** True metric; intuitive as geometric distance.
- **–** Less stable in very high dimensions; sensitive to scaling of features.

---

### **L2 Regularisation**

- Add penalty $\lambda |w|^2$ to loss.
- **+** Reduces overfitting by discouraging large weights.
- **–** Too strong λ → underfitting / oversmoothing.

---

## 5️⃣ EVALUATION

---

### **Perplexity (PPL)**

- $$PP = 2^{H_M}$$
- “Effective average number of choices” per token.
- **Appropriate:** LM evaluation, comparing LMs on same vocab.
- **Limitations:** Doesn’t correlate reliably with downstream task quality.
- **+** Directly measures LM uncertainty.
- **–** Not comparable across different tokenisations/vocabularies.

---

### **Accuracy**

- $$\text{acc} = \frac{\text{# correct}}{\text{# total}}$$
- **Appropriate:** balanced classification tasks, e.g. topic classification.
- **Limitations:** hides performance on rare classes.
- **+** Simple, intuitive.
- **–** Misleading for **imbalanced** data.

---

### **Precision, Recall, F1**

- $$
  P = \frac{TP}{TP+FP},\quad R = \frac{TP}{TP+FN},\quad
    F_1 = 2\frac{PR}{P+R}
  $$
- **Precision:** reliability of positive predictions.
- **Recall:** coverage of true positives.
- **F1:** balance of the two (esp. for imbalanced labels).
- **+** Capture trade-offs in detection tasks (NER, toxicity).
- **–** F1 ignores **true negatives** and class-specific priorities.

---

### **BLEU (MT)**

- N-gram precision (1–4) with **brevity penalty**.
- Good for literal MT systems; surface-based.
- **+** Historical standard; easy to compute and compare.
- **–** Penalises valid **paraphrases**, blind to deeper semantics.

---

### **ROUGE (Summarisation)**

- **ROUGE-N:** recall of n-gram overlap.
- **ROUGE-L:** LCS-based recall.
- **+** Focuses on **coverage** of key content, good for extractive summaries.
- **–** Ignores coherence, hallucinations, factuality.

---

### **LLM-as-a-Judge**

- Use LLM to score or rank outputs.
- **+** Can capture **semantic quality** that n-gram metrics miss; scalable.
- **–** Biased, non-transparent; models can be optimised to exploit judge quirks.

---

### **Win Rate & Elo**

- **Win rate:** $P(\text{A beats B})$ over pairwise comparisons.
- **Elo:** converts pairwise comparisons into global ranking.
- **+** Natural for “no single right answer” tasks (chat, creative).
- **–** Elo is **relative**, depends on pool, judge biases, and matchup set.

---

### **Intrinsic vs Extrinsic Evaluation**

- **Intrinsic:** evaluate component in isolation (perplexity, word similarity, parsing accuracy).
- **Extrinsic:** evaluate component via impact on downstream task (e.g. NER F1 with different embeddings).
- **+** Intrinsic: fast diagnostics; extrinsic: measures real utility.
- **–** Intrinsic may not correlate with downstream; extrinsic is expensive and noisy.

---

### **Corpora Issues (Collection, Annotation, Distribution)**

- **Collection:** sampling bias, domain mismatch, privacy & consent.
- **Annotation:** inter-annotator agreement, guidelines, cost, cultural bias.
- **Distribution:** licensing, GDPR, documentation (datasheets/model cards).
- **+** Good corpora underpin reliable models & fair evaluation.
- **–** Poorly documented/biased corpora lead to hidden harms and unreproducible results.

---

## 6️⃣ LINGUISTIC & REPRESENTATIONAL CONCEPTS

_(Short, but still with + / –)_

---

### **Ambiguity**

- Types: lexical, morphological, POS, syntactic (attachment), word order.
- **Relevance:** WSD, parsing, MT, QA.
- **+** Central challenge that motivates many NLP tasks.
- **–** Causes systematic errors if models lack contextual sensitivity.

---

### **Agreement & Long-Distance Dependencies**

- Verb-argument, coreference; features like case, number, gender.
- **+** Good testbed for syntax-sensitive models.
- **–** Hard for local models (N-grams), basic RNNs.

---

### **Morphology: Stems, Lemmas, Inflection vs Derivation**

- Inflection: grammatical features (tense, number, case).
- Derivation: makes new lexemes, often changes POS.
- **+** Lemmas & morphological analysis help generalise across forms.
- **–** Rich morphology → **data sparsity** for word-level models.

---

### **Word Senses & Distributional Hypothesis**

- Word senses: synonym, hypernym/hyponym; WSD tasks.
- Distributional hypothesis: “You shall know a word by the company it keeps.”
- **+** Supports embedding-based semantics.
- **–** Static embeddings conflate multiple senses.

---

### **Static vs Contextual Embeddings**

- **Static:** one vector per type (word2vec, GloVe).
- **Contextual:** vector per token-in-context (BERT, GPT).
- **+** Contextual embeddings capture polysemy & syntax.
- **–** More compute; harder to interpret.

---

## 7️⃣ RESOURCES & ETHICS

---

### **Resources (Labeled, Unlabeled, Lexical, Parallel, etc.)**

- Labeled corpora, unlabeled web text, WordNet/FrameNet, UniMorph, EuroParl, etc.
- Also: evaluation benchmarks, human annotators.
- **+** Provide the **ground truth** or raw material enabling modern NLP & LLMs.
- **–** Scarcity, bias, annotation inconsistency, legal/privacy issues.

---

### **Algorithmic Bias**

- Unequal error rates across demographic groups due to data/model/metric.
- **+** Framework for analysing fairness in NLP systems.
- **–** Leads to harms against underrepresented or stereotyped groups.

---

### **Direct vs Indirect Discrimination**

- **Direct:** uses protected attribute explicitly.
- **Indirect:** uses correlated proxies (e.g. postcode, dialect).
- **+** Distinction helps design mitigation strategies.
- **–** Indirect discrimination is hard to detect and prevent.

---

### **Representational vs Allocational Harm**

- **Representational:** stereotypes, erasure, misrepresentation (e.g. biased associations).
- **Allocational:** unequal access to opportunities/resources (jobs, credit, healthcare).
- **+** Makes it clear that harms are not only about accuracy but also **social narratives and material consequences**.
- **–** Difficult to measure and mitigate fully, especially across cultures.

---

## 8️⃣ MULTILINGUAL NLP

---

### **Data Paucity**

- Majority of languages have very little labelled data; sometimes limited unlabeled data.
- **+** Motivates multilingual models, transfer, and low-resource techniques.
- **–** Low-resource languages get **worse models**, reinforcing digital inequality.

---

### **Multilingual LLMs**

- Single model trained on many languages with shared subword vocab, shared parameters (mBERT, XLM-R, etc.).
- **+** Cross-lingual transfer → zero-shot performance on many languages.
- **–** High-resource languages dominate vocabulary and capacity; minority languages under-served.

---

### **Zero-Shot Transfer / Translate-Train / Translate-Test**

- **Zero-shot:** train on EN, test on other languages with no extra training.
- **Translate-train:** translate HRL training data into LRL, train LRL model.
- **Translate-test:** translate LRL test input to HRL, run HRL model.
- **+** Allow useful models for languages with little/no labeled data.
- **–** Translation errors and language distance can significantly degrade performance.

---

### **Multilingual Evaluation**

- Benchmarks: XTREME, XGLUE, etc.; per-language breakdown.
- **+** Gives quantitative picture across many languages.
- **–** Benchmarks may themselves be biased by domain, scripts, & availability.

---
