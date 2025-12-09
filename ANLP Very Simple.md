# **1. MODELS**

### **N-gram LM**

- **Idea:** Predict next token from last N–1 tokens.  
   **Formula:** $P(w_1^T)=\prod_i P(w_i\mid w_{i-N+1}^{i-1})$, MLE $=\frac{C(h,w)}{C(h)}$.
- **Note:** Needs smoothing (Add-α, KN).
- **+** Simple, interpretable.
- **–** No long-distance deps; high sparsity.

### **Logistic Regression**

- **Idea:** Binary classifier via sigmoid.  
   **Formula:** $P(y=1)=\sigma(w^\top x+b)$.
- **Training:** CE loss, SGD.
- **+** Convex & stable.
- **–** Only linear boundaries.

### **Softmax Regression**

- **Idea:** Multiclass linear classifier.  
   **Formula:** $P(y=k)=\frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}$.
- **+** Direct multiclass modelling.
- **–** Still linear; feature-dependent.

### **SGNS (Skip-gram Negative Sampling)**

- **Idea:** Predict context via binary classification (real vs negative).  
   **Loss:** $L=-\log\sigma(u_{pos}^\top v_t)-\sum_i\log\sigma(-u_{neg,i}^\top v_t)$.
- **Use:** Static embeddings.
- **+** Strong distributional vectors.
- **–** One vector per word → no polysemy.

### **MLP**

- **Idea:** $h=f(Wx+b)$ → nonlinear features.
- **Use:** Classification on embeddings.
- **+** Captures nonlinear interactions.
- **–** Ignores sequence structure.

### **RNN**

- **Idea:** Sequential hidden state: $h_t=f(W_x x_t+W_h h_{t-1})$.
- **Use:** LM, tagging.
- **+** Represents variable-length sequences.
- **–** Vanishing gradients → weak long-range memory.

### **RNN + Attention**

- **Idea:** Decoder attends over encoder states:  
   $e_{t,i}=s_t\cdot h_i$, $\alpha=\text{softmax}(e)$, $c_t=\sum\alpha h$.
- **Use:** MT, summarisation.
- **+** Explicit alignment.
- **–** Still sequential → slow.

### **Transformer Encoder-only (BERT)**

- **Idea:** Bidirectional self-attention.  
   **Attention:** $\text{softmax}(\frac{QK^\top}{\sqrt{d}})V$.
- **Train:** Masked LM.
- **+** Strong contextual semantics.
- **–** Not generative.

### **Transformer Decoder-only (GPT)**

- **Idea:** Causal self-attention; autoregressive.  
   **Loss:** $-\sum_t\log P(x_t\mid x_{<t})$.
- **+** Excellent generation.
- **–** No future context.

### **Transformer Encoder-Decoder (T5)**

- **Idea:** Encoder (bidirectional) + decoder + cross-attention.
- **Use:** MT, summarisation.
- **+** Best for seq2seq.
- **–** Most parameters.

---

# **2. LEARNING / OBJECTIVES**

### **MLE**

- Fit parameters to maximise data likelihood.
- **+** Simple, principled.
- **–** Fails with sparsity.

### **Add-α Smoothing**

- $P=\frac{C+\alpha}{\cdot+\alpha|V|}$
- **+** Removes zeros.
- **–** Over-smooths.

### **Cross-Entropy / NLL**

- $H(p,q)=-\sum p\log q$; one-hot → $-\log q(y)$.
- **+** Proper scoring rule.
- **–** Punishes overconfidence heavily.

### **Teacher Forcing**

- Train decoder with gold previous token.
- **+** Stable, fast training.
- **–** Exposure bias.

### **SGD / Backprop**

- Chain rule through layers; update via gradients.
- **+** Scales to large models.
- **–** Non-convex; LR-sensitive.

### **Negative Sampling**

- Multi-binary classification instead of full softmax.
- **+** Fast for large vocab.
- **–** Depends on neg sampling distribution.

### **Contrastive Learning**

- Bring positives close, push negatives apart.
- **+** Great for unlabeled representation learning.
- **–** Needs many negatives; tuning sensitive.

### **Transfer / In-Context / Zero-shot**

- Transfer = finetune pretrained LM.
- In-context = examples in prompt.
- Zero-shot = rely on pretrained abstractions.
- **+** Works with little/no labelled data.
- **–** Sensitive to domain/prompt.

### **Pretraining**

- **Causal LM:** + simple / – no bidirectionality.
- **Masked LM:** + bidirectional / – objective mismatch.
- **Denoising:** + robust / – heavy training.

### **Post-Training**

- **SFT:** + controllable / – dataset bias.
- **RLHF:** + aligns to preferences / – reward hacking.
- **RLVR:** + simpler / – still biased.

---

# **3. DECODING**

### **Greedy**

- Max prob each step.
- **+** Fast.
- **–** Local optima.

### **Beam Search**

- Keep top B partial sequences.
- **+** Higher quality.
- **–** Bland / repetitive.

### **Sampling**

- Sample from distribution.
- **+** Creative.
- **–** Can be incoherent.

### **Top-k / Top-p**

- Restrict candidates by size (k) or cumulative prob (p).
- **+** Controlled diversity.
- **–** Sensitive hyperparameters.

---

# **4. SIMILARITY / REGULARISATION**

### **Dot Product**

- $u\cdot v$
- **+** Simple.
- **–** Confounds norm & direction.

### **Cosine Similarity**

- $\frac{u\cdot v}{|u||v|}$
- **+** Direction-only.
- **–** Ignores magnitude.

### **Euclidean Distance**

- $|u-v|$
- **+** True metric.
- **–** Sensitive in high-D.

### **L2 Regularisation**

- Add $\lambda|w|^2$
- **+** Prevents overfitting.
- **–** Too strong → underfit.

---

# **5. EVALUATION**

### **Perplexity**

- $PP=2^H$
- **+** LM uncertainty metric.
- **–** Not comparable across vocabularies.

### **Accuracy**

- **+** Simple.
- **–** Bad for imbalance.

### **Precision / Recall / F1**

- $P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$
- **+** Good for detection tasks.
- **–** F1 ignores TNs.

### **BLEU**

- N-gram precision + brevity penalty.
- **+** MT standard.
- **–** Penalises paraphrases.

### **ROUGE**

- Recall-based overlap.
- **+** Good for summarisation coverage.
- **–** Ignores coherence.

### **LLM-as-Judge / Elo**

- Preference-based ranking.
- **+** Captures semantics.
- **–** Biased, relative only.

---

# **6. LINGUISTICS**

### **Ambiguity**

- Lexical, POS, morphological, syntactic, word-order.
- **+** Core challenge.
- **–** Causes systematic model errors.

### **Agreement**

- Verb–argument, gender/number/case.
- **+** Tests syntax sensitivity.
- **–** Hard for local models.

### **Morphology**

- Stems, lemmas, derivation vs inflection.
- **+** Helps generalisation.
- **–** Creates sparsity.

### **Word Senses**

- Synonymy, hypernymy, similarity.
- **+** Key to semantics.
- **–** Static embeddings conflate senses.

### **Static vs Contextual Embeddings**

- Static = one vector per type; contextual = token-specific.
- **+** Contextual captures polysemy.
- **–** More compute, harder to interpret.

---

# **7. MULTILINGUAL**

### **Data Paucity**

- **+** Motivates transfer.
- **–** Unequal quality across languages.

### **Multilingual LLMs**

- Shared vocab + parameters.
- **+** Zero-shot transfer.
- **–** HRL dominance.

### **Zero-shot / Translate-train / Translate-test**

- **+** Enables LRL tasks with limited data.
- **–** Translation errors / language distance issues.

### **Evaluation**

- XTREME, XGLUE, per-language analysis.
- **+** Broad coverage.
- **–** Domain/script biases.
