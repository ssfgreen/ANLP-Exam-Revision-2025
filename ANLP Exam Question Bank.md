# ANLP Exam Question Bank
## Part 1: Short Answer & Computational Questions

### Section A: Models, Probability & N-Grams

**1. N-gram Probability Calculation**
Imagine a corpus containing the sentence "the cat sat on the mat" and assume an additive-α smoother with $\alpha=0.5$.

- **a)** Compute the probability of this sentence under a bigram model trained on itself.
- **b)** How would the probability change if $\alpha=0$ (Maximum Likelihood Estimation)?

**2. Smoothing and Perplexity**

- **a)** Given a bigram and a trigram model both trained on the same corpus with MLE, explain which one would have lower perplexity on the _training_ data and why.
- **b)** In a larger corpus, you train bigram and trigram models with Kneser-Ney smoothing and evaluate on a _held-out_ set. Explain which model typically gives lower perplexity.
- **c)** Explain why add-alpha smoothing with $\alpha=0.1$ will give different probability estimates than $\alpha=1.0$. In what scenario might you prefer the smaller alpha value?

**3. Zero Probabilities**
You train a bigram model on a small corpus. The phrase "lovely day" appears in the training data, yet the probability $P(\text{day}|\text{lovely})$ is estimated as 0. How is this possible?

**4. Bayes' Rule Application**
A medical test for a disease has 90% sensitivity and 95% specificity. The disease prevalence is 1%. Use Bayes’ rule to compute the probability that a person has the disease given a positive result.

**5. Zipf's Law**
Given a corpus of word counts, plot the log-rank vs. log-frequency graph and estimate the slope. Discuss why natural language has many low-frequency words and how this affects model design (e.g., the need for subword tokenization).

---

### Section B: Classifiers (Logistic & Softmax Regression)

**6. Binary Logistic Regression**
A binary logistic regression classifier uses features $x_1=$ "contains the word excellent" and $x_2=$ "contains the word terrible". With weights $w=(2.0, -1.0)$ and bias $b=0$:

- **a)** Write down the conditional probability $P(y=1|x)$.
- **b)** Compute it for a review containing both "excellent" and "terrible" ($x=[1,1]$).

**7. Multinomial Logistic Regression**
Extend the above to a multinomial model with three sentiment classes (Negative, Neutral, Positive). Given a $3 \times 3$ weight matrix and a zero bias vector, explain how to compute the log-probabilities and identify the most probable class for a short review. Why is the normalization constant $Z$ computationally expensive if the output space is the entire vocabulary?

**8. Feature Engineering**
Explain the effect of adding a strongly predictive binary feature (e.g., "recommended for award") to a logistic regression model. How are the weights expected to change? What problem arises if this feature perfectly separates the classes in the training data?

**9. Loss Functions & Regularization**

- **a)** For a binary classifier that predicts probabilities 0.9 and 0.1 for two examples with true labels 1 and 0 respectively, compute the cross-entropy loss.
- **b)** Explain L2 regularization. How does it modify the loss function and the gradient update?

---

### Section C: Word Embeddings & Distributional Semantics

**10. Skip-gram with Negative Sampling**

- **a)** Give the objective function of skip-gram with negative sampling.
- **b)** Why do we use negative sampling rather than computing the full softmax over the entire vocabulary?
- **c)** How are negative samples drawn?

**11. Vector Arithmetic**
Using embeddings of dimension two for words _king_, _man_, _woman_, and _queen_:

- **a)** Show how the vector arithmetic $king - man + woman \approx queen$ demonstrates the distributional hypothesis.
- **b)** Compute the dot product, cosine similarity, and Euclidean distance between $v_1=(1,2,0)$ and $v_2=(0,1,2)$. Explain when Cosine similarity is preferable to Euclidean distance.

**12. Sentence Representations & Contrastive Learning**

- **a)** Explain how **Contrastive Learning** is used to train sentence embeddings. What constitutes a "positive" and "negative" pair in this context?
- **b)** How does this differ from simply averaging the static word embeddings of a sentence?

---

### Section D: Neural Networks (FFNNs & RNNs)

**13. Parameter Counting**
A feed-forward network takes a 3-word context represented by 50-dimensional pretrained embeddings and has one hidden layer of size 100. Calculate the number of parameters (weights only) in this network. Where do most parameters lie?

**14. RNN Manual Calculation**
A simple RNN with scalar hidden states uses the recurrence $h_t = \text{ReLU}(w h_{t-1} + u x_t + b)$. Given $w=0.5$, $u=1$, and $b=0$, and inputs $x_1=2, x_2=1$ (assuming $h_0=0$), compute $h_1$ and $h_2$ by hand.

**15. Vanishing Gradients & Long Distance Dependencies**

- **a)** Discuss the vanishing gradient problem in RNNs. Why does it make capturing long-distance dependencies (like subject-verb agreement across clauses) difficult?
- **b)** Give two architectural modifications (e.g., gates, residual connections) that help mitigate it.

**16. Teacher Forcing**
Explain teacher forcing and why it is used during training. What is "exposure bias" during inference?

---

### Section E: Transformers & LLMs

**17. Attention Mechanism**

- **a)** For a two-word sequence, compute the self-attention weights using dot-product attention given tiny query, key, and value vectors.
- **b)** Compare additive vs. multiplicative attention mechanisms in terms of computation.
- **c)** Give possible interpretations of high attention weights between specific tokens (e.g., subject and verb).

**18. Transformer Architectures**
Describe the differences between Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5) architectures. Identify an NLP task best suited to each.

**19. Positional Encodings**

- **a)** Explain the difference between **Absolute** and **Relative** positional encodings.
- **b)** Why are positional encodings strictly necessary for a Transformer to understand syntax?

**20. Scaling Laws**
What are scaling laws in language modelling? If you double the model size (parameters), roughly how much more data do you need to remain compute-optimal according to Chinchilla scaling laws?

**21. Pretraining Objectives**

- **a)** Describe the differences between Causal Language Modelling (CLM) and Masked Language Modelling (MLM).
- **b)** Contrast these with **Denoising Language Modelling** (as used in T5/BART). Why is the denoising objective better suited for sequence-to-sequence tasks than standard CLM?

**22. Post-Training Objectives (SFT, RLHF, RLVR)**

- **a)** Explain the purpose of Supervised Fine-Tuning (SFT) in the post-training phase.
- **b)** Explain the difference between **Reinforcement Learning from Human Feedback (RLHF)** and **Reinforcement Learning with Verifiable Rewards (RLVR)**. For which type of task is RLVR most appropriate?

**23. LLM Inference Phases**
Large Language Model inference consists of two distinct phases: _prefilling_ and _decoding_.

- **a)** Explain what happens during the prefilling phase regarding the Key-Value (KV) cache.
- **b)** Why is the prefilling phase typically compute-bound while the decoding phase is memory-bandwidth-bound?

---

### Section F: Algorithms & Decoding

**24. Byte-Pair Encoding (BPE)**
Given the word counts: _drive_ (5), _shift_ (3), and _a_ (10). Simulate the BPE algorithm until the vocabulary size is 16 (including basic characters), following the "break ties from the left" rule. Show the sequence of merges.

**25. Decoding Strategies**
A model outputs probabilities $[0.6, 0.3, 0.1]$ over the next three tokens.

- **a)** Show what token would be chosen by: Greedy decoding, Top-k sampling (k=2), and Nucleus (Top-p) sampling (p=0.7).
- **b)** Explain the trade-offs between beam search and sampling when generating text for creative writing vs. translation.

**26. Backpropagation**
Outline the steps of backpropagation for a two-layer neural network. Explain why mini-batch SGD often leads to faster convergence than full-batch gradient descent.

**27. Regular Expressions**
Write a Regular Expression to extract all hashtags from a tweet (strings starting with '#' followed by alphanumeric characters). Explain one limitation of using Regex for tokenization compared to BPE.

---

### Section G: Linguistics, Evaluation & Ethics

**28. Ambiguity**

- **a)** The sentence "I saw the man with the telescope" is ambiguous. Provide two unambiguous paraphrases and identify whether the ambiguity is syntactic or lexical.
- **b)** Explain why temporary (garden-path) ambiguities pose challenges for incremental parsers.

**29. Standard Evaluation Metrics**

- **a)** In a spam-filtering scenario, predicted and gold labels for 10 messages are given. Compute precision, recall, and F1.
- **b)** For machine translation, compute BLEU-1 and BLEU-2 scores for a candidate translation against a reference. Discuss the limitations of BLEU.

**30. Modern Evaluation (LLM-as-a-judge)**

- **a)** Explain the concept of "LLM-as-a-judge" for evaluating open-ended text generation.
- **b)** Define "win rate" in this context.
- **c)** Why might an **Elo ranking** system be preferred over simple win rates when comparing multiple models?

**31. Algorithmic Bias**

- **a)** Define algorithmic bias. Give an example of direct vs. indirect discrimination in an NLP system.
- **b)** Describe the difference between **representational harm** (e.g., negative stereotyping) and **allocational harm** (e.g., withholding resources).

**32. Multilingual NLP**

- **a)** Explain why data sparsity is a challenge for morphologically rich languages.
- **b)** Compare **Zero-shot cross-lingual transfer**, **Translate-train**, and **Translate-test** methods.

---

---

## Part 2: Extended Scenario Questions

### Scenario 1: Information Retrieval for Legal Documents

You work for a legal firm building a system to retrieve case law based on queries like "employment discrimination age".

- **a) Representation:** Compare using TF-IDF weighted bag-of-words vs. Dense embeddings from a pretrained BERT model. How would you handle the query and document representation in each? How do you compute relevance?
- **b) Out-of-Vocabulary:** Legal terms often contain specific jargon. How would each approach handle synonyms or terms not in the training vocabulary?
- **c) Evaluation:** You have 50 queries with annotated relevant documents. Specify an appropriate metric (e.g., Mean Reciprocal Rank, Precision@k) and justify your choice.

### Scenario 2: Code Completion Language Model

You are building a language model for Python code completion.

- **a) Architecture:** Would you use a Decoder-only (GPT) or Encoder-Decoder (T5) architecture? Justify your choice based on the nature of code generation.
- **b) Capacity:** Your model has 6 layers, 512-dim embeddings, and 8 heads. Calculate the approximate parameter count for the embedding layer and one self-attention layer.
- **c) Troubleshooting:** Training perplexity is 1.2, but Validation perplexity is 5.8. What is this problem called? Suggest two specific techniques (e.g., Dropout, Early Stopping, L2 Regularization) to address it.

### Scenario 3: Hate Speech Detection & Bias

Social media platform "SafeSpace" wants to flag hate speech. They have 50,000 labeled posts.

- **a) Model Choice:** Compare Logistic Regression with hand-crafted features vs. a Fine-tuned BERT model. Consider trade-offs regarding interpretability and performance.
- **b) Bias Analysis:** Annotators were 90% from the US. You find the model flags African American Vernacular English (AAVE) as hate speech 3x more often than Standard English for neutral content.
  - Is this representational or allocational harm?
  - Suggest a data-centric mitigation strategy (e.g., changing the annotation guidelines or sampling strategy).
- **c) Deployment:** You want to deploy in Germany (strict laws) and India (many languages). What additional steps are required regarding legal compliance and multilingual support?

### Scenario 4: Low-Resource Machine Translation

You are building a translation system for Yoruba (West African language) to English. You have very limited parallel data but large amounts of monolingual Yoruba text.

- **a) Strategies:** Compare three approaches:
  1.  Train from scratch on limited parallel data.
  2.  **Back-Translation:** Use an English-to-Yoruba system to generate synthetic parallel data from the English monolingual corpus.
  3.  **Multilingual Pretraining:** Fine-tune mBERT or XML-R.
- **b) Morphology:** Yoruba is a tonal language. How might this affect tokenization? Would BPE help or hinder compared to word-level tokenization?
- **c) Evaluation:** You cannot afford many bilingual human annotators. Describe a cost-effective evaluation strategy (e.g., using BLEU/COMET alongside a small human spot-check).

### Scenario 5: Sentiment Analysis Domain Adaptation

You have a movie review sentiment classifier (90% accuracy). You apply it to restaurant reviews and accuracy drops to 72%.

- **a) Error Analysis:** Why did performance drop? Give examples of features (words) that might shift meaning between domains (e.g., "cheap", "small").
- **b) Adaptation:** You have 1,000 labeled restaurant reviews. Compare:
  1.  Fine-tuning the movie model on the restaurant data.
  2.  Training a fresh model on only the restaurant data.
  3.  Multi-task learning on both.
- **c) Feature Transfer:** Which features (e.g., "boring", "tasty", review length) are domain-invariant and which are domain-specific? How would you computationally detect domain-specific keywords?

### Scenario 6: Conversational AI & Hallucination

A company is building a customer service chatbot using GPT-4 for insurance claims.

- **a) Hallucination:** The bot invents non-existent policies. Explain why LLMs hallucinate based on their training objective (next-token prediction). Suggest Retrieval Augmented Generation (RAG) as a mitigation strategy.
- **b) RLHF & Alignment:** Describe how Reinforcement Learning from Human Feedback works to align the model. How does the Reward Model function?
- **c) Safety:** The model is "jailbroken" by a user to produce offensive content. Discuss the role of the "system prompt" and post-hoc filtering (guardrails).
