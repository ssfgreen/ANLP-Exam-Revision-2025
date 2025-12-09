# Exam Guidelines by Week:

## Week 1: Fundamentals

### W101 - Introduction

- **Ambiguity:** Lexical (Word Sense), Morphological, Part of Speech, Syntactic (Attachment), Word Order.

### W102 - Words as data

- **Zipf's Law and Sparse Data:** Implications for probability and learning.
- **Word Types and Tokens:** Distinction and counting.
- **Part-of-Speech:** Open-class vs. Closed-class words.
- **Syntactic Roles:** Subject, object, indirect object.
- **Resources:** General principles of corpora; collection and quality issues.
- **Regular Expressions:** Concepts and relevance to tokenisation/extraction.

### W103 - Morphology

- **Morphology:** Stems, Affixes, Root, Lemma.
- **Inflectional and Derivational Morphology:** Differences and examples.
- **Agreement:** Gender, number, case marking.
- **Tokenisation:** Challenges (compounds, subwords) and task structure.
- **Byte-Pair Encoding (BPE) Algorithm:** Steps, simulation by hand, pros/cons.

## Week 2: Probability & N-Grams

### W201 - Probability, Models, and Data

- **Maximum Likelihood Estimation (MLE):** Concept, application, and overfitting errors.
- **Language Modelling:** Task definition (predicting sequence probability).

### W202 - N-Grams

- **N-gram Models:** Probability computation, parameter counting, generative process.
- **Cross-entropy:** Formula and usage as a loss function/metric.
- **Perplexity:** Definition, relationship to cross-entropy, use in evaluation.
- **Long-distance Dependencies:** Failure modes of N-gram models.

### W203 - Smoothing and Sampling

- **Add-One / Add-Alpha Smoothing:** Formula, application, strengths/weaknesses.
- **Sampling:** Random generation from distributions (concept introduction).
- **Training, Development, and Test Sets:** Purpose and proper usage.

## Week 3: Classification & Semantics

### W301 - Text Classification

- **Logistic Regression:** Probability computation, single step output, feature application.
- **Multinomial Logistic Regression (Softmax):** Probability computation, output classes.
- **Bayes' Rule:** Formula and application.
- **Sentiment Analysis / Topic Classification:** Task structure and challenges.

### W302 - Training Logistic Regression

- **Stochastic Gradient Descent (SGD):** Concept and difference from batch GD.
- **Cross-entropy Loss:** Objective function for classification.
- **L2 Regularization:** Formula, usage, and prevention of overfitting.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F-measure (formulas and application).
- **Intrinsic vs. Extrinsic Evaluation:** Differences and trade-offs.

### W303 - Lexical Semantics

- **Word Senses and Relations:** Synonyms, hypernyms, hyponyms.
- **Word Sense Disambiguation:** Task structure and difficulties.
- **Distributional Hypothesis:** Concept ("words in similar contexts have similar meanings").
- **Resources:** WordNet (lexical database).

## Week 4: Embeddings & Neural Networks

### W401 - Dense Word Embeddings

- **Skip-gram with Negative Sampling:** Probability computation, negative sampling mechanism.
- **Negative Sampling:** Approximating softmax for efficiency.
- **Sparse vs. Dense Vector Representations:** Differences and trade-offs.
- **Vector-based Similarity Measures:** Dot Product, Cosine Similarity, Euclidean Distance.
- **Static vs. Contextualized Embeddings:** Introduction to static embeddings (Word2Vec).

### W402 - Multilayer Perceptrons (MLPs)

- **Multilayer Perceptron (Feed-forward Network):** Architecture, parameter counting, forward pass.
- **Sentence-level Embeddings:** Pooling methods (averaging word vectors).
- **Language Modeling:** Neural approaches to the task.

### W403 - Training Neural Nets

- **Backpropagation:** Concept, chain rule application, importance for deep learning.
- **Stochastic Gradient Descent (SGD):** Applied to Neural Networks.
- **Cross-entropy Loss:** Applied to Neural Networks.
- **L2 Regularization:** Weight decay in Neural Networks.

## Week 5: RNNs & Ethics

### W501 - RNNs

- **Recurrent Neural Network (RNN):** Architecture, parameter counting, equations.
- **Long-distance Dependencies:** How RNNs handle them vs N-grams.
- **Teacher Forcing:** Concept, motivation, and pros/cons.
- **Machine Translation / Seq2Seq Tasks:** Task structure.

### W502 - Dialect and Discrimination

- **Dialects:** Definition and relevance to NLP performance/bias.
- **Algorithmic Bias:** Identification (demographic parity) and mitigation.
- **Direct vs. Indirect Discrimination:** Definitions and examples in NLP.
- **Representational vs. Allocational Harm:** Differences and examples.

### W503 - Discrimination and Data Ethics

- **Legal and Ethical Issues:** Privacy, consent, copyright in data collection.
- **Corpora:** Collection and annotation ethics.

## Week 6: Attention & Transformers

### W601 - Attention in Seq2Seq

- **Recurrent Neural Network with Attention:** Attention computation, parameter counting.
- **Machine Translation:** How attention improves seq2seq performance.
- **Long-distance Dependencies:** How attention solves RNN bottlenecks.

### W602 - Self-Attention in Transformers

- **Transformer Architecture (General):** Self-attention mechanism.
- **Long-distance Dependencies:** Transformers vs RNNs.

### W603 - Transformer Architecture

- **Transformer Components:** Feedforward, residuals, LayerNorm.
- **Positional Encodings:** Learned vs. Sinusoidal, Absolute vs. Relative.

## Week 7: Architectures & Transfer Learning

### W701 - Inputs and Outputs

- **Greedy Decoding:** Method and trade-offs.
- **Beam Search:** Method, beam width, trade-offs.
- **Sampling:** Top-k and Top-p (Nucleus) sampling methods.
- **Teacher Forcing:** Applied to Transformer training.

### W702 - Transfer Learning and BERT

- **Transformer (Encoder-only):** BERT architecture and objectives.
- **Masked Language Modelling:** Pre-training objective.
- **Transfer Learning:** Concept, fine-tuning vs. pre-training.
- **Static vs. Contextualised Embeddings:** BERT embeddings vs Word2Vec.
- **Sentence-level Embeddings:** BERT `[CLS]` token.

### W703 - Architectures (T5 & GPT)

- **Transformer (Decoder-only):** GPT architecture and objectives.
- **Transformer (Encoder-Decoder):** T5 architecture.
- **Causal Language Modelling:** Pre-training objective.
- **Denoising Language Modelling:** Pre-training objective.

## Week 8: Scaling & Efficiency

### W801 - In-Context Learning

- **In-context Learning:** Concept and motivation.
- **Zero-shot and Few-shot Learning:** Distinctions and usage.
- **Transformer (Decoder-only):** GPT-3 scale and behaviour.

### W802 - Scaling Laws and Evals

- **LLM Development Phases:** Pre-training requirements (Scaling laws).
- **Metrics for Generative Tasks:** BLEU (Translation), ROUGE (Summarisation).
- **LLM-as-a-judge:** Method and limitations.
- **Win Rate and Elo Ranking:** Pairwise comparison metrics.

### W803 - Memory and Compression

- **LLM Inference Phases:** Initial KV Cache Creation (Prefilling) vs. Auto-regressive Generation (Decoding).
- **Sparse Attention:** (Concept implicit in efficiency discussions).

## Week 9: Advanced Topics

### W901 - Multilingual LLMs

- **Multilingual LLMs:** Concepts and examples (mBERT, mT5).
- **Data Paucity:** Challenges for low-resource languages.
- **Cross-lingual Knowledge Transfer:** Zero-shot transfer, Translate-Train, Translate-Test.
- **Multilingual Evaluation:** Challenges and methods.
- **Contrastive Learning:** (e.g., LaBSE for alignment).

### W902 - Instruction Tuning

- **Supervised Fine-Tuning (SFT):** Concept, motivation, and data needs.
- **LLM Development Phases:** Distinction between Pre-training and Fine-tuning.
- **Open-ended Conversational AI:** Task structure and challenges.

### W903 - RLHF

- **RLHF (Reinforcement Learning from Human Feedback):** Concept, motivation (preferences), and pipeline.
- **RLVR (Reinforcement Learning from Verifiable Rewards):** Concept and motivation.
- **Post-training Objectives:** SFT vs RLHF vs RLVR.
