# 10. Resources

## **1. What types of resources are needed?**

### **A. Labeled Data**

Used when a task requires _supervised learning_.

- **Examples:** POS-tagged corpora, dependency treebanks, NER datasets.
- **Relevant tasks:** tagging, parsing, NER, sentiment classification.
- **Pros:** high-quality signal, task-specific.
- **Cons:** expensive, slow to obtain, annotation inconsistency.

---

### **B. Unlabeled Corpora**

Large-scale text collected from the web or domain sources.

- **Examples:** **CommonCrawl**, Wikipedia, BooksCorpus.
- **Relevant tasks:** LM pretraining, unsupervised embeddings, self-supervised learning.
- **Pros:** very large, cheap, useful for representation learning.
- **Cons:** noisy, biased, legality of scraping unclear.

---

### **C. Lexical & Semantic Resources**

Hand-curated or semi-curated structured databases.

- **Examples:**
  - **WordNet** (synsets, hypernyms).
  - **FrameNet**, **PropBank** (semantic roles).
  - **VerbNet** (verb classes).
- **Relevant tasks:** WSD, semantic similarity, SRL, lexicon-based sentiment.
- **Pros:** interpretable, high precision.
- **Cons:** expensive to build; limited domain coverage; may encode cultural bias.

---

### **D. Morphological Resources**

Used for languages with rich morphology.

- **Examples:** CELEX, UniMorph.
- **Relevant tasks:** morphological parsing, MT, lemmatization.
- **Pros:** reliable structured forms.
- **Cons:** incomplete across languages.

---

### **E. Multilingual & Parallel Corpora**

Aligned text across languages.

- **Examples:** EuroParl, UN Parallel Corpus.
- **Relevant tasks:** MT (alignment models, NMT training), cross-lingual embeddings.
- **Pros:** essential for MT; structured alignment.
- **Cons:** limited domains (parliament, UN), not representative of everyday language.

---

### **F. Evaluation Benchmarks**

Standardised test suites to compare systems.

- **Examples:** GLUE, SuperGLUE, SQuAD, CoNLL shared tasks.
- **Relevant tasks:** QA, NER, coreference, reasoning.
- **Pros:** comparability across models.
- **Cons:** overfitting to benchmarks; narrow task framing.

---

### **G. Human Expertise / Annotators**

Humans provide labels, guidelines, and quality checks.

- **Examples:** Crowdworkers on MTurk; linguists building treebanks.
- **Relevant tasks:** any supervised task.
- **Pros:** high-quality if trained.
- **Cons:** annotation bias, cost, ethical concerns.

---

## **2. Pros & Cons — High-Level Summary**

### **Pros**

- Resources provide structure, ground truth, or massive raw data enabling **learning**.
- Curated lexicons improve **interpretability**.
- Large-scale corpora enable **scaling** of LLMs.
- Benchmarks enable **comparability** and **progress tracking**.

### **Cons**

- Data scarcity in low-resource languages.
- Annotation is expensive and inconsistent.
- Web data is noisy and may contain harmful content.
- Lexicons can be outdated or culturally narrow.
- Benchmark-driven development encourages overfitting and ignores real-world use.

---

## **3. Legal & Ethical Issues to Identify**

### **A. Copyright & Licensing**

- Web text (CommonCrawl) often includes copyrighted material.
- Training use vs distribution use may be legally distinct.
- Some corpora prohibit _commercial_ use.

### **B. Consent**

- Many scraped datasets include text _not intended_ for ML training.
- Private messages, social networks, or forum posts may include personal data.

### **C. Personal Data & Privacy**

- GDPR issues for EU subjects.
- Presence of names, addresses, sensitive attributes.
- Risks: deanonymisation, model memorisation.

### **D. Bias & Representation Harm**

- Large web corpora encode stereotypes (gender, race, dialect).
- Under-representation of minority dialects → model underperformance.
- Lexicons often reflect Western, academic linguistic assumptions.

### **E. Toxicity & Harmful Content**

- Hate speech, misinformation, extremist content in web data.
- Models can reproduce harmful patterns unless filtered.

### **F. Worker Ethics**

- Low-paid annotators exposed to traumatising content.
- Unclear guidelines or inadequate compensation for crowdworkers.

### **G. Transparency & Documentation**

- Need for **datasheets**, **model cards**, provenance metadata.
- Failure to document → misuse, risk, unclear bias sources.

---

## **4. Typical Exam Angles**

You may be asked to:

- Compare resources (e.g., WordNet vs CommonCrawl).
- Identify which resource a task needs and justify why.
- Describe limitations of a given dataset.
- Discuss legal/ethical risks in collecting new data.
- Explain how resource quality affects model performance.

---

# **TLDR**

- Know **resource types** (labeled, unlabeled, lexical, morphological, parallel, benchmarks).
- Know **examples** (WordNet, CommonCrawl, FrameNet, EuroParl).
- Understand their **pros/cons** (coverage, cost, noise, bias).
- Be able to identify **legal/ethical issues** (copyright, privacy, bias, informed consent, annotator welfare).

# 11. Evaluation Concepts & Methods

## **1. Perplexity (PPL)**

How **uncertain** a language model is when predicting the next token.

**What it measures:**

- Perplexity is the model’s **effective average number of choices** at each step.
  - Perplexity ≈ 2 → model is very confident (only ~2 likely next words).
  - Perplexity ≈ 100 → model is very uncertain (many plausible next words).
  - **Lower perplexity = better model (less confused).**
- **Example:** If a model has cross-entropy $H_M = 3$ bits per word, then $PP = 2^3 = 8$
  - “On average, the model behaves like it has about **8** plausible next-word options.”

**Appropriate for:**

- **Language modelling**, next-word prediction, generative pretraining.
- **Model comparison:**
  - Unigram → high perplexity (no context); Bigram → lower; Trigram → lower still
  - More context → fewer effective choices → **lower perplexity**.

**Why:**

- Directly reflects model’s predictive uncertainty.
- Task-agnostic measure of fluency.

**Limitations:**

- Not aligned with human judgments for downstream tasks.
- Cannot compare models with different vocabularies/tokenization schemes.

---

## **2. Accuracy**

**What it measures:**

- % of predictions that are correct.
  $$\text{accuracy} = \frac{\text{num correct predictions}}{\text{num total predictions}}$$
- Flaw:\_ Misleading if classes are **unbalanced** (e.g., a spam detector that always predicts "not spam" might be 90% accurate but useless).

**Appropriate for:**

- **Classification** tasks with balanced labels:
  - POS tagging (when balanced), sentiment analysis, NLI.
- Good for single-label prediction

**Why:**

- Simple metric when classes are roughly balanced and task is closed-form.
- Interpretable, task-aligned
- Clear, stable signal during model development
- Easy to compare across models using same datasets and labels

**Limitations:**

- Misleading for **imbalanced classes** (e.g., rare NER labels, spam detection).
  - If 95% of emails are non-spam, a model predicting "not-spam" always gets 95% accuracy but is useless
- Conceals type-specific errors:
- Cannot express partial credit
  - Predicting the righ category but wrong subcalss recieves zero

---

## **3. Precision, Recall, F-measure**

### **Precision**

**Meaning:** Of all the items the system _predicted as positive_, how many were **actually correct**?  
**Formula:**

$\text{Precision} = \frac{TP}{TP + FP}$

**When it matters:**

- **False positives are costly.**
- You prefer **conservatism** over overgeneration.

**Examples:**

- **NER:** Avoid labelling non-entities as entities.
- **Toxic content detection:** Don’t wrongly flag harmless text.

---

### **Recall**

**Meaning:** Of all the _true_ positives that exist, how many did the system **actually find**?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**When it matters:**

- **Missing items is costly.**
- You prefer **coverage** over precision.

**Examples:**

- **Coreference resolution:** Missing links breaks downstream tasks.
- **Information extraction:** Better to capture all mentions for analysis.

---

### **F1-score (F-measure)**

**Meaning:** The **harmonic mean** of precision and recall.

$$F_1 = 2 \cdot \frac{PR}{P + R}$$

**Why harmonic mean?**

- It **penalises imbalance** (e.g., high precision but terrible recall).
- Encourages a model that is **jointly good** at both.

**When it shines:**

- **Sparse labels** (e.g., NER entities are rare).
- **Imbalanced classes** (many negatives, few positives).
- **Token-level structure prediction** where TP/FP/FN matter more than TN.

### **Macro vs Micro Averaging**

- **Macro averaging:** average the metric across classes/examples.
- **Micro averaging:** sum the metric across classes/examples and divide by the total number of classes/examples.

Normal _F1-Score_ is micro averaging.

Macro-averaging is $F1_{macro} = \frac{F1_1 + F1_2 + ... + F1_n}{n}$

**When to use:**

- **Macro averaging:** when the classes are unbalanced.
- **Micro averaging:** when the classes are balanced.

---

# **4. Metrics for Generative Tasks**

## **BLEU (Machine Translation)**

**What BLEU measures**

- **N-gram precision**: how many n-grams in the candidate also appear in the reference(s).
- **Geometric mean** of 1-gram … 4-gram precisions.
- **Brevity Penalty (BP)**: reduces score if the system output is **too short**.

**Why appropriate**

- Captures **local phrase correctness** (short, frequent MT errors).
- Historically correlated **moderately** with human translation quality.
- Works well when translations are **literal** and n-gram overlap is meaningful.

**Limitations**

- **Surface-form bias**: penalises valid paraphrases that differ lexically
- Weak on **semantics**, **discourse**, **gender agreement**, **style**.
- High BLEU ≠ good translation if system “games” n-gram overlap.

---

## **ROUGE (Summarisation)**

**What ROUGE measures**

- **ROUGE-N:** recall of n-gram overlap
- **ROUGE-L:** **Longest Common Subsequence (LCS)** — measures shared ordering.
- Recall-oriented because we care about whether summaries **cover important content**.

**Why appropriate**

- Summaries must capture **key information**; ROUGE focuses on whether important words/phrases were retrieved.
- Historically correlated with **human judgments** for extractive summaries.

**Limitations**

- Insensitive to **coherence**, **fluency**, **logical structure**.
- Cannot detect **hallucinated content** not present in the source.
- Overly rewards **extractive copying**, under-rewards paraphrasing.

---

## **Other Metrics**

### **METEOR**

- Aligns words using **synonyms**, **stemming**, **paraphrase tables**.
- Better recall–precision balance than BLEU.

### **BERTScore**

- Computes similarity using contextual embeddings.
- Captures **semantic similarity**, not just surface overlap.

### **chrF**

- Character-level F-score.
- Great for **morphologically rich languages** (Slavic, Finnish, Turkish).
- More robust to tokenisation artefacts.

---

## **5. LLM-as-a-Judge**

**What this evaluates**

- An LLM scores or ranks system outputs using **task-specific rubrics** (e.g., correctness, fluency, helpfulness).
- Used for: **summaries**, **QA**, **dialogue**, **code correctness**, **reasoning tasks**.

**Why appropriate**

- Models can evaluate **meaning**, **style**, **faithfulness**, **reasoning steps** — dimensions overlap metrics cannot capture.
- **Scalable**, **cheap**, **fast** → useful for large evaluations (thousands of samples).
- Closer to human preference judgments than BLEU/ROUGE.

**Limitations / Concerns**

- **Bias:** judges prefer outputs that match their own stylistic priors.
- **Non-transparency:** unclear internal criteria.
- **Reward hacking:** systems may optimise for judge quirks, not true quality.
- Requires **calibration**, comparisons to **human-annotated** sets, and ideally **multi-judge** ensembles.

---

## **6. Win Rate & Elo Ranking**

### **Win Rate**

**Definition**

- Given two model outputs (A vs B), a judge (human or LLM) chooses the better one.
- **Win rate = P(A beats B)** across many samples.

**Why useful**

- Simple, robust for tasks with **no single correct answer** (dialogue, creative tasks).
- Directly measures **preference** rather than accuracy.

---

### **Elo Ranking**

**Definition**

- Converts pairwise preferences into a **global skill score** for multiple models.
- Analogous to **chess ratings**: each match updates both players’ Elo.

**Why appropriate**

- Produces a **stable global ordering** of many models.
- Handles **non-absolute quality** (no gold truth).
- More expressive than raw win rate.

**Limitations**

- Assumes approximate **transitivity** (if A > B and B > C → A > C).
- Sensitive to judge biases, sampling strategy, and match difficulty.
- No absolute meaning: Elo is **relative** to the pool of models tested.

---

# **7. Intrinsic vs Extrinsic Evaluation**

## **Intrinsic Evaluation**

**Definition**  
Evaluate a model **component** directly, outside any downstream task.

**Examples**

- **Perplexity** on a corpus (language modelling).
- **Word similarity** tasks for embeddings (SimLex, WordSim).
- **Parsing accuracy** (LAS/UAS) on annotated treebanks.

**Pros**

- Fast, cheap, diagnostic.
- Allows controlled experiments (change one component at a time).

**Cons**

- **Weak correlation** with end-task performance.
- Encourages optimisation for metrics that do not reflect **real utility**.

---

## **Extrinsic Evaluation**

**Definition**  
Evaluate a model by how well it improves **downstream task performance**.

**Examples**

- Better embeddings → higher **NER F1**.
- Improved LM → stronger **QA** or **MT** accuracy.
- Stronger parser → better **information extraction**.

**Pros**

- Measures **actual task usefulness**.
- Captures interactions between components.

**Cons**

- Expensive: requires full pipeline training.
- Hard to interpret: performance changes may come from multiple factors.
- Noisy due to hyperparameters and system design.

---

# **8. Corpora: Collection, Annotation, Distribution Issues**

## **Collection Issues**

- **Sampling bias:**
  - Source skews (news vs Twitter vs Wikipedia) → changes style, dialects, topics.
  - Leads to models that generalise poorly to real-world data.
- **Privacy & consent:**
  - Risk of collecting PII, medical content, minors' data.
  - GDPR and global privacy regimes impose strict constraints.
- **Domain mismatch:**
  - Training on one domain (news) but testing on another (medical) → large performance drop.

---

## **Annotation Issues**

- **Inter-annotator agreement (IAA):**
  - Measures reliability; use κ (kappa) or α (alpha).
  - Low IAA → task fundamentally ambiguous or guidelines unclear.
    - Noisy ground truth, test set evaluation is misleading
- **Quality control:**
  - Annotator training, gold-check questions, adjudication by experts.
- **Cost:**
  - High-quality annotation (e.g., NER, SRL, discourse) requires expertise.
- **Bias:**
  - Annotators bring cultural assumptions → impacts sentiment, toxicity, emotion labels.

---

## **Distribution Issues**

- **Licensing constraints:**
  - Some corpora cannot be redistributed (copyright).
  - Limits reproducibility.
- **Data documentation:**
  - Datasheets / model cards → ensure transparency about collection, cleaning, biases.
- **Legal risk:**
  - Copyright violation, defamation risk, GDPR obligations.
- **Fairness considerations:**
  - Underrepresentation of minority dialects → models fail for underrepresented users.
  - Reinforces socio-linguistic inequalities.

---

# **TLDR**

- **Perplexity** → LM predictive quality.
- **Accuracy** → overall correctness (balanced classes).
- **Precision/Recall/F1** → imbalanced tasks, extraction tasks.
- **BLEU/ROUGE** → generative tasks, based on n-gram overlap.
- **LLM-as-judge** → semantic & quality eval, but biased.
- **Win rate & Elo** → preference-based ranking of LLMs.
- **Intrinsic vs extrinsic** → component-only vs task-based evaluation.
- **Corpora issues** → bias, consent, privacy, annotation quality, licensing.

# 12. Ethical Issues

# **Ethical Issues in NLP — Revision Notes**

---

## **1. Algorithmic Bias**

**Definition:**  
Systematic, unfair performance differences across demographic groups due to **biased data**, **biased models**, or **unequal error rates**.

### **Sources of bias**

- **Training data bias:**
  - Over-representation of Standard American English → poor performance on dialects.
  - Toxicity classifiers mislabel AAVE sentences as “offensive.”
- **Label bias:**
  - Annotators bring cultural assumptions → sentiment or hate-speech datasets skewed.
- **Measurement bias:**
  - Metrics that fail to capture performance differences (accuracy hides minority errors).

### **Implications for tasks**

- MT may produce gender-stereotyped translations (_doctor → he_, _nurse → she_).
- Coreference systems may misresolve pronouns for unrepresented groups.
- Speech or text models may fail on dialects or low-resource languages.

### **Mitigation**

- Diverse data sampling; bias audits; counterfactual data augmentation.
- Group-specific evaluation metrics (per-group F1).
- Explicit fairness constraints during training.

---

## **2. Direct vs Indirect Discrimination**

### **Direct discrimination**

**Definition:**  
Model _explicitly_ uses a protected attribute (e.g., gender, race) to make decisions.

**Examples:**

- Sentiment classifier that assigns lower positivity to names associated with specific ethnic groups.
- Hiring model penalising applicants with “female” as a feature.

**Mitigation:**

- Remove protected attributes; enforce feature-dropout; independent bias checks.

---

### **Indirect discrimination (a.k.a. disparate impact)**

**Definition:**  
A model uses **proxy features** that correlate with protected attributes, even without explicit reference.

**Examples:**

- ZIP code predicting socioeconomic status or ethnicity.
- Word embeddings encoding stereotypes captured from biased corpora.
- MT systems assigning stereotyped gender pronouns due to corpus bias.

**Mitigation:**

- Detect and reduce proxy correlations; adversarial training; fairness-aware loss functions.
- Redesign features to remove protected-attribute leakage.

---

## **3. Representational vs Allocational Harm**

### **A. Representational Harm**

**Definition:**  
When a system **reinforces negative stereotypes**, erases identities, or misrepresents groups.

**Examples:**

- Associating Muslim names with terrorism in word embeddings.
- Autocomplete suggesting harmful or biased continuations.
- MT translating gender-neutral forms into stereotyped gender roles.

**Relevance:**  
Affects perception, identity, and social narratives — even when no resource allocation is involved.

**Mitigation:**

- Debiasing embeddings; curated safe-text filters; red-team evaluations.
- Inclusive dataset design; culturally aware annotation protocols.

---

### **B. Allocational Harm**

**Definition:**  
When a system causes **unequal access to opportunities, services, or material resources**.

**Examples:**

- Credit scoring models assigning lower credit limits to speakers of certain dialects.
- Automated hiring tools privileging certain linguistic styles or backgrounds.
- Health-care NLP triage system misclassifying symptoms for minority groups.

**Relevance:**  
Material consequences → financial, medical, educational disparities.

**Mitigation:**

- Fairness constraints; group fairness metrics; post-hoc calibration.
- Regulatory oversight; transparency documentation; algorithmic audits.

---

# **4. How to use these concepts in exam scenarios**

You may be asked to:

- Analyse a dataset/model pipeline and identify **types of harm**.
- Explain whether an issue is **representational** or **allocational** (or both).
- Describe fairness risks for an example task (e.g., MT for public-service signs).
- Suggest **mitigations** appropriate to the task and resource type.
- Connect ethical risks to **dataset issues** (consent, bias, documentation) and **evaluation** issues (per-group metrics).

---

# **TLDR**

- **Algorithmic bias** = unequal model behaviour due to skewed data, labels, or training.
- **Direct discrimination** = explicit use of protected attributes.
- **Indirect discrimination** = proxy features cause unequal impact.
- **Representational harm** = stereotypes & misrepresentation.
- **Allocational harm** = unequal access to resources & opportunities.
- **Mitigate with:** better data, fairness metrics, audits, debiasing, documentation, and inclusive design.

# 13. Multilingual

# **Multilingual NLP — Revision Notes**

---

## **1. Data Paucity**

**Core idea:**  
Most of the world’s languages have **little or no labeled data**, and many have limited unlabeled corpora.

### **Why it matters**

- Many NLP models assume **large-scale corpora**, which low-resource languages lack.
- Leads to **poor performance**, unstable training, and biased multilingual systems.

### **Typical problems**

- Sparse morphology → harder to learn inflectional forms.
- High dialectal variation with small datasets.
- Non-standard or no orthography.

### **Relevance to tasks**

- MT, POS tagging, NER, ASR — all suffer when training data is small.
- Encourages methods like **transfer learning**, **cross-lingual embeddings**, **few-shot learning**.

---

## **2. Multilingual LLMs**

**Definition:**  
A single model jointly trained on text from **many languages**, sharing parameters and often a shared subword vocabulary (BPE/SentencePiece).

### **Advantages**

- **Parameter sharing** enables cross-lingual transfer.
- Low-resource languages benefit from **shared semantics** captured via high-resource languages.
- Models can perform **zero-shot** generation or classification.

### **Challenges**

- **Vocabulary imbalance**: high-resource languages dominate the subword vocabulary.
- **Interference**: languages compete for model capacity.
- **Script issues**: different scripts create uneven coverage.

### **Examples**

- mBERT, XLM-R, BLOOM, multilingual GPT-class models.

### **Relevant tasks**

- MT, cross-lingual QA, NER, document classification, embedding similarity.

---

## **3. Zero-shot Cross-linguistic Transfer**

**Definition:**  
Model trained on a task in one language (e.g., English) performs the same task **in an unseen language** without additional training.

### **Why it works**

- Shared embeddings and shared model layers build **language-neutral representations**.
- Based on the **distributional hypothesis across languages**.

### **When it works well**

- Languages with **similar scripts** and **similar syntactic structures**.
- Tasks relying on **semantics** more than fine-grained morphology.

### **When it fails**

- Languages structurally distant (e.g., English → Inuktitut).
- Rich morphology with low training support.

---

## **4. Translate-Train & Translate-Test**

### **Translate Train**

**Process:**

1. Translate training data from a high-resource language (HRL) → low-resource language (LRL).
2. Train model directly on synthetic LRL data.

**Pros:**

- Produces an LRL-specific model.
- Good when MT into LRL is accurate.

**Cons:**

- Translation errors propagate into training.
- Expensive to generate large synthetic corpora.

---

### **Translate Test**

**Process:**

1. Test input in LRL is translated → HRL.
2. Apply HRL-trained model.
3. Optionally translate output back.

**Pros:**

- No need to train a dedicated LRL model.
- Works well if MT is good from LRL → HRL.

**Cons:**

- MT errors during inference cause evaluation noise.
- Semantic drift is common.

---

### **Zero-shot vs Translate-Train/Test — Summary**

- **Zero-shot:** no translation; relies fully on model’s internal shared representations.
- **Translate-train:** creates training data in the target language.
- **Translate-test:** avoids retraining; uses translation pipeline at inference.

---

## **5. Multilingual Evaluation**

### **Challenges**

- **Benchmark imbalance:** HRLs have high-quality datasets; LRLs often do not.
- **Cultural + domain mismatch:** evaluation content may not transfer across languages well.
- **Script diversity:** tokenization and vocabulary create uneven performance baselines.

### **Common evaluation strategies**

- **Multilingual benchmarks:** XTREME, XGLUE, AmericasNLI.
- **Per-language breakdowns:** accuracy/F1 reported for each language.
- **Transfer tests:** train on HRL, test on LRL.
- **Code-switching performance:** test robustness in mixed-language input.

### **What evaluators look for**

- Stability across languages.
- Whether performance drops correlate with data size (often they do).
- Bias against minority dialects or scripts.

---

# **TLDR**

- **Data paucity** drives the challenge: most languages have minimal training data.
- **Multilingual LLMs** share parameters across languages, enabling **transfer** but risking interference.
- **Zero-shot transfer** uses shared semantics; good when languages are similar.
- **Translate-train** adds synthetic LRL training data; **translate-test** uses translation at inference.
- **Multilingual evaluation** must compare performance _per language_, account for script bias, and use multilingual benchmarks.

# 2. Models

## 🔁 Big Picture: What You Must Do For _Any_ Model

For **each model** below, you should be able to:

- **Compute**
  - **P(sequence)** or **one forward step** (e.g. next-word prob, class prob, hidden state update).
  - For **linear / log-linear models**: dot product + nonlinearity (sigmoid/softmax).
- **Count parameters**
  - **Exactly** for small models (given vocab sizes, hidden sizes, etc.).
  - **Approximately** for LLMs (e.g. embeddings + layers × per-layer params).
- **Explain training**
  - What **data** is used, what **objective** (loss) is optimised, and roughly how **gradient descent** updates parameters.
- **Explain smoothing / regularisation**
  - Why it is needed (overfitting / zeros), and when it matters most.
- **Map to tasks**
  - Given a model, say which **task types** it’s suited for (classification / seq2seq / language modelling / representation learning) and **how** it’s applied.
- **Analyse what the model can/can’t capture**
  - What linguistic phenomena it can represent; where it fails (e.g. long-distance deps, ambiguity).
- **Compare pros/cons**
  - For a given task, explain why you might pick one model over another.

Keep that meta-template in mind while revising each model.

---

## 1️⃣ N-gram Models

**Core idea**

- **N-gram assumption**:  
   $$P(w_i \mid w_1,\dots,w_{i-1}) \approx P(w_i \mid w_{i-N+1},\dots,w_{i-1})$$.  
   Only **last N−1 words** matter (Markov assumption).

**Compute probability / forward step**

- Sentence probability:  
   $P(w_1^T) = \prod_{i=1}^T P(w_i \mid w_{i-N+1}^{i-1})$.
- You should be able to:
  - Use **counts**: $P(w_i \mid h) = \dfrac{C(h,w_i)}{C(h)}$ (MLE).
  - Apply **smoothing** (e.g. add-α, backoff) when counts are zero.

**Parameters**

- Number of parameters ≈ **number of distinct N-grams** with non-zero probability.
- Upper bound: $|V|^N$ parameters (huge for large N).
- You should be able to count params for **small V, small N** exactly.

**Training**

- **Data**: Text corpus (tokens).
- **Objective**: Maximise **likelihood** (or minimise **cross-entropy**) of training data.
- **Procedure**: Count N-grams → compute probabilities (maybe with smoothing).

**Smoothing**

- Needed because many N-grams **never appear** in training → zero probability.
- **Most important** when:
  - Data is **small** or
  - Vocab is **large** or
  - N is **big** (e.g. tri-/4-grams).
- Examples: **Add-α**, **Kneser–Ney**, **backoff**, **interpolation**.

**Typical tasks**

- **Language modelling** (next-word prediction, perplexity).
- Baseline for **speech recognition**, **MT**, **spell correction**, etc.

**What it can / can’t capture**

- **Can**:
  - Local word patterns, collocations, short-range dependencies.
- **Fails at**:
  - **Long-distance dependencies** (“if … then …”).
  - **Global semantics**, **discourse**, **world knowledge**.
  - Generalising to unseen contexts beyond smoothed interpolation.

**Pros / cons**

- **Pros**: Simple, interpretable, fast for small N, easy to compute.
- **Cons**: Data-hungry, sparse, poor at long-range syntax/semantics, doesn’t share parameters across similar contexts.

**Extra exam point (generative process)**

- **Generative story**:
  1. Pick first token(s) from start-of-sentence distribution.
  2. For each position (i), **sample $(w_i)$** from $P(w_i \mid w_{i-N+1}^{i-1})$.
  3. Continue until EOS token.
- Joint probability of a **sequence** (no latent vars in basic N-gram):  
   $P(w_1^T) = \prod_i P(w_i \mid w_{i-N+1}^{i-1})$.
- If the question mentions **latent variables**, you might be expected to **write a generic form**  
   $P(x,z) = P(z)P(x\mid z)$ and relate to any extended N-gram variant discussed (e.g. tags).

---

## 2️⃣ Logistic Regression (Binary)

**Core idea**

- **Linear classifier** with **sigmoid** output:  
   $P(y=1 \mid x) = \sigma(w^\top x + b)$.
- Used for **binary classification** (e.g. positive vs negative).

**Compute step**

- Given **weights**, **bias**, **feature vector**, compute:
  - **Score**: $s = w^\top x + b$.
  - **Probability**: $P(y=1 \mid x) = \dfrac{1}{1+e^{-s}}$.
- In exam: you may **not** have to compute $e^s$, but must:
  - **Set up** the expression
  - Say which class has **higher probability** based on comparing scores.

**Parameters**

- For input with **d features**: **d weights + 1 bias**.
- If you treat it as logistic for $y∈{0,1}$, that’s it.

**Training**

- **Data**: pairs $(x^{(i)}, y^{(i)})$.
- **Objective**: maximise **log-likelihood** (equiv. minimise **cross-entropy loss**).
- **Training**: gradient descent / variants, with **regularisation** (L2, etc.).

**Regularisation**

- To prevent **overfitting**, especially with **many features** (e.g. bag-of-words).
- Common: **L2** penalty $\lambda |w|^2$.
- Most important when number of features ≫ number of examples.

**Typical tasks**

- Binary **sentiment analysis** (pos/neg).
- Binary **spam detection**.
- **Any** yes/no classification where features are interpretable.

**What it can / can’t capture**

- **Can**: linear decision boundaries in feature space; works well with **good features**.
- **Cannot**: model **interactions** or **nonlinear** relations unless features encode them; no sequence structure.

**Pros / cons**

- **Pros**: Interpretable, convex training objective, relatively easy.
- **Cons**: Limited expressivity; relies heavily on **feature engineering**.

---

## 3️⃣ Multinomial Logistic Regression (Softmax Regression)

**Core idea**

- Generalises logistic regression to **K > 2** classes with **softmax**:  
   $P(y=k \mid x) = \dfrac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)}$.

**Compute step**

- Given features ($x$) and class weight vectors ($w_k$):
  - Compute **scores** $s_k = w_k^\top x + b_k$.
  - Compute **softmax probabilities** using those scores.

**Parameters**

- For **d features** and **K classes**:
  - Weights: $K \times d$, biases: $K$.
  - Total params: $K \cdot d + K$.

**Training & regularisation**

- Same story as logistic, but with **multiclass cross-entropy loss**.
- Regularisation again important when many features.

**Tasks & features**

- **POS tagging per token** (if you treat each token independently).
- **Topic classification**, **intent classification**, etc.
- Features: bag-of-words, n-gram counts, lexical features, etc.

**Extra exam requirements**

- **You must be able to**:
  - **Write the softmax formula** clearly.
  - Given weights/features, **identify most probable class** (highest score or highest unnormalised logit).
  - Reason about how **changing a weight** would affect class probabilities.

---

## 4️⃣ Skip-gram with Negative Sampling (Word2Vec)

**Core idea**

- A **neural model** to learn **word embeddings** by predicting **context words** from a **target word** (Skip-gram).
- Negative sampling approximates full softmax by contrasting **true context words** vs **sampled negatives**.

**Architecture**

- **Input**: one-hot word → **embedding lookup** (vector $v_w$).
- **Output**: separate **context embedding** $u_c$.
- **Scoring**: $u_c^\top v_w$ → passed through sigmoid for positive vs negative pairs.

**Compute step**

- Given **target embedding** $v_w$ and **context embedding** $u_c$:
  - Score: $s = u_c^\top v_w$.
  - Positive pair probability (binary logistic): $\sigma(s)$.

**Parameters**

- Two embedding matrices (often):
  - **Input embeddings**: $|V| \times d$.
  - **Output/context embeddings**: $|V| \times d$.
- Total: $≈ 2|V|d$ params.

**Training**

- **Data**: text → pairs (target, context) within a window.
- **Objective**: for each positive pair:
  - maximise $\log \sigma(u_c^\top v_t) + \sum_{neg} \log \sigma(-u_{n}^\top v_t)$.
- Uses **stochastic gradient descent**.

**Regularisation**

- Mainly **implicit** via:
  - Limited embedding size
  - Negative sampling distribution.
- You can also use **L2** on embeddings, but often not emphasised.

**Tasks**

- Not a task model per se; it **learns representations** for use in:
  - Downstream classifiers (sentiment, NER, etc.).
  - Similarity tasks, analogies, nearest neighbours.

**What it can / can’t capture**

- Captures **distributional semantics** (“you know a word by the company it keeps”).
- Fails at:
  - Sentence/sequence structure; it’s bag-of-context windows.
  - Word sense disambiguation (single vector per type).

**Pros / cons**

- **Pros**: Fast, effective embeddings, simple.
- **Cons**: No contextualisation, static word meaning, limited to co-occurrence.

---

## 5️⃣ Multilayer Perceptron (Feed-forward Network)

**Core idea**

- Fully connected **layers** with **nonlinear activations** (e.g. ReLU, tanh):
  - $h = f(Wx + b)$,
  - $y = g(Uh + c)$ (for classification, g=softmax/sigmoid).

**Compute step**

- Given small dimensions, you should be able to:
  - Multiply input by weight matrix, add bias.
  - Apply nonlinearity (ReLU, tanh).
  - Apply final linear + softmax for class probs.

**Parameters**

- For each layer with **input dim in**, **hidden dim out**:
  - Weights: in × out, biases: out.
- Total params = **sum over layers**.

**Training & regularisation**

- **Objective**: cross-entropy for classification; MSE or others for regression.
- **Training**: backpropagation + gradient descent.
- Regularisation: **L2**, **dropout**, **early stopping**, etc.

**Tasks**

- **Classification** from fixed-size features (e.g. sentence embeddings → sentiment).
- **Regression** tasks (e.g. scoring).

**What it can / can’t capture**

- **Can**: complex **nonlinear mappings** from features to labels.
- **Cannot**: handle **variable-length sequences** directly (unless you summarise first); no explicit temporal structure.

**Pros / cons**

- **Pros**: flexible, universal approximator.
- **Cons**: needs good features, no sequence inductive bias.

---

## 6️⃣ Recurrent Neural Network (RNN)

**Core idea**

- Processes **sequences** step-by-step, maintaining a **hidden state** $h_t$:  
   $h_t = f(W_x x_t + W_h h_{t-1} + b)$.

**Compute step**

- Given small dimensions, you must be able to:
  - Start with $h_0$ (often zeros).
  - Compute $h_1$ from $x_1, h_0$;
  - Then $h_2$, etc.
  - Optionally compute output $y_t = g(W_y h_t + c)$ (e.g. softmax over vocab).

**Parameters**

- For vanilla RNN:
  - Input→hidden: $W_x$ (dim: $d_{in} \times d_h$).
  - Hidden→hidden: $W_h$ ($d_h \times d_h$).
  - Hidden→output: $W_y$ ($d_h \times d_{out}$).
  - Plus biases.
- Total = sum of these matrices + biases.

**Training & regularisation**

- **Objective**: e.g. next-token cross-entropy for LM.
- **Training**: backpropagation through time (BPTT).
- Regularisation: dropout on hidden states, gradient clipping, etc.

**Tasks**

- **Language modelling**, sequence classification, tagging, simple MT (with encoder-decoder RNN variants).

**What it can / can’t capture**

- **Can**: some **longer context** than N-grams, sequential patterns.
- **Cannot (well)**: very long dependencies (vanishing/exploding gradients); parallelisation is hard.

**Pros / cons**

- **Pros**: sequence-aware, more expressive than N-grams.
- **Cons**: training difficulties, slower than Transformers for long sequences.

---

## 7️⃣ RNN with Attention (Seq2Seq + Attention)

**Core idea**

- **Encoder RNN** reads input → sequence of hidden states.
- **Decoder RNN** generates output, at each step **attending** to encoder states via attention mechanism.

**Attention computation**

- For decoder state $s_t$ and encoder states $h_1,\dots,h_T$:
  - Scores: $e_{t,i} = \text{score}(s_t, h_i)$ (dot, MLP, etc.).
  - Weights: $\alpha_{t,i} = \text{softmax}_i(e_{t,i})$.
  - Context: $c_t = \sum_i \alpha_{t,i} h_i$.
  - Decoder uses $[s_t ; c_t]$ to predict next token.

**Parameters**

- Encoder RNN params + decoder RNN params + **attention parameters** (for score function).
- You should be able to count these for a small example.

**Tasks**

- **Seq2seq**: machine translation, summarisation, etc.

**What it can / can’t capture**

- **Can**: explicitly model **alignments**, focus on specific input tokens.
- **Still limited by**: sequential processing, training stability, long sequences vs Transformer.

---

Gotcha — let’s clean this up so the LaTeX is actually readable.

I’ll redo the **Transformer notes with correct MathJax**, keeping:

- Your **structure** (8️⃣ / 9️⃣ / 🔟, A–F sections, etc.)
- The **worked examples** in the style you showed
- All the **math properly typeset**

---

# 8️⃣ **Transformer Encoder-Only (BERT-style)**

---

## **Core idea**

- Stack of **self-attention + feedforward** blocks with **residual connections** and **LayerNorm**.
- Processes the whole sequence **in parallel** → outputs **contextualised representations** for all tokens.

---

## **A. Computational / Forward Step**

Let input be a matrix
$X \in \mathbb{R}^{N \times d}$ (N tokens, model dimension $d$).

### 1. Linear projections

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V
$$

where

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}.
$$

### 2. Self-attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Call the result $A \in \mathbb{R}^{N \times d_k}$

### 2.5 Output Projection

$$A_{proj} = AW_O$$
Where $A_{proj} \in \mathbb{R}^{N \times d}$ and $W_O \in \mathbb{R}^{d_k \times d}$

Each head has a dimension $d_k$ which $= \frac{d}{h}$ with $h$ heads, the concatenated output is $hd_k = d$. $W_O$ mixes head outputs back to the correct shape to be able to have layerNorm + Residuals applied.

### 3. Residual + LayerNorm (attention block)

$$
X' = \text{LayerNorm}(X + A_{proj}).
$$

### 4. Feed-forward network (FFN, per token)

$$
H = \phi(X' W_1 + b_1) W_2 + b_2,
$$

where

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$,
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$.

### 5. Residual + LayerNorm (FFN block)

$$
X_{\text{out}} = \text{LayerNorm}(X' + H).
$$

You should be able to do this **for tiny toy shapes** (e.g. $N=2, d=2, d_k=2$).

---

## **B. Parameter Counting (Per Layer)**

Let:

- model dim: $d$
- head dim: $d_k$
- FFN inner dim: $d_{\text{ff}}$
- single head for simplicity.

### Attention projections

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k},\quad
W_O \in \mathbb{R}^{d_k \times d}.
$$

Number of **attention weights**:

$$
3 \cdot d \cdot d_k + d_k \cdot d = 4 d d_k.
$$

### Feed-forward network

$$
W_1 \in \mathbb{R}^{d \times d_{\text{ff}}},\quad
W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}.
$$

Number of **FFN weights**:

$$
d \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d
= 2 d d_{\text{ff}}.
$$

Biases:

- $b_1 \in \mathbb{R}^{d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d}$ → add $d_{\text{ff}} + d$ if needed.

### LayerNorm

Two LayerNorms per layer, each with scale and bias:

- each LayerNorm: $2d$ params
- per layer: $4d$.

### Embeddings (global, not per-layer)

- Token embeddings: $|V| \times d$.
- Positional embeddings (if learned): $N_{\max} \times d$.

---

## **C. Training + Regularisation**

### Training (BERT-style)

- **Masked Language Modelling (MLM)**:
  mask some tokens and predict them from **bidirectional context**.
- Sometimes **Next Sentence Prediction** (NSP) objective.

Optimisation:

- Cross-entropy loss on masked positions.
- Backprop through the whole network.
- Optimiser: some variant of SGD (e.g. AdamW).

### Regularisation

- **Dropout** on attention outputs and FFN outputs.
- **LayerNorm** for stabilising activations.
- **Weight decay** (L2-style) on parameters.

---

## **D. Typical Tasks**

- **Sentence/document classification** (use [CLS] token).
- **Token classification** (NER, POS, chunking).
- **Span-based QA**.
- **NLI, paraphrase detection, similarity**.

All framed as **“encode input → add small head → predict labels”**.

---

## **E. What It Can / Can’t Capture**

### Can

- **Bidirectional context** for each token.
- **Long-distance dependencies** via global self-attention.
- Rich semantic representations.

### Can’t

- Autoregressive generation (without adding a generative head / procedure).
- Very long sequences efficiently (attention is $O(N^2)$ in sequence length).

### Typical failure modes

- On very long sequences, attention may become **diffuse**.
- Positional encodings trained for max length $L$ may **not extrapolate** beyond $L$.
- Still struggles with **deep hierarchical syntax** (e.g. many nested clauses).

---

## **F. Pros / Cons**

**Pros**

- Excellent for **understanding** tasks.
- Parallel over tokens (good GPU utilisation).
- Powerful and expressive.

**Cons**

- Quadratic time/memory in sequence length.
- Not naturally generative.
- Pretraining is compute- and data-heavy.

---

## **G. Positional Encodings (Exam-Specific)**

You should know:

### Absolute positional encodings

- **Sinusoidal** (fixed):
  deterministic functions of the position $p$ and dimension index $i$.
- **Learned absolute** (BERT):
  trainable embedding $P_p \in \mathbb{R}^d$ added to token embeddings.

### Relative positional encodings

- Encode **offset** between positions (e.g. $j-i$), not their absolute indices.
- Often implemented via **added biases** to attention scores or shifted embeddings.
- Help generalise to **different sequence lengths** and capture relative order.

---

## **H. Scaling Laws (Exam-Specific)**

High-level ideas:

- **Parameter scaling**:
  For $L$ layers, model dimension $d$, feedforward dim $d_{\text{ff}}$,

  $$
  \text{params} \approx L \cdot (c_1 d^2 + c_2 d d_{\text{ff}})
  $$

  (for constants $c_1, c_2$ depending on heads etc.).

- **Performance scaling**:
  Empirically, loss often follows a **power law** in **data size**, **model size**, **compute**.

- Doubling $d$ tends to **quadruple** param count; doubling $L$ increases linearly.

You don’t need exact exponents, just this **qualitative scaling story**.

---

## **I. Interpreting Attention Weights (Exam-Specific)**

Possible interpretations:

- Some heads focus on **syntactic relations** (e.g. subject ↔ verb).
- Others track **coreference** (pronouns ↔ antecedents).
- Some track **relative positions** or punctuation.

Caveats:

- Attention weights do **not necessarily equal explanations**.
- Many heads are **diffuse** or encode information that’s not human-interpretable.

---

## **Worked Example 1 — Self-Attention (One Query, Full Weights)**

This is exactly in the style you gave.

We take 3 tokens, dim = 2, and assume projections already done:

$$
Q = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix},\quad
K = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix},\quad
V = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}.
$$

We compute attention for **token 1**, with query
$q_1 = [1, 0]$.

### Scores

$$
e_i = q_1 \cdot k_i
$$

- $e_1 = [1,0] \cdot [1,0] = 1$
- $e_2 = [1,0] \cdot [0,1] = 0$
- $e_3 = [1,0] \cdot [1,1] = 1$

So the score vector is:

$$
z = [1, 0, 1].
$$

### Softmax

$$
\alpha_i = \frac{\exp(z_i)}{\exp(1) + \exp(0) + \exp(1)}
= \frac{\exp(z_i)}{2e + 1}.
$$

So:

$$
\alpha_1 = \alpha_3 = \frac{e}{2e + 1},\quad
\alpha_2 = \frac{1}{2e + 1}.
$$

### Attended vector

$$
\text{attn}(q_1) = \sum_i \alpha_i v_i
= \alpha_1 [1,0] + \alpha_2 [0,1] + \alpha_3 [1,1].
$$

Compute component-wise:

$$
\text{attn}(q_1)
= \begin{bmatrix}
\alpha_1 + \alpha_3 \\
\alpha_2 + \alpha_3
\end{bmatrix}
= \begin{bmatrix}
\frac{2e}{2e+1} \\
\frac{1 + e}{2e+1}
\end{bmatrix}.
$$

No need for numeric approximation — the **structure** is what matters.

---

## **Worked Example 2 — Encoder Param Count (1 Layer)**

Exactly in your requested style.

> **Q:** Approximate parameter count for a 1-layer encoder with
> • vocab size $V = 10$
> • model dim $d_{\text{model}} = 4$
> • feed-forward dim $d_{\text{ff}} = 8$
> • 1 attention head

### Embedding layer

Token embeddings:

$$
V \times d_{\text{model}} = 10 \times 4 = 40.
$$

(We’ll ignore positional embeddings here.)

### Self-attention (single head)

- $W_Q: 4 \times 4 = 16$
- $W_K: 4 \times 4 = 16$
- $W_V: 4 \times 4 = 16$

So

$$
3 \times 16 = 48.
$$

Output projection:

$$
W_O: 4 \times 4 = 16.
$$

Total attention weights:

$$
48 + 16 = 64.
$$

### Feed-forward

- $W_1: 4 \times 8 = 32$
- $W_2: 8 \times 4 = 32$

Total FFN:

$$
32 + 32 = 64.
$$

### Total (ignoring biases & LayerNorm)

$$
40\ (\text{emb}) + 64\ (\text{attn}) + 64\ (\text{FFN}) = 168\ \text{parameters}.
$$

---

Below are **fully expanded, encoder-level versions** of both **decoder-only** and **encoder–decoder** transformers.
Format, depth, and sectioning **exactly match** your encoder template.

---

# 9️⃣ **Transformer Decoder-Only (GPT-style)**

(Updated to be as detailed as encoder)

---

# **Core idea**

- Same building blocks as encoder: **self-attention + FFN**, each wrapped with **residuals + LayerNorm**.
- BUT with a **causal mask** so token _t_ only attends to tokens ≤ _t_.
- Trained on **autoregressive LM objective**:

$$
P(x) = \prod_{t=1}^N P(x_t \mid x_{<t}).
$$

---

# **A. Computational / Forward Step**

Input sequence (with positions added):

$$
X \in \mathbb{R}^{N \times d}.
$$

---

## **1. Linear projections**

Exactly as in encoder:

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V,
$$

with

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}.
$$

---

## **2. Causal self-attention**

Compute raw attention scores:

$$
S = \frac{QK^\top}{\sqrt{d_k}}.
$$

Apply **causal mask**:

$$
S_{ij} =
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

Softmax row-wise:

$$
A = \text{softmax}(S)V
\quad\in\mathbb{R}^{N \times d_k}.
$$

Interpretation:
**Row i** attends only to **columns 1..i**.

---

## **2.5 Output projection**

$$
A_{\text{proj}} = A W_O
\quad \text{where } W_O \in \mathbb{R}^{d_k \times d}.
$$

---

## **3. Residual + LayerNorm**

$$
X' = \text{LayerNorm}(X + A_{\text{proj}}).
$$

---

## **4. Feed-Forward Network (FFN)**

Same as encoder:

$$
H = \phi(X' W_1 + b_1) W_2 + b_2,
$$

with

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$

---

## **5. Residual + LayerNorm**

$$
X_{\text{out}} = \text{LayerNorm}(X' + H).
$$

This is final hidden state for each token.

---

## **6. LM head for generation**

Take **last token hidden state** ($h_t$):

$$
\text{logits} = h_t W_E^\top,
$$

where embeddings are often **tied**:

$$
W_E \in \mathbb{R}^{|V| \times d}.
$$

Softmax → probability over next token.

---

# **B. Parameter Counting (Per Layer)**

Identical to encoder except **no cross-attention**.

### Attention projections

$$
W_Q, W_K, W_V: d \times d_k,\quad
W_O: d_k \times d
$$

Total:

$$
4d d_k.
$$

### Feed-forward

$$
2d d_{\text{ff}}.
$$

### LayerNorm

Two LayerNorms →

$$
4d.
$$

### Embeddings

- Token: $|V| \times d$
- Positional: $N_{\max} \times d$

### LM Head

Often **tied** → no extra params.

---

# **C. Training + Regularisation**

### **Training objective**

$$
\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t})
$$

- True past tokens fed in (“**teacher forcing**”).
- Standard cross-entropy over vocabulary.

### **Regularisation**

- Dropout (attention weights, FFN output).
- Weight decay (L2).
- LayerNorm stabilisation.

---

# **D. Typical Tasks**

Everything framed as **next-token prediction**:

- Text continuation / generation.
- Summarisation (as generation).
- Translation (as generation).
- Retrieval-augmented tasks.
- Classification via instruction prompting.

---

# **E. What It Can / Can’t Capture**

### **Can**

- Strong generative expressivity.
- Long-range context (within window).
- Few-shot / zero-shot via prompting.

### **Can’t**

- Bidirectional context in a single forward pass.
- Efficient very-long-sequence computation.

---

# **F. Pros / Cons**

### **Pros**

- Amazing generative capabilities.
- Simplest architecture (one stack).
- Natural for in-context learning.

### **Cons**

- No backward context.
- Quadratic attention cost.
- Pretraining compute cost huge.

---

# 1️⃣0️⃣ **Transformer Encoder–Decoder (T5 / Seq2Seq)**

(Updated to match full encoder detail)

---

# **Core idea**

Two stacks:

- **Encoder**: exactly like encoder-only BERT (bidirectional self-attn).
- **Decoder**: GPT-style masked self-attn **plus cross-attention** to encoder output.

Used for sequence-to-sequence mapping:

$$
x_{1..N} \to y_{1..M}.
$$

---

# **A. Computational / Forward Step**

We track shapes explicitly.

---

## **1. Encoder stack**

Input:

$$
X \in \mathbb{R}^{N \times d}.
$$

For each layer:

1. **Self-attn** (bidirectional):

   $$
   Q = XW_Q,\quad K = XW_K,\quad V = XW_V
   $$

   $$
   A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
   $$

2. **Residual + LayerNorm**:

   $$
   X' = \text{LayerNorm}(X + A W_O).
   $$

3. **FFN**:

   $$
   H = \phi(X'W_1 + b_1)W_2 + b_2.
   $$

4. **Residual + LayerNorm**:
   $$
   H_{\text{enc}} = \text{LayerNorm}(X' + H).
   $$

Output of encoder:

$$
H_{\text{enc}} \in \mathbb{R}^{N \times d}.
$$

---

## **2. Decoder stack (per layer)**

Input decoder tokens:

$$
S \in \mathbb{R}^{M \times d}.
$$

---

### **2.1 Masked self-attention (decoder)**

Same as GPT:

$$
Q_s = S W_Q^{(\text{self})},\quad
K_s = S W_K^{(\text{self})},\quad
V_s = S W_V^{(\text{self})}.
$$

Causal mask on:

$$
S_s = \frac{Q_s K_s^\top}{\sqrt{d_k}}
\quad\text{with } j > i \text{ masked}.
$$

Then:

$$
A_s = \text{softmax}(S_s)V_s.
$$

Residual + LN:

$$
S' = \text{LayerNorm}(S + A_s W_O^{(\text{self})}).
$$

---

### **2.2 Cross-attention (key difference)**

Queries from decoder; keys/values from encoder.

$$
Q_c = S' W_Q^{(\text{cross})},\quad
K_c = H_{\text{enc}} W_K^{(\text{cross})},\quad
V_c = H_{\text{enc}} W_V^{(\text{cross})}.
$$

Attention:

$$
A_c = \text{softmax}\left(\frac{Q_c K_c^\top}{\sqrt{d_k}}\right)V_c.
$$

Residual + LN:

$$
S'' = \text{LayerNorm}(S' + A_c W_O^{(\text{cross})}).
$$

---

### **2.3 Feed-Forward Network**

Same FFN structure:

$$
F = \phi(S''W_1 + b_1)W_2 + b_2.
$$

Residual + LN:

$$
H_{\text{dec}} = \text{LayerNorm}(S'' + F)
\quad \in \mathbb{R}^{M \times d}.
$$

This is final decoder output.

---

## **3. Output head**

$$
\text{logits} = H_{\text{dec}} W_E^\top.
$$

Softmax gives $P(y_t \mid y_{<t}, x)$.

---

# **B. Parameter Counting**

Total params include:

---

## **1. Encoder (per layer)**

Same as encoder-only:

$$
4d d_k + 2d d_{\text{ff}} + 4d.
$$

---

## **2. Decoder (per layer)**

Has **three** attention modules:

1. Masked self-attn:
   $4d d_k$

2. Cross-attn (Q from decoder, K/V from encoder):
   also $4d d_k$

3. FFN:
   $2d d_{\text{ff}}$

4. 3 LayerNorms →
   $6d$

**Total per decoder layer:**

$$
8d d_k + 2d d_{\text{ff}} + 6d.
$$

---

## **3. Embeddings**

- Input token embeddings: $|V_{\text{in}}| \times d$

- Output token embeddings: $|V_{\text{out}}| \times d$
  (Often tied in text-to-text models.)

- Positional embeddings for input + output.

---

# **C. Training + Regularisation**

### **Training objective**

Teacher-forced seq2seq loss:

$$
\mathcal{L} = -\sum_t \log P(y_t \mid y_{<t}, x).
$$

### **Pretraining (T5)**

- **Span corruption** (mask random spans with sentinel tokens).
- Predict the missing text with decoder.

### Regularisation

- Dropout
- LayerNorm
- Weight decay

---

# **D. Typical Tasks**

- Machine translation (canonical use).
- Abstractive summarisation.
- Paraphrasing.
- Question answering.
- Text-to-text unification (T5 slogan: **“Everything is text-to-text.”**)

---

# **E. What It Can / Can’t Capture**

### **Can**

- Rich input understanding (encoder).
- Strong generation conditioned on input (decoder).
- Flexible seq2seq modelling structure.

### **Can’t**

- Avoid quadratic attention cost.
- Efficiently handle very long inputs and outputs.

---

# **F. Pros / Cons**

### **Pros**

- Best architecture for **supervised seq2seq**.
- Encoder specialises in reading; decoder specialises in writing.
- Excellent controllability.

### **Cons**

- Heaviest architecture (two stacks).
- More complex attention.
- Slower training/inference than pure encoder or pure decoder models.

# 3. Forumulas

## 1️⃣ BAYES’ RULE

### **Formula**

$$P(A \mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

### **Meaning**

Infer the probability of a **hidden cause** $A$ (class, tag, topic) from an **observed signal** $B$ (words, features).

---

### **Strengths**

- **Uses priors** $P(A)$ → robust when data is sparse.
- Natural fit for **generative classifiers** (Naive Bayes).
- Gives **posterior** $P(A\mid B)$, which is what we actually want for classification.

### **Weaknesses**

- Needs $P(B)$, often computed via **law of total probability** (can be intractable with many classes).
- Sensitive to **incorrect priors** and assumptions (e.g. Naive Bayes independence).

---

### **Uses in NLP**

- **Naive Bayes** (sentiment, topic classification).
- **WSD**: sense as hidden variable.
- **HMMs**: interpreting hidden states given observations.

---

### **Worked Example (Sentiment classification)**

Given:

- $P(\text{positive}) = 0.4$
- $P(\text{negative}) = 0.6$
- $P(\text{“excellent”} \mid \text{positive}) = 0.1$
- $P(\text{“excellent”} \mid \text{negative}) = 0.01$

Compute $P(\text{positive} \mid \text{“excellent”})$.

1. **Denominator** via law of total probability:

$$
P(\text{“excellent”}) =
0.1 \cdot 0.4 + 0.01 \cdot 0.6
= 0.04 + 0.006
= 0.046
$$

2. **Posterior**:

$$
P(\text{positive} \mid \text{“excellent”})
= \frac{0.1 \cdot 0.4}{0.046}
= \frac{0.04}{0.046}
\approx 0.87
$$

**Interpretation:** Seeing “excellent” makes the text **very likely positive**, even though positive is not the majority class.

---

## 2️⃣ CONDITIONAL PROBABILITY

### **Formula**

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

### **Meaning**

Probability of **A** happening given that **B** has happened — the basic building block of **sequence models**.

---

### **Strengths**

- Lets us **factor** complex joint distributions into manageable pieces:
  $P(w_1,\dots,w_T)=\prod_t P(w_t\mid w_{<t})$.
- Underlies **N-grams**, **HMMs**, and **autoregressive LMs**.

### **Weaknesses**

- Accurate estimation needs large data; suffers badly from **sparse counts**.
- Motivates **smoothing** and more powerful models (RNNs, Transformers).

---

### **Uses in NLP**

- **N-gram LMs**: $P(w_i\mid w_{i-n+1}^{i-1})$.
- **HMMs**: $P(\text{tag}_t \mid \text{tag}_{t-1})$, $P(\text{word}\_t \mid \text{tag}\_t)$.
- Conceptually: **attention** defines a conditional distribution over positions.

---

### **Worked Example (Corpus counts)**

A corpus has 200 sentences:

- 40 sentences contain _cat_
- 10 sentences contain both _cat_ and _chases_

Approximate sentence-level probabilities:

- $P(\text{cat}) = 40 / 200 = 0.2$
- $P(\text{chases}, \text{cat}) = 10 / 200 = 0.05$

Then:

$$
P(\text{chases} \mid \text{cat})
= \frac{0.05}{0.2}
= 0.25
$$

**Interpretation:** Among sentences with _cat_, **25%** also contain _chases_.

---

## 2️⃣.2️⃣ JOINT PROBABILITY

### **Formula**

$P(A,B) = P(A \cap B)$

and its key factorisations:

$P(A,B) = P(A\mid B)P(B) = P(B\mid A)P(A)$

### **Meaning**

Probability that **A and B happen together**.  
It is the core building block that **Bayes’ rule** and **conditional probability** are derived from.

---

### **Strengths**

- Fundamental to **generative models**: entire sequence probabilities are joint probabilities.
- Allows factorisation via the **chain rule**:
  $P(w_1,\dots,w_T)=\prod_t P(w_t \mid w_{<t})$

### **Weaknesses**

- Direct estimation of high-dimensional joints is impossible with sparse data → must factorise into conditionals and use **smoothing** or **neural models**.
- Co-occurrence alone doesn’t tell you direction of dependence.

---

### **Uses in NLP**

- **N-gram LMs**: joint over sequences via product of conditionals.
- **HMMs**: joint over tags + words
  $P(z_{1:T}, x_{1:T}) = P(z_1)\prod_t P(z_t\mid z_{t-1})P(x_t\mid z_t)$
- **Co-occurrence matrices** used by SGNS / distributional semantics.

---

## 3️⃣ LAW OF TOTAL PROBABILITY

### **Formula**

$$P(B)=\sum_i P(B\mid A_i)P(A_i)$$

### **Meaning**

Total probability of **B** is the sum over contributions from all **latent causes** $A_i$.

---

### **Strengths**

- Connects **latent-variable models** (like HMMs) to observed probabilities.
- Provides denominator in **Bayes’ Rule**.

### **Weaknesses**

- Requires a **complete, disjoint** set of $A_i$ (often unrealistic).
- Can be intractable if there are many possible hidden states.

---

### **Uses in NLP**

- Computing $P(\text{word})$ from POS-tag-conditioned distributions.
- Marginalising hidden **HMM states**.
- Normalisation sums in generative models.

---

### **Worked Example (Noun vs verb)**

Let $C \in {\text{noun}, \text{verb}}$. Suppose:

- $P(\text{noun}) = 0.6$, $P(\text{verb}) = 0.4$
- $P(w \mid \text{noun}) = 0.1$
- $P(w \mid \text{verb}) = 0.02$

Then:

$$
P(w)
= P(w\mid \text{noun})P(\text{noun}) + P(w\mid \text{verb})P(\text{verb})
= 0.1\cdot 0.6 + 0.02\cdot 0.4
= 0.06 + 0.008
= 0.068
$$

**Interpretation:** Overall, **6.8%** of tokens are this word, aggregating across noun/verb uses.

---

## 4️⃣ ADD-ONE / ADD-ALPHA SMOOTHING

### **Formula (unigram)**

$$P(w)=\frac{C(w)+\alpha}{N+\alpha |V|}$$

For conditional (e.g. N-gram):

$$P(w_i\mid h)=\frac{C(h,w_i)+\alpha}{C(h)+\alpha |V|}$$

---

### **Meaning**

Adds a small **pseudo-count** $\alpha$ to every event so unseen events have **non-zero** probability.

---

### **Strengths**

- Simple, closed-form, easy to compute in exam.
- Prevents zero probabilities → critical for **N-gram products**.

### **Weaknesses**

- **Over-smooths**, especially with large vocabularies.
- Unrealistic distributions → replaced in practice by **Kneser–Ney**.

---

### **Uses in NLP**

- Textbook **N-gram LMs**.
- Naive Bayes when many features are unseen in a class.

---

### **Worked Example (Unigram)**

Vocabulary size: $|V| = 5$
Total tokens: $N = 100$
Word $w$ appears $C(w)=3$ times.
Let $\alpha = 1$ (add-one).

$$P(w)=\frac{3+1}{100+1\cdot 5}=\frac{4}{105}\approx 0.0381$$

**Interpretation:** The smoothed probability is slightly **higher** than raw MLE (3/100=0.03) because we expanded the denominator and added pseudo-counts.

---

## 5️⃣ DOT PRODUCT

### **Formula**

$$u \cdot v = \sum_i u_i v_i$$

### **Meaning**

Measures alignment scaled by magnitude.

- Large if vectors point in the same direction and are long.
- Small if vectors are short or orthogonal.

---

### **Strengths**

- Very fast to compute.
- Used directly in attention mechanisms (dot-product attention).
- Captures both magnitude and orientation → useful when vector length carries semantic information (e.g., SGNS frequency effects).

### **Weaknesses**

- Magnitude-sensitive: longer vectors yield larger dot products even if direction is not highly aligned.
- Cannot be used to compare vectors when norms differ widely (common in word embeddings).

---

### **Uses in NLP**

- **Self-attention / cross-attention**: $\text{score}(q,k) = q^\top k$
- **SGNS objective**: $u_{\text{pos}}^\top v_t$
- Encoding co-occurrence or similarity in simpler models.

---

### **Worked Example**

Two embeddings:

- $u = (1, 2)$
- $v = (3, 4)$

Compute:

$$u \cdot v = 1 \cdot 3 + 2 \cdot 4 = 3 + 8 = 11$$

**Interpretation:** A large positive dot product → vectors point in a broadly similar direction and have sizable magnitudes.

## 5️⃣.2️⃣ COSINE SIMILARITY

### **Formula**

$$\cos(u,v) = \frac{u \cdot v}{\|u\| \|v\|}$$

### **Meaning**

Measures pure directional alignment, ignoring magnitude.

- $= 1$: same direction
- $= 0$: orthogonal
- $= -1$: opposite direction

---

### **Strengths**

- Norm-invariant → frequency effects do not distort similarity.
- Performs well in high-dimensional embedding spaces.
- Better for semantic similarity where direction matters more than length.

### **Weaknesses**

- Affected by global geometry issues such as embedding anisotropy.
- Cannot distinguish antonyms (e.g., hot and cold may have similar directions).
- Does not incorporate magnitude information when that might be meaningful.

---

### **Uses in NLP**

- Similar word retrieval (synonyms, paraphrases).
- Document/sentence similarity.
- Clustering embeddings.
- Intrinsic evaluation of word vectors (e.g. SimLex, WordSim-353).

---

### **Worked Example**

Vectors:

- $u = (1, 2)$
- $v = (3, 4)$

Dot product (from earlier):

$$u \cdot v = 11$$

Norms:

$$\|u\| = \sqrt{1^2 + 2^2} = \sqrt{5}, \quad \|v\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

Cosine similarity:

$$\cos(u,v) = \frac{11}{\sqrt{5} \cdot 5} = \frac{11}{5\sqrt{5}} \approx 0.984$$

**Interpretation:** Near 1 → vectors point in almost the same direction, indicating strong semantic similarity.

---

## 6️⃣ EUCLIDEAN DISTANCE

### **Formula (2D)**

$$d(x,y)=\sqrt{(x_1-y_1)^2 + (x_2-y_2)^2}$$

### **Meaning**

Geometric distance between two embeddings in space.

---

### **Strengths**

- Intuitive interpretation as “how far apart”.

### **Weaknesses**

- In high dimensions, distances tend to **concentrate** (curse of dimensionality).
- Less useful than cosine for semantic similarity.

---

### **Uses**

- KNN classification.
- Clustering.
- Detecting embedding outliers.

---

### **Worked Example**

Word embeddings:

- $x = (1,2)$ (cat)
- $y = (3,4)$ (dog)

$$
d(x,y)=\sqrt{(1-3)^2+(2-4)^2}
= \sqrt{(-2)^2 + (-2)^2}
= \sqrt{4+4}
= \sqrt{8} \approx 2.828
$$

**Interpretation:** Larger distance → less similar; here they’re moderately far apart.

---

## 7️⃣ L2 REGULARISATION

### **Formula**

$$L' = L + \lambda |w|^2$$

where $|w|^2 = \sum_j w_j^2$.

---

### **Meaning**

Adds penalty for **large weights**, encouraging smaller, smoother models.

---

### **Strengths**

- Reduces **overfitting**.
- Makes optimisation more stable; discourages extreme weights.

### **Weaknesses**

- Does **not** create sparsity (unlike L1).
- Too large $\lambda$ → **underfitting** (weights shrunk too much).

---

### **Uses**

- Logistic / softmax regression for text.
- Neural networks, weight decay.

---

### **Worked Example**

Suppose:

- Original loss $L = 0.40$
- Weight vector $w = [2,1]$
- $\lambda = 0.1$

1. Compute squared norm:

$$|w|^2 = 2^2 + 1^2 = 4 + 1 = 5$$

2. New loss:

$$L' = 0.40 + 0.1 \cdot 5 = 0.40 + 0.5 = 0.90$$

**Interpretation:** The model is penalised for large weights; training will prefer smaller weights if they don’t hurt performance too much.

---

## 8️⃣ PRECISION, RECALL, F1

### **Formulas**

$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = \frac{2PR}{P+R}$$

---

### **Meaning**

- **Precision**: among predicted positives, how many are correct?
- **Recall**: among true positives, how many did we find?
- **F1**: harmonic mean, balances both.

---

### **Strengths**

- Handle **class imbalance** better than raw accuracy.
- F1 good when both FP and FN matter.

### **Weaknesses**

- F1 ignores **true negatives**.
- Precision alone ignores missed positives; recall alone ignores false positives.

---

### **Uses**

- NER, hate-speech detection, toxicity detection, IE.
- Any **classification / sequence labelling** tasks.

---

### **Worked Example (Spam)**

Given:

- TP = 20
- FP = 5
- FN = 10

1. Precision:

$$P = \frac{20}{20 + 5} = \frac{20}{25} = 0.8$$

2. Recall:

$$R = \frac{20}{20 + 10} = \frac{20}{30} \approx 0.667$$

3. F1:

$$
F_1 = \frac{2 \cdot 0.8 \cdot 0.667}{0.8 + 0.667}
\approx \frac{1.067}{1.467} \approx 0.727
$$

**Interpretation:** Model is strong but misses some spam (recall < precision).

**Metric that penalises false negatives most directly:** **Recall**.

---

## 9️⃣ CROSS-ENTROPY

### 1️⃣ Entropy

#### What Entropy Measures

Entropy $H(X)$ quantifies unpredictability of a random variable.

- If outcomes are evenly distributed → high entropy (high uncertainty).
- If one outcome is dominant → low entropy (predictable).

#### Intuition

Measured in bits. “How many yes/no questions do I need, on average, to identify the outcome?”

#### Formula

$$H(X) = -\sum_x P(x) \log_2 P(x)$$

#### Interpretation

- Uniform distribution over $N$ outcomes:
  $$H = \log_2 N$$
- Skewed distribution: Entropy drops because uncertainty drops.
- English entropy: ~1.3 bits/character (strong frequency biases → less uncertainty).

### 2️⃣ Cross-Entropy

#### General Formula

For gold distribution $p(y)$ and model distribution $q(y)$:

$$H(p,q) = -\sum_y p(y) \log q(y)$$

#### One-Hot Case

If the true class is $c$ - $p(c) = 1$ for correct $c$ and $p(y) = 0$ for all others:

$$H(p,q) = -\log q(c)$$

Given $q(c) = P(y_i = c | x_i; \theta)$

for each training example:

$$\text{Cross-Entropy} = -logP(y_i | x_i; \theta)$$

Which is equivalent to **Negative log-likelihood**

#### Meaning

Expected **surprise** of the **true labels** under the model’s predicted distribution.

Equivalent to the average negative log-probability assigned to the correct outputs.

- **High Cross-Entropy** (High Surprise) = predictions are far from their labels
- **Low Cross-Entropy** (Low Surprise) = Prediction close to labels

#### Strengths

- Directly corresponds to **maximum likelihood training**.
- Smooth gradient → works well with backprop.
- Standard for classification, LM token prediction, SFT.

#### Weaknesses

- Punishes confidently wrong predictions very strongly.
- Not always aligned with human quality metrics in generation tasks.

#### Uses in NLP

- Logistic regression / softmax regression.
- RNN / Transformer LMs (token-level training loss).
- SFT for LLMs and general classification tasks.

#### Worked Example (Word Prediction)

Given:

- Model predicts:

  | word | prob |
  | ---- | ---- |
  | the  | 0.5  |
  | a    | 0.3  |
  | cat  | 0.2  |

- True word = "cat" → one-hot: $p(\text{cat}) = 1$.

#### Cross-entropy:

$$H(p,q) = -\log q(\text{cat}) = -\log(0.2) \approx 1.609$$

Interpretation:

- Somewhat surprised.
- Higher probability on the true class → lower cross-entropy.

### 3️⃣ Perplexity

#### Definition

$$PP = 2^{H_M}$$

where $H_M$ is the model's cross-entropy on the data.

#### Intuition

- Perplexity ≈ model’s **effective number of choices** at each prediction step.
- Lower = better.

#### Interpretation Scale

- $PP \approx 2$: model is very confident (≈ 2 plausible next tokens).
- $PP \approx 100$: highly uncertain (many plausible next tokens).

#### Why Logs Appear

- Avoid numerical underflow when multiplying many small probabilities.
- Cross-entropy uses logs; perplexity simply exponentiates.

#### Example

If cross-entropy = 3 bits/word:

$$PP = 2^3 = 8$$

→ model behaves as if it has 8 effective choices per word.

#### Model Comparison

- Unigram: high perplexity (no context)
- Bigram: lower.
- Trigram: lower still.

More context → more certainty → lower perplexity.

### 4️⃣ How These Quantities Fit Together

How These Quantities Fit Together

- Entropy: Intrinsic unpredictability of the data distribution.
- Cross-Entropy: Model’s estimate of that unpredictability.
- Equals entropy only if model = true distribution.
- Perplexity: A readability-friendly transformation of cross-entropy. “How confused is the model?”

# 4. Probability Estimation

# **1. Maximum Likelihood Estimation (MLE)**

**Definition**

- Choose parameters $\theta$ that **maximise the likelihood** of observed corpus data.
- For an N-gram:
  $$P(w_i | h) = \frac{\text{count}(h, w_i)}{\text{count}(h)}$$

**Pros**

- **Unbiased** estimator (given enough data).
- **Simple**, closed-form for many models.
- **Interpretable**.

**Cons / Characteristic Errors**

- **Zero-probabilities** for unseen events ⇒ catastrophic failure in generative models.
- **Overfitting** on small corpora.
- Poor generalisation beyond observed contexts.

**When acceptable?**

- Very **large corpora**.
- Tasks where **exact probabilities matter less** (e.g., ranking with smoothing fallback).

**When unacceptable?**

- Small corpora, rare events, OOV-heavy domains.

---

# **2. Add-One / Add-Alpha (Laplace) Smoothing**

**Definition**

- Add a small constant ($\alpha$) to counts to avoid zero probabilities.
- Formula:
  $$P(w | h) = \frac{\text{count}(h,w) + \alpha}{\text{count}(h) + \alpha \cdot |V|}$$

**Pros**

- Eliminates **zero-probabilities**.
- Very **simple**.

**Cons / Errors**

- **Over-smooths**: probability mass redistributed too uniformly.
- Degrades performance on high-frequency items.

**Acceptable?**

- Toy examples, demonstrations, extremely low-resource settings.

**Unacceptable?**

- Real NLP tasks where accuracy matters (prefer Kneser–Ney, Good–Turing).

---

# **3. Cross-Entropy Loss**

**Definition**

- Measures how well predicted distribution $q$ approximates true distribution $p$.
- $$H(p,q) = -\sum p(w) \log q(w)$$
- For supervised training: use target labels as $p$ (one-hot).

$p(w) = \begin{cases} 1 & \text{if } w = y \\ 0 & \text{otherwise}. \end{cases}$

Since $p(w) = 0$ for all classes except $w=y$

$H(p,q) = -\Big( p(y)\log q(y) + \sum_{w\neq y} p(w)\log q(w) \Big)$

But since $p(y)=1$ and $p(w\neq y) = 0$:

$H(p,q) = - \log q(y)$

**Used for:**

- Training classifiers, language models, seq2seq models.

**Strengths**

- Directly optimises model likelihood.
- Differentiable, compatible with SGD.

**Weaknesses**

- Sensitive to **miscalibrated probabilities**.
- Encourages overconfidence.

**Alternatives**

- KL divergence (equivalent up to constant).
- Margin-based losses.

---

# **4. Teacher Forcing**

**Definition**

- During training, RNN/Transformer decoder receives **gold previous token** instead of its own prediction.

**Pros**

- Faster, more stable training.
- Helps model learn correct conditional distributions.

**Cons / Errors**

- **Exposure bias**: model never sees own prediction errors during training.
- At inference, small mistakes can **cascade**.

**Mitigations**

- Scheduled sampling.
- Sequence-level training (e.g., RL, minimum-risk training).

---

# **5. Stochastic Gradient Descent (SGD)**

**Definition**

- Update parameters using **gradient estimates** from minibatches.

**Pros**

- Scales to huge datasets.
- Good exploration of parameter space due to noise.
- Fast convergence with variants (Adam, RMSProp).

**Cons / Errors**

- Highly sensitive to **learning rate**.
- Can get stuck in bad minima/saddle points.
- Noisy gradients ⇒ unstable without tuning.

**Acceptable?**

- Always the default for neural models.

**Unacceptable?**

- When data is tiny (closed-form solutions preferable).

---

# **6. Backpropagation**

**Definition**

- Compute gradients via the **chain rule** through all layers.

**Core idea**

- Local gradient × upstream gradient.
- Enables training deep networks.

**Failure modes**

- Vanishing/exploding gradients (mitigated by LSTMs, residual connections).
- Requires differentiable components.

---

# **7. Negative Sampling**

**Definition**

- Approximate softmax by contrasting **true pairs** with a small set of **noise samples**.
- Common in **word2vec skip-gram**.

**Pros**

- Huge speed improvements.
- Good embeddings with limited computation.

**Cons / Errors**

- Choice of **negative distribution** strongly affects performance.
- Estimates only **relative** probabilities (not full normalised softmax).

**Typical errors**

- Rare words poorly learned when insufficient negatives.

---

# **8. Contrastive Learning**

**Definition**

- Learn representations by **bringing positives closer** and **pushing negatives apart**.
- Often uses **InfoNCE** loss.

**Examples**

- Sentence embedding models.
- Image-text alignment (CLIP).

**Pros**

- Strong generalisation.
- No need for labelled data.

**Cons**

- Requires large batch sizes / memory.
- Negative selection crucial.

**Sources of error**

- False negatives (different sentences with same meaning).

---

# **9. Transfer Learning**

**Definition**

- Use parameters trained on one task/domain for a different task.

**Forms**

- **Feature-based** (use frozen embeddings).
- **Fine-tuning** (update all weights).
- **LoRA/adapter layers**.

**Pros**

- Massive performance boost on small datasets.
- Leverages world knowledge.

**Cons**

- **Catastrophic forgetting**.
- Domain mismatch and biases propagate.

---

# **10. In-Context Learning (ICL)**

**Definition**

- Model learns behaviour from **examples in the prompt**, not parameter updates.

**Pros**

- No training required.
- Flexibility across tasks.
- Data Efficient
- Immediate Deployment

**Cons / Errors**

- Very sensitive to **prompt order**, formatting, bias.
- Task performance unstable for small LMs.
- No guarantee of generalisation.

---

# **11. Zero-Shot and Few-Shot Learning**

**Zero-shot**

- Provide only natural-language task description.

**Few-shot**

- Provide small set of labelled examples in context.

**Pros**

- Extremely efficient.
- Works best with LLMs trained on broad mixtures.

**Cons**

- Erratic for structured tasks.
- Relies heavily on pretraining prior.

**When unacceptable?**

- Safety-critical or high-precision tasks.

---

# **12. Cross-Lingual Knowledge Transfer**

### **Definition**

- Using a model trained on **source languages** to perform tasks in a **different target language**, without requiring large labelled datasets in the target language.

### **Mechanisms**

- **Multilingual embeddings** → map words across languages into a shared semantic space.
- **Shared subword vocabularies (BPE/SentencePiece)** → overlapping morphology/scripts supports transfer.
- **Emergent alignment** → shared layers naturally align languages even _without_ parallel data.

### **Transfer Strategies**

- **Zero-Shot Transfer** → train on EN/DE, test directly on FR (no FR labels).
- **Translate-Test** → translate FR input → EN; use a monolingual EN model at inference.
- **Translate-Train** → translate EN training data → FR; fine-tune on synthetic FR data.

**Pros**

- **Helps low-resource languages**.
- Enables **zero-shot multilingual tasks**.
- **Single large model** vs thousands of monolingual
- Wider **global access to NLP** systems

**Cons / Errors**

- Transfer depends on **linguistic similarity**. (i.e shared morphology - latin, germanic)
- **Script Mismatch** (latin vs Arabic) -> weak transfer
- **Curse of Multilinguality** -> too many languages dilute parameter capacity per language
- **Tokenisation bias** ("fertility") -> Low resource languages get more subwords -> longer sequences -> higher compuite cost
- **Cultural shift issues:** models fail on concepts absent/different in high resoruce training data

---

# **13. Pretraining Objectives**

## **a) Causal Language Modelling (CLM)**

- Predict next token using only **left context**.
- Used in GPT-type models.
  **Strength:** excels at generation.
  **Weakness:** poorer at bidirectional understanding.

## **b) Masked Language Modelling (MLM)**

- Predict masked tokens using **both sides of context**.
- Used in BERT.
  **Strength:** strong encoding representations.
  **Weakness:** not generative.

## **c) Denoising Language Modelling**

- Corrupt input (shuffling, dropping, masking) and reconstruct.
- Used in T5, BART.
  **Strength:** robust representations.
  **Weakness:** expensive pretraining.

---

# **14. Post-Training Objectives (Expanded)**

Post-training modifies a **pre-trained next-token predictor** into a model that **follows instructions**, **aligns with human preferences**, and **avoids harmful behaviour**.  
Three dominant families of objectives are used: **SFT**, **RLHF**, and **DPO/RLVR-family objectives**.

---

# **a) Supervised Fine-Tuning (SFT)**

## **What It Is**

- Fine-tuning on curated **(instruction → response)** pairs.
- Examples come from **human demonstrations**, synthetic Teacher–Student pipelines, or multi-task instruction datasets (e.g., **Natural Instructions**, SNI).

## **How It Works**

- Input: formatted prompt, often with `<|user|>` and `<|assistant|>` delimiters.
- Model predicts output tokens.
- **Loss:** cross-entropy only on **assistant tokens**.

## **What It Teaches**

- **Task grounding:** recognising “what task is being asked”.
- **Behavioural priors:** politeness, format, safety scaffolding.
- **Generalisation:** across tasks via multitask training.

## **Pros**

- **Stable** (pure gradient descent).
- **Controllable behaviour** through curated demonstrations.
- No negative examples needed.

## **Cons**

- **Imitation only** → no notion of “better vs worse”.
- Strongly affected by **dataset quality, biases, demographic skew**.
- Can make models **overly verbose** or **pattern-copying**.

## **Typical Errors**

- **Shallow compliance:** follows templates without deep understanding.
- **Hallucination style-bias:** repeats demo patterns even when wrong.
- **Limited safety:** cannot resolve ambiguous / harmful user intents.

---

# **b) RLHF (Reinforcement Learning from Human Feedback)**

## **Why RLHF Exists**

- SFT cannot encode **preferences**, only behaviour.
- We need a mechanism to **compare outputs**:
  - “This answer is better than that one.”

## **Pipeline Components**

1. **Preference Data**

   - Human raters pick **preferred** output from pairs.

2. **Reward Model (RM)**

   - Trained to predict which output humans prefer.
   - Implements a **Bradley–Terry** comparison:
     - $\sigma(R(o^+) - R(o^-))$

3. **RL Optimisation (PPO)**

   - Policy (the LLM) is updated to **maximise RM score**  
      while staying close to the SFT model via **KL penalty**.

## **What It Teaches**

- **Helpful** (follows preferred behaviours).
- **Honest** (doesn’t fabricate confident lies when penalised).
- **Harmless** (avoids dangerous instructions).

## **Pros**

- Embeds **preference learning**, not mere imitation.
- Captures **nuanced human signals** (tone, respectfulness).
- Helps avoid pathological or unsafe behaviours.

## **Cons / Characteristic Errors**

- **Reward hacking:**
  - Model finds outputs that trick the RM but degrade truthfulness.
- **Over-optimisation / collapse:**
  - Becoming verbose, generic, or stylistically “safe”.
- **Drift:**
  - If KL penalty too small → catastrophic loss of base abilities.
- **Expensive:**
  - Must load **policy + reference + reward model** simultaneously.

## **Why PPO?**

- Stabilises updates by **clipping** large policy shifts.
- Needed because language-generation RL is notoriously unstable.

---

# **c) RLVR / DPO-family (Direct Preference Optimisation)**

These methods aim to **avoid full RL**, keeping the alignment signal but simplifying optimisation.

---

## **RLVR (RL with Verifiable Rewards)**

### **Core Idea**

- Replace Reward Model with a **programmatic verifier**:
  - Math tasks: exact numerical answer.
  - Code tasks: runs + passes tests.
  - Safety tasks: rule-based validators.

### **Pros**

- **Cheaper & simpler:**
  - Only Policy + Reference model required.
- **Stable:** No PPO instability.
- Works well for **objective-verifiable** tasks.

### **Cons**

- **Reasoning mismatch:**
  - Model may reach correct answer via flawed logic.
- **Reward gaming:**
  - Exploiting quirks in verifiers (e.g., formatting hacks).
- **Limited applicability:**
  - Not useful for open-ended dialogue, ethics, writing style.

---

# **TLDR SUMMARY**

- **MLE**: simple but brittle → fails on unseen events.
- **Smoothing (Add-α)**: prevents zeros, over-smooths.
- **Cross-entropy**: core loss for all LMs.
- **Teacher forcing**: stable training but exposure bias.
- **SGD/Backprop**: optimisation backbone; sensitive to tuning.
- **Negative sampling / contrastive**: efficient learning of embeddings.
- **Transfer / ICL / few-shot**: leverage pretrained knowledge.
- **Pretraining objectives**: CLM (generation), MLM (encoding), denoising (robust).
- **Post-training**: SFT, RLHF, RLVR align models to human preferences.

# 5. Generation & Decoding

## **1. Core Idea**

- At inference, the model outputs a **probability distribution** over the vocabulary at each timestep.
- Decoding strategies determine **which token to choose** from this distribution.
- Trade-off: **determinism vs. diversity**, **efficiency vs. quality**, **search width vs. computational cost**.

---

# **2. Greedy Decoding**

### **What it is**

- Always pick the **argmax** token at each timestep.

### **How it works**

1. Model outputs probabilities: e.g. `{"cat": 0.4, "dog": 0.3, "sat": 0.2, ...}`
2. Choose the **highest-probability** token.
3. Feed it back into the model and repeat.

### **Strengths**

- **Fast**, **deterministic**, **low computation**.

### **Weaknesses**

- Gets stuck in **local optima**.
- Produces **generic / repetitive** text.
- Often misses globally better sequences.

### **Typical exam phrasing**

- “Given these output probabilities, apply _greedy_ decoding to produce the next token.”

---

# **3. Beam Search**

### **What it is**

- A **heuristic search** that keeps multiple candidate sequences (**beam width = k**) at each timestep.

### **How it works**

1. For each partial hypothesis, expand with **all possible next tokens**.
2. **Score** each sequence (usually sum of log-probs).
3. Keep the **top k** sequences.
4. Continue until EOS.

### **Strengths**

- Explores more of the search space than greedy.
- Produces **higher probability** sequences.
- Standard for machine translation / summarisation.

### **Weaknesses**

- **Computationally expensive** (≈ k× slower).
- Still approximate.
- Larger beam can produce **short, generic, or repetitive** outputs (length-bias issue).

### **Exam must-knows**

- You must be able to **do one step**:  
   Expand candidates → compute **sequence log-probabilities** → keep best **k**.

---

# **4. Sampling (Stochastic Decoding)**

### **What it is**

- Choose the next token by **sampling** from the full probability distribution.

### **How it works**

1. Model outputs probabilities.
2. Treat them as a **categorical distribution**.
3. Draw a random sample.

### **Strengths**

- Introduces **diversity** / variation.
- Avoids deterministic repetition.

### **Weaknesses**

- Can sample **very low-quality** or incoherent tokens from the long tail.
- High variance; outputs unpredictable.

---

# **5. Top-k Sampling**

### **What it is**

- Restrict sampling to the **k most probable tokens**, renormalise, then sample.

### **How it works**

1. Sort tokens by probability.
2. Keep top **k** (e.g. k=40).
3. Renormalise probabilities.
4. Sample from the reduced set.

### **Strengths**

- Removes low-probability noise.
- Keeps some diversity while maintaining stability.

### **Weaknesses**

- Fixed k does not adapt to distribution shape (e.g. whether tail is flat or steep).

### **Exam skill**

- Given probabilities and a **k**, you must show:
  - the truncated list
  - renormalised probabilities
  - sampled token.

---

# **6. Top-p Sampling (Nucleus Sampling)**

### **What it is**

- Select the **smallest** subset of tokens whose cumulative probability ≥ **p**, then sample.

### **How it works**

1. Sort tokens by probability.
2. Compute cumulative sum.
3. Take the **nucleus** where cumulative ≥ p (e.g. p=0.9).
4. Renormalise and sample.

### **Strengths**

- **Adaptive** to distribution shape.
- Avoids both:
  - long-tail noise
  - overly restrictive fixed-k behaviour.

### **Weaknesses**

- May still allow unpredictable outputs if distribution is flat.
- More difficult to compute manually (but still required in exams).

### **Exam requirement**

- Show **which tokens enter the nucleus**, renormalise, then sample.

---

# **7. Summary Table (Useful for Exam Comparisons)**

| Method          | Deterministic?    | Diversity   | Computation | Typical Failure                 |
| --------------- | ----------------- | ----------- | ----------- | ------------------------------- |
| **Greedy**      | Yes               | Low         | Very low    | Local optimum; repetition       |
| **Beam search** | Yes (for fixed k) | Low–Med     | High        | Length bias; generic outputs    |
| **Sampling**    | No                | High        | Low         | Chaotic, incoherent outputs     |
| **Top-k**       | No                | Medium      | Low         | k not adaptive                  |
| **Top-p**       | No                | Medium–High | Low         | Sensitive to distribution shape |

---

# **TLDR**

- **Greedy:** take argmax. Fast but gets stuck.
- **Beam search:** keep top-k sequences. Higher quality but expensive.
- **Sampling:** draw from full distribution. Diverse but noisy.
- **Top-k:** sample from k best tokens. Controls noise.
- **Top-p:** sample from smallest cumulative-p set. Adaptive + diverse.

# 6. Algorithms & Computational Methods

# **1. Byte-Pair Encoding (BPE)**

## **What BPE is used for**

- A **subword tokenisation algorithm**.
- Solves the **unknown word** problem by breaking rare words into frequent subword units.
- Reduces vocabulary size while keeping expressive power.
- Used in GPT, RoBERTa, many modern LLMs.

---

## **How the BPE algorithm works (steps)**

### **Training phase**

1. Start with training text split into **characters** (plus a word boundary symbol, often `</w>`).
2. Count **all symbol pairs** (adjacent characters/subwords).
3. Find the **most frequent pair**.
4. **Merge** that pair into a new symbol.
5. Replace all occurrences in the corpus.
6. Repeat **N merges** (hyperparameter).

### **Inference (encoding)**

- Apply stored merge rules **in order**, merging greedily until no further merges apply.

---

## **Hand-simulation example (exam essential)**

### **Corpus**

`low lowish`

### **Initial segmentation**

`l o w </w> l o w i s h </w>`

### **Step 1: Count pairs**

- `l o`: 2
- `o w`: 2
- `w </w>`: 1
- `w i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent pair: **`l o`** or **`o w`** (tie; either is acceptable in exam).  
Assume we merge **`l o → lo`**.

### **After merge**

`lo w </w> lo w i s h </w>`

### **Step 2: Recount pairs**

- `lo w`: 2
- `w </w>`: 1
- `w i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent: **`lo w` → low**.

### **After merge**

`low </w> low i s h </w>`

### **Step 3: Recount**

- `low </w>`: 1
- `low i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent: **`i s` → is**.

### **After merge**

`low </w> low is h </w>`

### **Step 4: Recount**

- `is h`: 1
- `h </w>`: 1

Most frequent: **`is h` → ish**.

### **Final segmentation of words**

- **low**
- **low + ish**

### **Key exam point**

- You must show **pair counts**, **merge selection**, and **updated corpus** at each step.
- **Initial vocab = all characters + end-of-word.**
- **Each merge adds one new symbol.**
- **Stop when |V| reaches target.**

---

# **2. Backpropagation**

## **What backpropagation is used for**

- Computes **gradients** of the loss w.r.t. all model parameters in a neural network.
- Enables **Stochastic Gradient Descent (SGD)** or Adam to update weights.
- Essential for training any neural model: MLPs, CNNs, RNNs, Transformers.

---

## **Core steps of backpropagation**

### **1. Forward pass**

- Compute outputs layer by layer.
- Compute loss $L$ (e.g. cross-entropy).

### **2. Backward pass**

For each layer from top to bottom:

#### **a. Compute local derivatives**

- For linear layer:  
   $z = Wx + b$,  
   $\frac{\partial z}{\partial W} = x^\top$,  
   $\frac{\partial z}{\partial x} = W^\top$.
- For activation $a = f(z)$:  
   Multiply by **f'(z)**.

#### **b. Apply chain rule**

- **upstream gradient x local derivative = downstream gradient**
- Gradient to propagate further.

#### **c. Accumulate gradients w.r.t. parameters**

- Store $\partial L / \partial W$ and $\partial L / \partial b$.

### **3. Update step**

For each parameter $\theta$:
$$\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$$
where $\eta$ is the learning rate.

---

## **Minimal worked example (exam-ready)**

For a 1-layer network:

### **Forward**

- $z = Wx + b$
- $y = \text{softmax}(z)$
- Loss: $L = -\log y_{target}$

### **Backprop**

1. $\frac{\partial L}{\partial z} = y - t$ (where $t$ is one-hot target)
2. $\frac{\partial L}{\partial W} = (y - t)\,x^\top$
3. $\frac{\partial L}{\partial b} = y - t$
4. $\frac{\partial L}{\partial x} = W^\top (y - t)$

Because (chain rule)

$\frac{\partial L}{\partial z} = y - t$

$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}} = \delta_i \cdot x_j$

$\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_i} = \delta_i$

$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j} = (W^\top \delta)_j$

#### **Example**

Input: $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

Weights:

$W = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}$

Bias:

$b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$

True label:

$\text{t or y} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

Forward pass:

$z = Wx = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
$y=softmax(z)=[0.730.27]y = \text{softmax}(z) = \begin{bmatrix} 0.73 \\ 0.27 \end{bmatrix}$
$L = -\log 0.73$

---

### ⭐ 1. **Compute $\frac{\partial L}{\partial z}$​** (error signal)

Softmax + cross-entropy has the known closed-form:

$\frac{\partial L}{\partial z_i} = y_i - t_i$

#### **Plug in values**

$\frac{\partial L}{\partial z} = \begin{bmatrix} 0.73 - 1 \\ 0.27 - 0 \end{bmatrix} = \begin{bmatrix} -0.27 \\ 0.27 \end{bmatrix}$

We call this:

$\delta = \frac{\partial L}{\partial z}$

---

### ⭐ 2. **Compute $\frac{\partial L}{\partial W}$​** (using chain rule)

Each weight affects the loss only through **z**, so:

$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}}$

We already know:

- $\frac{\partial L}{\partial z_i} = \delta_i$
- $z_i = \sum_j W_{ij} x_j \Rightarrow \frac{\partial z_i}{\partial W_{ij}} = x_j$

So:

$\frac{\partial L}{\partial W_{ij}} = \delta_i x_j$

#### **Now compute each number manually**

$\partial L/\partial W = \begin{bmatrix} 0.73 \\ 0.27 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

$\frac{\partial L}{\partial W} = \begin{bmatrix} -0.27 & -0.54 \\ 0.27 & 0.54 \end{bmatrix}$

---

### ⭐ 3. **Compute $\frac{\partial L}{\partial b}$​**

Bias affects z additively:

$z_i = W_i x + b_i \Rightarrow \frac{\partial z_i}{\partial b_i} = 1$

So:

$\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i} \cdot 1 = \delta_i$

### With numbers:

$\frac{\partial L}{\partial b} = \begin{bmatrix} -0.27 \\ 0.27 \end{bmatrix}$

---

#### ⭐ 4. **Compute $\frac{\partial L}{\partial x}$​**

Again use chain rule:

$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j}$

But:

$z_i = W_{i1} x_1 + W_{i2} x_2 \Rightarrow \frac{\partial z_i}{\partial x_j} = W_{ij}$

So in vector form:

$\frac{\partial L}{\partial x} = W^\top \delta$

#### **Compute numerically**

Transpose:

$W^\top = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}$

Now multiply:

$\frac{\partial L}{\partial x} = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix} \begin{bmatrix}-0.27 \\ 0.27\end{bmatrix}$

$\frac{\partial L}{\partial x} = \begin{bmatrix} 0.27 \\ -0.27 \end{bmatrix}$

Exam versions often give you a **tiny input vector**, **tiny weight matrix**, and ask for one forward + backward step.

---

**Some key notes:**

- _“Why does BPTT cause memory issues?”_
  - BPTT is for RNNs: storing every intermediate hidden state and activation for every timesept grows linearly with sequence length.
- Backprop for Transformers
  - For sequence N, memory for attention is $O(N^2L)$
  - QKV, softmax weights also scales with Nxd
- _“How can you avoid exploding gradients?”_
  **Solutions**: gradient clipping, truncated BPTT, gating units (LSTM/GRU).
- _“What happens to gradients through repeated multiplication?”_
  - **Vanishing**: repeated multiplication by values <1 shrinks gradients.
  - **Exploding**: repeated multiplication by values >1 grows gradients.

# **Summary**

## **BPE**

- Used for **subword tokenisation**.
- Steps: **count pairs → merge most frequent → repeat**.
- Must be able to **simulate manually**.

## **Backpropagation**

- Used for **gradient computation** in neural networks.
- Steps: **forward pass → compute loss → backward pass using chain rule → update**.

---

# **TLDR**

- **BPE:** iterative pair-merging subword algorithm; know how to run it by hand.
- **Backprop:** chain rule applied layer-by-layer to compute gradients; central to training neural nets.

# 7. Additional Maths Concepts

---

# **ADDITIONAL MATHEMATICAL & COMPUTATIONAL CONCEPTS — REVISION NOTES**

---

## **1. Zipf’s Law & Sparse Data**

### **Zipf’s Law**

- **Definition:** In natural language, the **frequency** of a word is **inversely proportional to its rank**:
  $$f(r) \propto \frac{1}{r}$$
- **Implication:**
  - A **few words** occur extremely often (e.g., _the, of, and_).
  - **Most words** occur **very rarely** → **long tail** distribution.
- **Why it matters:**
  - Models trained with MLE will **underfit** or produce **0-probability** for rare words.
  - Forces need for **smoothing**, **subword models** (BPE), and **large corpora**.

### **Sparse Data**

- **Meaning:** Many valid word combinations **never appear in training**, even in huge corpora.
- **Consequences for NLP tasks:**
  - **Language modelling:** N-grams fail for unseen sequences → smoothing needed.
  - **Parsing:** Rare constructions weaken model probability estimates.
  - **Classification:** Rare features produce unstable estimates.

---

## **2. Training / Development / Test Sets**

### **Purpose**

- **Training set:** Fit model parameters.
- **Development (validation) set:**
  - Tune hyperparameters (learning rate, smoothing constant, model depth).
  - Early stopping.
- **Test set:**
  - Final **unbiased estimate** of generalisation.
  - Must **never** influence model decisions.

### **Why they matter**

- **Avoid overfitting**
  - Distinct splits ensure improvements are not just memorisation.
- **Hyperparameter control**
  - Dev set provides feedback without contaminating test performance.
- **Fair model comparison**
  - Competitors must evaluate on the **same untouched test distribution**.
- **Detect domain shift**
  - If dev ≠ test domain, tuning may generalise poorly — common in NLP (e.g., Wikipedia → Twitter).

### **Task examples**

- **Language modelling:** Dev set used to monitor **perplexity**.
- **Classification:** Dev set guides regularisation strength.
- **LLM finetuning:** Dev set prevents model collapse during SFT.

- **Using test set for hyperparameter tuning**  
   → **data leakage**, inflated scores, invalid evaluation.
- **Randomly mixing genres across splits** (e.g., training on Wikipedia, testing on Twitter)  
   → leads to unexpected ↑ perplexity / ↓ accuracy.
- **Too small dev set**  
   → unstable hyperparameter decisions.
- **Training and dev from different distributions**  
   → hyperparameters tuned for the wrong domain.
- **Overfitting visible on dev before test**  
   → solved via early stopping.

---

## **3. LLM Development Phases**

### **Pre-training**

- **Objective:** Learn broad linguistic and world knowledge via
  - **Causal LM (next-token prediction)**
  - **Masked LM (MLM)**
  - **Denoising**
- **Data:** Massive, diverse, often web-scale.

### **Post-training**

1. **SFT (Supervised Fine-Tuning):**

   - Learn to follow instructions; dataset contains **input → desired output**.

2. **RLHF (Reinforcement Learning from Human Feedback):**

   - Model generates responses → human/ranker provides preference → **reward model** → policy optimisation (PPO-like).

3. **RLVR (Reinforcement Learning from Verifiable Rewards):**

   - Similar to RLHF but uses **automatically verifiable signals** (e.g., maths correctness, constraint satisfaction).

### **Fine-tuning (task-specific)**

- Smaller, focused datasets.
- Examples: sentiment classifier, legal summarisation model, domain-specific chatbot.

---

## **4. LLM Inference Phases**

### **Initial KV Cache Creation (“prefill”)**

- Input prompt is passed through the model **once** to initialise **Key/Value activations** for attention.
- Speeds up generation: future tokens only attend to cached vectors, not recompute entire sequence.

### **Auto-regressive Decoding**

- At each step, the model:
  1. Uses KV cache
  2. Computes next-token logits
  3. Applies **decoding strategy** (greedy, sampling, top-k, top-p).
- Continues until EOS token or length limit.

---

# **5. Regular Expressions (Regex)**

### **Definition**

A compact notation for specifying **patterns over strings**.

### **Examples**

- `\d{3}-\d{2}-\d{4}` → US-style number pattern.
- `^[A-Z][a-z]+$` → Capitalised word.

### **Relevant NLP tasks**

- **Tokenisation**
- **Preprocessing / cleaning**
- **Pattern extraction** (dates, emails)
- **Rule-based NER**

### **Why important**

Still widely used despite deep learning; fundamental to text pipelines.

---

# **6. Sparse vs. Dense Word Representations**

## **Sparse Representations**

- **Definition:** High-dimensional vectors with mostly **zero** entries.
  - Example: **one-hot vectors**, **bag-of-words**.
- **Pros:**
  - Simple, interpretable.
- **Cons:**
  - No notion of **similarity** or **semantic structure**.
  - Huge dimensionality → inefficient.
  - Sparse data problem persists.

## **Dense Representations (Embeddings)**

- **Definition:** Low-dimensional, continuous vectors learned by models.
- **Properties:**
  - Capture **semantic similarity** (e.g., _king_ and _queen_ close in vector space).
  - Enable downstream models to generalise.

### **Relevant NLP tasks**

- **LMs**, **classification**, **translation**, **similarity search**, **retrieval**.

---

# **7. Vector-Based Similarity Measures**

### **Dot Product**

- Measures alignment: $\vec{a} \cdot \vec{b}$.
- Large positive = similar direction.

### **Cosine Similarity**

- **Scale-invariant** measure of angle:
  $$
  \cos(\theta) = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}
  $$
- Most common for embeddings.

### **Euclidean Distance**

- Measures absolute distance in space.
- Sensitive to embedding magnitude.

### **Task examples**

- **Nearest-neighbour word similarity**
- **Sentence retrieval**
- **Clustering** (e.g., semantic categories)

---

# **8. Sentence-level Embeddings**

### **Definition**

Represent entire sentences as **single dense vectors**.

### **Methods**

- **Pooling over token embeddings** (mean/max).
- **Sentence Transformers** (e.g., SBERT).
- **LLM hidden-state averaging**.

### **Uses**

- **Semantic similarity**
- **Clustering / retrieval**
- **Information retrieval (IR)**
- **Document classification**
- **Duplicate question detection** (e.g., Quora dataset)

### **Why important**

They make long-text comparison **computationally tractable** without expensive cross-attention.

---

# **TL;DR**

- **Zipf’s Law** → rare events dominate; leads to **sparse data**, motivates **smoothing** & **subword models**.
- **Train/dev/test** → prevent overfitting; evaluate generalisation.
- **LLM development** = _pre-training → SFT → RLHF → RLVR → fine-tuning_.
- **Inference** = _KV cache prefill → autoregressive decoding_.
- **Regex** → rule-based string patterns, still essential.
- **Sparse vs dense embeddings** → dense vectors encode semantics; sparse are simple but limited.
- **Similarity metrics** → cosine sim is dominant.
- **Sentence embeddings** → enable semantic search, retrieval, classification.

# 8. Linguistic and representational concepts

## **1. Ambiguity (multiple forms)**

**Core idea:** A single surface form can map to multiple interpretations.  
**Why it matters:** NLP systems must resolve ambiguity to correctly parse, classify, translate, or answer questions.

### **Types**

- - **Lexical ambiguity (word sense):**  
    _bank_ = _financial institution_ vs _river edge_.  
    Relevant to: **WSD**, MT, IR.
- **Parts of speech ambiguity:**  
   _flies_ = noun (“insects”) or verb (“he flies”).  
   Relevant to: **tagging**, morphological analyzers.
- Morphological ambiguity:
  - "Untiable" = able to be untied OR not able to be tied?
- **Syntactic (structural) ambiguity:**  
   _I saw an elephant in my pyjamas._  
   (Attachment of PP unclear)
- **Attachment ambiguity:**  
   _I saw the man with the telescope._  
   Relevant to: **parsing**, semantic role labelling.
- **Word-order ambiguity:**  
   _Old men and women._  
   Relevant to: **parsing**, MT.
- **Referential ambiguity**
  _Mary told Jane she should leave._ (“she” unclear)
- **Idiomatic ambiguity:**  
   _kick the bucket_ → literal vs idiomatic.

---

## **2. Agreement**

**Core idea:** Certain grammatical features must match across elements.

### **Types of agreement**

- **Verb–argument agreement:**
  - _She runs_ vs _They run_.
  - Relevant to: **syntactic parsing**, LM scoring.
- **Co-reference agreement:**
  - Pronouns must match antecedent number/gender: _Mary said she…_
  - Relevant to: **coreference resolution**.
- **Language-specific agreement features:**
  - **Case** (German, Russian), **gender** (Spanish), **number**.
  - Relevant to: MT, parsing, morphology modelling.

### **Long-distance dependencies**

- Agreement can span clauses:
  - _The bouquet of roses **was** lovely._
  - Relevant to: assessing **context windows**, identifying **limitations of n-gram models**.

---

## **3. Word Types vs Tokens & Tokenization**

- **Tokens:** individual surface occurrences in text.
  - _“the cat sat”_ → 3 tokens.
- **Types:** unique word forms in a corpus.
  - In that sentence, 3 types (all unique).

**Tokenization issues:**

- Hyphens, contractions (_don’t → do + n’t_), multiword expressions.
- Relevant to: **BPE**, vocabulary construction, embeddings.

---

## **4. Stems, Affixes, Root, Lemma**

- **Root:** irreducible lexical core (_scrib-_).
- **Stem:** form to which affixes attach (_scrib(e)_).
- **Affixes:** prefixes/suffixes (_un-_, _-ing_).
- **Lemma:** dictionary form (_run_ vs _running_, _ran_).

**Relevant to:** morphological analyzers, MT, IR normalisation.

---

## **5. Inflectional vs Derivational Morphology**

- **Inflectional:** alters grammatical features; **does not change word class**.
  - _walk → walked_ (tense), _cat → cats_ (number).
  - Relevant to: POS tagging, parsing.
- **Derivational:** forms **new lexemes**, often **changes word class**.
  - _happy → happiness_, _teach → teacher_.
  - Relevant to: vocabulary growth, embeddings, MT.

**Case, gender, number marking:**

- Languages vary: Romance (gender), Slavic (case), English (mostly number/tense).

Note: Morphological richness → **data sparsity**, more word forms → harder for n-grams, embeddings, tagging.

---

## **6. Dialects**

- Variation in vocabulary, syntax, spelling, morphology.
- Variety by region, class or culture. typically mutually intelligable.
- Examples: _colour/color_, _you all/y’all_.
- Relevant to: LM robustness, MT, speech-to-text, fairness.

---

## **7. Part-of-Speech (POS)**

- Grammatical categories (noun, verb, adj…).
- Important because POS constrains syntactic structure.
- Ambiguity common: _book_ (noun/verb).
- Relevant to: tagging, parsing, downstream tasks.

---

## **8. Open-class vs Closed-class Words**

- **Open-class:** nouns, verbs, adjectives, adverbs.
  - Semantically rich; new words appear.
  - Important for embeddings, semantics.
- **Closed-class:** determiners, prepositions, pronouns, conjunctions.
  - Rarely grow; grammatical glue.
  - Important for syntax modelling.

---

## **9. Long-Distance Dependencies**

- Dependencies spanning large distances or intervening material.
- Examples:
  - Subject–verb agreement: _The key to the cabinets **is** missing._
  - Wh-movement: _What did John say Mary bought \_\_?_

**Relevance:**

- N-grams fail (limited context).
- Transformers handle via self-attention.

---

## **10. Syntactic Roles**

- **Subject:** agent/performer.
- **Object:** undergoer/patient.
- **Indirect object:** recipient or benefactive.

**Why needed:** semantic role labelling, parsing, MT disambiguation.

---

## **11. Word Senses & Semantic Relations**

- **Synonym:** _big / large_.
- **Hypernym:** _animal_ (hypernym of _dog_).
- **Hyponym:** _poodle_ (hyponym of _dog_).
- **Similarity:** distributional or conceptual closeness.

**Relevance:** WSD, MT, IR, embeddings.

---

## **12. Distributional Hypothesis**

**Definition:**  
_Words that occur in similar contexts tend to have similar meanings._

- Forms basis of **word embeddings**, co-occurrence matrices, skip-gram/CBOW.
- Relevant to: semantics, clustering, similarity tasks.

---

## **13. Static vs Contextualized Embeddings**

### **Static embeddings**

- One vector per word type (e.g., **word2vec**, **GloVe**).
- Cannot capture polysemy (_bank_ has one vector).
- Trained using distributional context.

### **Contextual embeddings**

- One vector per **token** in context (e.g., **ELMo**, **BERT**, **GPT**).
- Capture polysemy, subtle syntactic/semantic relations.
- Generated via **deep language models** during inference.

**Relevance:** nearly all modern NLP (NER, MT, QA, sentiment).

---

# **TLDR**

- Ambiguity drives most NLP difficulty; know types and examples.
- Agreement involves number/gender/case dependencies; can be long-distance.
- Tokens ≠ types; tokenization matters for BPE and embeddings.
- Morphology: roots, lemmas, inflection vs derivation.
- POS, syntactic roles, open/closed classes guide parsing.
- Word senses + semantic relations feed into WSD, MT, IR.
- Distributional hypothesis underpins embeddings.
- Static embeddings = type-level; contextual embeddings = token-level, context-sensitive.

# 9. Tasks and Task Structures

# **1. Tokenization**

**What it is:**

- Splitting raw text into **tokens** (words, subwords, punctuation).
- Defines the _basic units_ of all downstream NLP models.

**Examples:**

- “don’t” → {“don”, “’”, “t”} (char-based)
- BPE: “unhappiness” → “un”, “happi”, “ness”

**Ambiguity / Difficulty:**

- **Multi-word expressions:** “New York”, “hot dog”.
- **Agglutinative languages:** very long words (e.g., Turkish).
- **No whitespace languages:** Chinese, Japanese → segmentation is non-trivial.

**Algorithms / Methods:**

- Rule-based tokenizers, whitespace tokenizers.
- **Subword models**: BPE, WordPiece, SentencePiece.

**Evaluation:**

- Intrinsically rare; typically evaluated indirectly via downstream task performance.

**Task Structure:**

- **Preprocessing step**, not a prediction task.

---

# **2. Language Modelling (LM)**

**What it is:**

- Predicting the **next token** given previous context:  
   **P(wₜ | w₁ … wₜ₋₁)**.

**Examples:**

- Predict the next word: “The cat sat on the \_\_\_”.

**Ambiguity / Difficulty:**

- **Long-distance dependencies:** LM must capture grammar agreement, discourse cues.
- **Sparsity:** Rare n-grams severely weaken count-based models.
- **Unbounded vocabulary / OOV:** mitigated by subword tokenization.

**Algorithms / Methods:**

- **N-gram models (+ smoothing).**
- **Neural LMs:** RNN, LSTM, GRU.
- **Transformers:** GPT-style causal LMs.

**Evaluation:**

- **Perplexity** (exp of average negative log-likelihood).
- Intrinsic, task-specific.

**Task Structure:**

- **Sequence prediction** (autoregressive).
- **Generative model.**

---

# **3. Text Categorization (Sentiment, Topic Classification, etc.)**

**What it is:**

- Assigning one label (or set of labels) to a text span.
- Examples: **Sentiment** (positive/negative), **Spam detection**, **Topic** (sports, politics).

**Ambiguity / Difficulty:**

- **Sarcasm / Irony:** “Great job…” (negative sentiment).
- **Domain shift:** model trained on movie reviews performs poorly on finance tweets.
- **Multi-label vs single-label:** deciding task structure affects modelling.

**Algorithms / Methods:**

- Logistic regression, linear SVMs.
- Bag-of-words, TF-IDF, embeddings → classifiers.
- Fine-tuned transformers (BERT, RoBERTa).

**Evaluation:**

- **Accuracy**, **Precision/Recall/F1**, **Confusion matrix**.
- Macro-F1 for imbalanced classes.

**Task Structure:**

- **Classification** (single label) or **multi-label classification**.

---

# **4. Word Sense Disambiguation (WSD)**

**What it is:**

- Assigning the correct **sense** of a polysemous word in context.
- Example: “bank” = _riverbank_ vs _financial institution_.

**Ambiguity / Difficulty:**

- **Lexical ambiguity** is large and context-dependent.
- Senses in resources (WordNet) may not align with real usage.
- Many senses are extremely rare → sparse data problem.

**Algorithms / Methods:**

- Knowledge-based: Lesk algorithm (overlap).
- Supervised classifiers using context windows.
- Transformer contextual embeddings (state-of-the-art).

**Evaluation:**

- Sense-level **accuracy** vs a gold dataset (e.g., SemCor).

**Task Structure:**

- **Classification** over senses of a single target word.

---

# **5. Sequence-to-Sequence Tasks**

_(Machine translation, summarization, data-to-text, style transfer)_

**What they are:**

- Input sequence → **output sequence**, often of different length.
- Examples:
  - **Machine Translation:** “Je suis fatigué” → “I am tired.”
  - **Summarization:** long article → short abstract.

**Ambiguity / Difficulty:**

- **Multiple correct outputs** → evaluation is hard.
- **Alignment** between source and target tokens is implicit and complex.
- **Long-range reasoning** required (especially summarization).
- **Hallucinations** in generative models.

**Algorithms / Methods:**

- Seq2Seq RNNs w/ attention (Bahdanau, Luong).
- Transformer encoder–decoder (e.g., T5, BART).
- Causal LMs with prompt-based decoding.

**Evaluation:**

- **BLEU**, **ROUGE**, **METEOR** (n-gram overlap).
- Increasingly: human evaluation, BERTScore.

**Task Structure:**

- **Sequence-to-sequence generation**, conditional generation.

---

# **6. Open-Ended Conversational AI**

**What it is:**

- Systems that generate **contextually appropriate, multi-turn dialogue**.
- Not a closed classification task—output space is unbounded.

**Examples:**

- Chatbots, customer support assistants, tutoring systems.

**Ambiguity / Difficulty:**

- **Ambiguous intent**: user messages underspecified.
- **Long context tracking**: maintaining state across turns.
- **Safety & grounding**: misinformation, hallucinations, harmful responses.
- **Evaluation** inherently subjective.

**Algorithms / Methods:**

- Large Language Models (GPT-style).
- Retrieval-augmented generation (RAG).
- Reinforcement learning (e.g., RLHF, RLVR).

**Evaluation:**

- Hard! Typically:
  - **Human judgments** (quality, relevance, safety).
  - **Automated metrics** (BLEU, embedding similarity) but imperfect.
  - **Task-specific** scoring (goal completion in task-oriented agents).

**Task Structure:**

- **Open-ended generation** (not finite-label).
- Often modeled as **sequence-to-sequence** or **autoregressive LM** with memory.

---

# **7. Identifying Task Structures (Meta-Skill)**

You should be able to recognise, for a **newly described task**, which structure it fits:

| Task                            | Structure                              |
| ------------------------------- | -------------------------------------- |
| POS tagging                     | **Sequence labelling**                 |
| Named entity recognition        | **Sequence labelling**                 |
| Spam detection                  | **Classification**                     |
| MT / Summarization              | **Seq2Seq**                            |
| Dialogue generation             | **Open-ended generation**              |
| Question answering (extractive) | **Span prediction**                    |
| Coreference resolution          | **Clustering / structured prediction** |

Key cues:

- **Is the output one label?** → classification.
- **Label per token?** → sequence labelling.
- **Output is a new sequence?** → seq2seq.
- **Output is unconstrained free text?** → generative LM task.

---

# **TLDR — Core Expectations**

- Know **what each task is**, with **examples**, **ambiguities**, **difficulty sources**.
- Know **typical algorithms**: N-grams, logistic regression, transformers, seq2seq.
- Know **evaluation metrics**: accuracy, F1, perplexity, BLEU/ROUGE.
- For **new tasks**, quickly identify if it's **classification / seq-labelling / seq2seq / open-gen**.

# merged

## **1. What types of resources are needed?**

### **A. Labeled Data**

Used when a task requires _supervised learning_.

- **Examples:** POS-tagged corpora, dependency treebanks, NER datasets.
- **Relevant tasks:** tagging, parsing, NER, sentiment classification.
- **Pros:** high-quality signal, task-specific.
- **Cons:** expensive, slow to obtain, annotation inconsistency.

---

### **B. Unlabeled Corpora**

Large-scale text collected from the web or domain sources.

- **Examples:** **CommonCrawl**, Wikipedia, BooksCorpus.
- **Relevant tasks:** LM pretraining, unsupervised embeddings, self-supervised learning.
- **Pros:** very large, cheap, useful for representation learning.
- **Cons:** noisy, biased, legality of scraping unclear.

---

### **C. Lexical & Semantic Resources**

Hand-curated or semi-curated structured databases.

- **Examples:**
  - **WordNet** (synsets, hypernyms).
  - **FrameNet**, **PropBank** (semantic roles).
  - **VerbNet** (verb classes).
- **Relevant tasks:** WSD, semantic similarity, SRL, lexicon-based sentiment.
- **Pros:** interpretable, high precision.
- **Cons:** expensive to build; limited domain coverage; may encode cultural bias.

---

### **D. Morphological Resources**

Used for languages with rich morphology.

- **Examples:** CELEX, UniMorph.
- **Relevant tasks:** morphological parsing, MT, lemmatization.
- **Pros:** reliable structured forms.
- **Cons:** incomplete across languages.

---

### **E. Multilingual & Parallel Corpora**

Aligned text across languages.

- **Examples:** EuroParl, UN Parallel Corpus.
- **Relevant tasks:** MT (alignment models, NMT training), cross-lingual embeddings.
- **Pros:** essential for MT; structured alignment.
- **Cons:** limited domains (parliament, UN), not representative of everyday language.

---

### **F. Evaluation Benchmarks**

Standardised test suites to compare systems.

- **Examples:** GLUE, SuperGLUE, SQuAD, CoNLL shared tasks.
- **Relevant tasks:** QA, NER, coreference, reasoning.
- **Pros:** comparability across models.
- **Cons:** overfitting to benchmarks; narrow task framing.

---

### **G. Human Expertise / Annotators**

Humans provide labels, guidelines, and quality checks.

- **Examples:** Crowdworkers on MTurk; linguists building treebanks.
- **Relevant tasks:** any supervised task.
- **Pros:** high-quality if trained.
- **Cons:** annotation bias, cost, ethical concerns.

---

## **2. Pros & Cons — High-Level Summary**

### **Pros**

- Resources provide structure, ground truth, or massive raw data enabling **learning**.
- Curated lexicons improve **interpretability**.
- Large-scale corpora enable **scaling** of LLMs.
- Benchmarks enable **comparability** and **progress tracking**.

### **Cons**

- Data scarcity in low-resource languages.
- Annotation is expensive and inconsistent.
- Web data is noisy and may contain harmful content.
- Lexicons can be outdated or culturally narrow.
- Benchmark-driven development encourages overfitting and ignores real-world use.

---

## **3. Legal & Ethical Issues to Identify**

### **A. Copyright & Licensing**

- Web text (CommonCrawl) often includes copyrighted material.
- Training use vs distribution use may be legally distinct.
- Some corpora prohibit _commercial_ use.

### **B. Consent**

- Many scraped datasets include text _not intended_ for ML training.
- Private messages, social networks, or forum posts may include personal data.

### **C. Personal Data & Privacy**

- GDPR issues for EU subjects.
- Presence of names, addresses, sensitive attributes.
- Risks: deanonymisation, model memorisation.

### **D. Bias & Representation Harm**

- Large web corpora encode stereotypes (gender, race, dialect).
- Under-representation of minority dialects → model underperformance.
- Lexicons often reflect Western, academic linguistic assumptions.

### **E. Toxicity & Harmful Content**

- Hate speech, misinformation, extremist content in web data.
- Models can reproduce harmful patterns unless filtered.

### **F. Worker Ethics**

- Low-paid annotators exposed to traumatising content.
- Unclear guidelines or inadequate compensation for crowdworkers.

### **G. Transparency & Documentation**

- Need for **datasheets**, **model cards**, provenance metadata.
- Failure to document → misuse, risk, unclear bias sources.

---

## **4. Typical Exam Angles**

You may be asked to:

- Compare resources (e.g., WordNet vs CommonCrawl).
- Identify which resource a task needs and justify why.
- Describe limitations of a given dataset.
- Discuss legal/ethical risks in collecting new data.
- Explain how resource quality affects model performance.

---

# **TLDR**

- Know **resource types** (labeled, unlabeled, lexical, morphological, parallel, benchmarks).
- Know **examples** (WordNet, CommonCrawl, FrameNet, EuroParl).
- Understand their **pros/cons** (coverage, cost, noise, bias).
- Be able to identify **legal/ethical issues** (copyright, privacy, bias, informed consent, annotator welfare).

## **1. Perplexity (PPL)**

How **uncertain** a language model is when predicting the next token.

**What it measures:**

- Perplexity is the model’s **effective average number of choices** at each step.
  - Perplexity ≈ 2 → model is very confident (only ~2 likely next words).
  - Perplexity ≈ 100 → model is very uncertain (many plausible next words).
  - **Lower perplexity = better model (less confused).**
- **Example:** If a model has cross-entropy $H_M = 3$ bits per word, then $PP = 2^3 = 8$
  - “On average, the model behaves like it has about **8** plausible next-word options.”

**Appropriate for:**

- **Language modelling**, next-word prediction, generative pretraining.
- **Model comparison:**
  - Unigram → high perplexity (no context); Bigram → lower; Trigram → lower still
  - More context → fewer effective choices → **lower perplexity**.

**Why:**

- Directly reflects model’s predictive uncertainty.
- Task-agnostic measure of fluency.

**Limitations:**

- Not aligned with human judgments for downstream tasks.
- Cannot compare models with different vocabularies/tokenization schemes.

---

## **2. Accuracy**

**What it measures:**

- % of predictions that are correct.
  $$\text{accuracy} = \frac{\text{num correct predictions}}{\text{num total predictions}}$$
- Flaw:\_ Misleading if classes are **unbalanced** (e.g., a spam detector that always predicts "not spam" might be 90% accurate but useless).

**Appropriate for:**

- **Classification** tasks with balanced labels:
  - POS tagging (when balanced), sentiment analysis, NLI.
- Good for single-label prediction

**Why:**

- Simple metric when classes are roughly balanced and task is closed-form.
- Interpretable, task-aligned
- Clear, stable signal during model development
- Easy to compare across models using same datasets and labels

**Limitations:**

- Misleading for **imbalanced classes** (e.g., rare NER labels, spam detection).
  - If 95% of emails are non-spam, a model predicting "not-spam" always gets 95% accuracy but is useless
- Conceals type-specific errors:
- Cannot express partial credit
  - Predicting the righ category but wrong subcalss recieves zero

---

## **3. Precision, Recall, F-measure**

### **Precision**

**Meaning:** Of all the items the system _predicted as positive_, how many were **actually correct**?  
**Formula:**

$\text{Precision} = \frac{TP}{TP + FP}$

**When it matters:**

- **False positives are costly.**
- You prefer **conservatism** over overgeneration.

**Examples:**

- **NER:** Avoid labelling non-entities as entities.
- **Toxic content detection:** Don’t wrongly flag harmless text.

---

### **Recall**

**Meaning:** Of all the _true_ positives that exist, how many did the system **actually find**?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**When it matters:**

- **Missing items is costly.**
- You prefer **coverage** over precision.

**Examples:**

- **Coreference resolution:** Missing links breaks downstream tasks.
- **Information extraction:** Better to capture all mentions for analysis.

---

### **F1-score (F-measure)**

**Meaning:** The **harmonic mean** of precision and recall.

$$F_1 = 2 \cdot \frac{PR}{P + R}$$

**Why harmonic mean?**

- It **penalises imbalance** (e.g., high precision but terrible recall).
- Encourages a model that is **jointly good** at both.

**When it shines:**

- **Sparse labels** (e.g., NER entities are rare).
- **Imbalanced classes** (many negatives, few positives).
- **Token-level structure prediction** where TP/FP/FN matter more than TN.

---

# **4. Metrics for Generative Tasks**

## **BLEU (Machine Translation)**

**What BLEU measures**

- **N-gram precision**: how many n-grams in the candidate also appear in the reference(s).
- **Geometric mean** of 1-gram … 4-gram precisions.
- **Brevity Penalty (BP)**: reduces score if the system output is **too short**.

**Why appropriate**

- Captures **local phrase correctness** (short, frequent MT errors).
- Historically correlated **moderately** with human translation quality.
- Works well when translations are **literal** and n-gram overlap is meaningful.

**Limitations**

- **Surface-form bias**: penalises valid paraphrases that differ lexically
- Weak on **semantics**, **discourse**, **gender agreement**, **style**.
- High BLEU ≠ good translation if system “games” n-gram overlap.

---

## **ROUGE (Summarisation)**

**What ROUGE measures**

- **ROUGE-N:** recall of n-gram overlap
- **ROUGE-L:** **Longest Common Subsequence (LCS)** — measures shared ordering.
- Recall-oriented because we care about whether summaries **cover important content**.

**Why appropriate**

- Summaries must capture **key information**; ROUGE focuses on whether important words/phrases were retrieved.
- Historically correlated with **human judgments** for extractive summaries.

**Limitations**

- Insensitive to **coherence**, **fluency**, **logical structure**.
- Cannot detect **hallucinated content** not present in the source.
- Overly rewards **extractive copying**, under-rewards paraphrasing.

---

## **Other Metrics**

### **METEOR**

- Aligns words using **synonyms**, **stemming**, **paraphrase tables**.
- Better recall–precision balance than BLEU.

### **BERTScore**

- Computes similarity using contextual embeddings.
- Captures **semantic similarity**, not just surface overlap.

### **chrF**

- Character-level F-score.
- Great for **morphologically rich languages** (Slavic, Finnish, Turkish).
- More robust to tokenisation artefacts.

---

## **5. LLM-as-a-Judge**

**What this evaluates**

- An LLM scores or ranks system outputs using **task-specific rubrics** (e.g., correctness, fluency, helpfulness).
- Used for: **summaries**, **QA**, **dialogue**, **code correctness**, **reasoning tasks**.

**Why appropriate**

- Models can evaluate **meaning**, **style**, **faithfulness**, **reasoning steps** — dimensions overlap metrics cannot capture.
- **Scalable**, **cheap**, **fast** → useful for large evaluations (thousands of samples).
- Closer to human preference judgments than BLEU/ROUGE.

**Limitations / Concerns**

- **Bias:** judges prefer outputs that match their own stylistic priors.
- **Non-transparency:** unclear internal criteria.
- **Reward hacking:** systems may optimise for judge quirks, not true quality.
- Requires **calibration**, comparisons to **human-annotated** sets, and ideally **multi-judge** ensembles.

---

## **6. Win Rate & Elo Ranking**

### **Win Rate**

**Definition**

- Given two model outputs (A vs B), a judge (human or LLM) chooses the better one.
- **Win rate = P(A beats B)** across many samples.

**Why useful**

- Simple, robust for tasks with **no single correct answer** (dialogue, creative tasks).
- Directly measures **preference** rather than accuracy.

---

### **Elo Ranking**

**Definition**

- Converts pairwise preferences into a **global skill score** for multiple models.
- Analogous to **chess ratings**: each match updates both players’ Elo.

**Why appropriate**

- Produces a **stable global ordering** of many models.
- Handles **non-absolute quality** (no gold truth).
- More expressive than raw win rate.

**Limitations**

- Assumes approximate **transitivity** (if A > B and B > C → A > C).
- Sensitive to judge biases, sampling strategy, and match difficulty.
- No absolute meaning: Elo is **relative** to the pool of models tested.

---

# **7. Intrinsic vs Extrinsic Evaluation**

## **Intrinsic Evaluation**

**Definition**  
Evaluate a model **component** directly, outside any downstream task.

**Examples**

- **Perplexity** on a corpus (language modelling).
- **Word similarity** tasks for embeddings (SimLex, WordSim).
- **Parsing accuracy** (LAS/UAS) on annotated treebanks.

**Pros**

- Fast, cheap, diagnostic.
- Allows controlled experiments (change one component at a time).

**Cons**

- **Weak correlation** with end-task performance.
- Encourages optimisation for metrics that do not reflect **real utility**.

---

## **Extrinsic Evaluation**

**Definition**  
Evaluate a model by how well it improves **downstream task performance**.

**Examples**

- Better embeddings → higher **NER F1**.
- Improved LM → stronger **QA** or **MT** accuracy.
- Stronger parser → better **information extraction**.

**Pros**

- Measures **actual task usefulness**.
- Captures interactions between components.

**Cons**

- Expensive: requires full pipeline training.
- Hard to interpret: performance changes may come from multiple factors.
- Noisy due to hyperparameters and system design.

---

# **8. Corpora: Collection, Annotation, Distribution Issues**

## **Collection Issues**

- **Sampling bias:**
  - Source skews (news vs Twitter vs Wikipedia) → changes style, dialects, topics.
  - Leads to models that generalise poorly to real-world data.
- **Privacy & consent:**
  - Risk of collecting PII, medical content, minors' data.
  - GDPR and global privacy regimes impose strict constraints.
- **Domain mismatch:**
  - Training on one domain (news) but testing on another (medical) → large performance drop.

---

## **Annotation Issues**

- **Inter-annotator agreement (IAA):**
  - Measures reliability; use κ (kappa) or α (alpha).
  - Low IAA → task fundamentally ambiguous or guidelines unclear.
    - Noisy ground truth, test set evaluation is misleading
- **Quality control:**
  - Annotator training, gold-check questions, adjudication by experts.
- **Cost:**
  - High-quality annotation (e.g., NER, SRL, discourse) requires expertise.
- **Bias:**
  - Annotators bring cultural assumptions → impacts sentiment, toxicity, emotion labels.

---

## **Distribution Issues**

- **Licensing constraints:**
  - Some corpora cannot be redistributed (copyright).
  - Limits reproducibility.
- **Data documentation:**
  - Datasheets / model cards → ensure transparency about collection, cleaning, biases.
- **Legal risk:**
  - Copyright violation, defamation risk, GDPR obligations.
- **Fairness considerations:**
  - Underrepresentation of minority dialects → models fail for underrepresented users.
  - Reinforces socio-linguistic inequalities.

---

# **TLDR**

- **Perplexity** → LM predictive quality.
- **Accuracy** → overall correctness (balanced classes).
- **Precision/Recall/F1** → imbalanced tasks, extraction tasks.
- **BLEU/ROUGE** → generative tasks, based on n-gram overlap.
- **LLM-as-judge** → semantic & quality eval, but biased.
- **Win rate & Elo** → preference-based ranking of LLMs.
- **Intrinsic vs extrinsic** → component-only vs task-based evaluation.
- **Corpora issues** → bias, consent, privacy, annotation quality, licensing.

# **Ethical Issues in NLP — Revision Notes**

---

## **1. Algorithmic Bias**

**Definition:**  
Systematic, unfair performance differences across demographic groups due to **biased data**, **biased models**, or **unequal error rates**.

### **Sources of bias**

- **Training data bias:**
  - Over-representation of Standard American English → poor performance on dialects.
  - Toxicity classifiers mislabel AAVE sentences as “offensive.”
- **Label bias:**
  - Annotators bring cultural assumptions → sentiment or hate-speech datasets skewed.
- **Measurement bias:**
  - Metrics that fail to capture performance differences (accuracy hides minority errors).

### **Implications for tasks**

- MT may produce gender-stereotyped translations (_doctor → he_, _nurse → she_).
- Coreference systems may misresolve pronouns for unrepresented groups.
- Speech or text models may fail on dialects or low-resource languages.

### **Mitigation**

- Diverse data sampling; bias audits; counterfactual data augmentation.
- Group-specific evaluation metrics (per-group F1).
- Explicit fairness constraints during training.

---

## **2. Direct vs Indirect Discrimination**

### **Direct discrimination**

**Definition:**  
Model _explicitly_ uses a protected attribute (e.g., gender, race) to make decisions.

**Examples:**

- Sentiment classifier that assigns lower positivity to names associated with specific ethnic groups.
- Hiring model penalising applicants with “female” as a feature.

**Mitigation:**

- Remove protected attributes; enforce feature-dropout; independent bias checks.

---

### **Indirect discrimination (a.k.a. disparate impact)**

**Definition:**  
A model uses **proxy features** that correlate with protected attributes, even without explicit reference.

**Examples:**

- ZIP code predicting socioeconomic status or ethnicity.
- Word embeddings encoding stereotypes captured from biased corpora.
- MT systems assigning stereotyped gender pronouns due to corpus bias.

**Mitigation:**

- Detect and reduce proxy correlations; adversarial training; fairness-aware loss functions.
- Redesign features to remove protected-attribute leakage.

---

## **3. Representational vs Allocational Harm**

### **A. Representational Harm**

**Definition:**  
When a system **reinforces negative stereotypes**, erases identities, or misrepresents groups.

**Examples:**

- Associating Muslim names with terrorism in word embeddings.
- Autocomplete suggesting harmful or biased continuations.
- MT translating gender-neutral forms into stereotyped gender roles.

**Relevance:**  
Affects perception, identity, and social narratives — even when no resource allocation is involved.

**Mitigation:**

- Debiasing embeddings; curated safe-text filters; red-team evaluations.
- Inclusive dataset design; culturally aware annotation protocols.

---

### **B. Allocational Harm**

**Definition:**  
When a system causes **unequal access to opportunities, services, or material resources**.

**Examples:**

- Credit scoring models assigning lower credit limits to speakers of certain dialects.
- Automated hiring tools privileging certain linguistic styles or backgrounds.
- Health-care NLP triage system misclassifying symptoms for minority groups.

**Relevance:**  
Material consequences → financial, medical, educational disparities.

**Mitigation:**

- Fairness constraints; group fairness metrics; post-hoc calibration.
- Regulatory oversight; transparency documentation; algorithmic audits.

---

# **4. How to use these concepts in exam scenarios**

You may be asked to:

- Analyse a dataset/model pipeline and identify **types of harm**.
- Explain whether an issue is **representational** or **allocational** (or both).
- Describe fairness risks for an example task (e.g., MT for public-service signs).
- Suggest **mitigations** appropriate to the task and resource type.
- Connect ethical risks to **dataset issues** (consent, bias, documentation) and **evaluation** issues (per-group metrics).

---

# **TLDR**

- **Algorithmic bias** = unequal model behaviour due to skewed data, labels, or training.
- **Direct discrimination** = explicit use of protected attributes.
- **Indirect discrimination** = proxy features cause unequal impact.
- **Representational harm** = stereotypes & misrepresentation.
- **Allocational harm** = unequal access to resources & opportunities.
- **Mitigate with:** better data, fairness metrics, audits, debiasing, documentation, and inclusive design.

# **Multilingual NLP — Revision Notes**

---

## **1. Data Paucity**

**Core idea:**  
Most of the world’s languages have **little or no labeled data**, and many have limited unlabeled corpora.

### **Why it matters**

- Many NLP models assume **large-scale corpora**, which low-resource languages lack.
- Leads to **poor performance**, unstable training, and biased multilingual systems.

### **Typical problems**

- Sparse morphology → harder to learn inflectional forms.
- High dialectal variation with small datasets.
- Non-standard or no orthography.

### **Relevance to tasks**

- MT, POS tagging, NER, ASR — all suffer when training data is small.
- Encourages methods like **transfer learning**, **cross-lingual embeddings**, **few-shot learning**.

---

## **2. Multilingual LLMs**

**Definition:**  
A single model jointly trained on text from **many languages**, sharing parameters and often a shared subword vocabulary (BPE/SentencePiece).

### **Advantages**

- **Parameter sharing** enables cross-lingual transfer.
- Low-resource languages benefit from **shared semantics** captured via high-resource languages.
- Models can perform **zero-shot** generation or classification.

### **Challenges**

- **Vocabulary imbalance**: high-resource languages dominate the subword vocabulary.
- **Interference**: languages compete for model capacity.
- **Script issues**: different scripts create uneven coverage.

### **Examples**

- mBERT, XLM-R, BLOOM, multilingual GPT-class models.

### **Relevant tasks**

- MT, cross-lingual QA, NER, document classification, embedding similarity.

---

## **3. Zero-shot Cross-linguistic Transfer**

**Definition:**  
Model trained on a task in one language (e.g., English) performs the same task **in an unseen language** without additional training.

### **Why it works**

- Shared embeddings and shared model layers build **language-neutral representations**.
- Based on the **distributional hypothesis across languages**.

### **When it works well**

- Languages with **similar scripts** and **similar syntactic structures**.
- Tasks relying on **semantics** more than fine-grained morphology.

### **When it fails**

- Languages structurally distant (e.g., English → Inuktitut).
- Rich morphology with low training support.

---

## **4. Translate-Train & Translate-Test**

### **Translate Train**

**Process:**

1. Translate training data from a high-resource language (HRL) → low-resource language (LRL).
2. Train model directly on synthetic LRL data.

**Pros:**

- Produces an LRL-specific model.
- Good when MT into LRL is accurate.

**Cons:**

- Translation errors propagate into training.
- Expensive to generate large synthetic corpora.

---

### **Translate Test**

**Process:**

1. Test input in LRL is translated → HRL.
2. Apply HRL-trained model.
3. Optionally translate output back.

**Pros:**

- No need to train a dedicated LRL model.
- Works well if MT is good from LRL → HRL.

**Cons:**

- MT errors during inference cause evaluation noise.
- Semantic drift is common.

---

### **Zero-shot vs Translate-Train/Test — Summary**

- **Zero-shot:** no translation; relies fully on model’s internal shared representations.
- **Translate-train:** creates training data in the target language.
- **Translate-test:** avoids retraining; uses translation pipeline at inference.

---

## **5. Multilingual Evaluation**

### **Challenges**

- **Benchmark imbalance:** HRLs have high-quality datasets; LRLs often do not.
- **Cultural + domain mismatch:** evaluation content may not transfer across languages well.
- **Script diversity:** tokenization and vocabulary create uneven performance baselines.

### **Common evaluation strategies**

- **Multilingual benchmarks:** XTREME, XGLUE, AmericasNLI.
- **Per-language breakdowns:** accuracy/F1 reported for each language.
- **Transfer tests:** train on HRL, test on LRL.
- **Code-switching performance:** test robustness in mixed-language input.

### **What evaluators look for**

- Stability across languages.
- Whether performance drops correlate with data size (often they do).
- Bias against minority dialects or scripts.

---

# **TLDR**

- **Data paucity** drives the challenge: most languages have minimal training data.
- **Multilingual LLMs** share parameters across languages, enabling **transfer** but risking interference.
- **Zero-shot transfer** uses shared semantics; good when languages are similar.
- **Translate-train** adds synthetic LRL training data; **translate-test** uses translation at inference.
- **Multilingual evaluation** must compare performance _per language_, account for script bias, and use multilingual benchmarks.

## 🔁 Big Picture: What You Must Do For _Any_ Model

For **each model** below, you should be able to:

- **Compute**
  - **P(sequence)** or **one forward step** (e.g. next-word prob, class prob, hidden state update).
  - For **linear / log-linear models**: dot product + nonlinearity (sigmoid/softmax).
- **Count parameters**
  - **Exactly** for small models (given vocab sizes, hidden sizes, etc.).
  - **Approximately** for LLMs (e.g. embeddings + layers × per-layer params).
- **Explain training**
  - What **data** is used, what **objective** (loss) is optimised, and roughly how **gradient descent** updates parameters.
- **Explain smoothing / regularisation**
  - Why it is needed (overfitting / zeros), and when it matters most.
- **Map to tasks**
  - Given a model, say which **task types** it’s suited for (classification / seq2seq / language modelling / representation learning) and **how** it’s applied.
- **Analyse what the model can/can’t capture**
  - What linguistic phenomena it can represent; where it fails (e.g. long-distance deps, ambiguity).
- **Compare pros/cons**
  - For a given task, explain why you might pick one model over another.

Keep that meta-template in mind while revising each model.

---

## 1️⃣ N-gram Models

**Core idea**

- **N-gram assumption**:  
   $$P(w_i \mid w_1,\dots,w_{i-1}) \approx P(w_i \mid w_{i-N+1},\dots,w_{i-1})$$.  
   Only **last N−1 words** matter (Markov assumption).

**Compute probability / forward step**

- Sentence probability:  
   $P(w_1^T) = \prod_{i=1}^T P(w_i \mid w_{i-N+1}^{i-1})$.
- You should be able to:
  - Use **counts**: $P(w_i \mid h) = \dfrac{C(h,w_i)}{C(h)}$ (MLE).
  - Apply **smoothing** (e.g. add-α, backoff) when counts are zero.

**Parameters**

- Number of parameters ≈ **number of distinct N-grams** with non-zero probability.
- Upper bound: $|V|^N$ parameters (huge for large N).
- You should be able to count params for **small V, small N** exactly.

**Training**

- **Data**: Text corpus (tokens).
- **Objective**: Maximise **likelihood** (or minimise **cross-entropy**) of training data.
- **Procedure**: Count N-grams → compute probabilities (maybe with smoothing).

**Smoothing**

- Needed because many N-grams **never appear** in training → zero probability.
- **Most important** when:
  - Data is **small** or
  - Vocab is **large** or
  - N is **big** (e.g. tri-/4-grams).
- Examples: **Add-α**, **Kneser–Ney**, **backoff**, **interpolation**.

**Typical tasks**

- **Language modelling** (next-word prediction, perplexity).
- Baseline for **speech recognition**, **MT**, **spell correction**, etc.

**What it can / can’t capture**

- **Can**:
  - Local word patterns, collocations, short-range dependencies.
- **Fails at**:
  - **Long-distance dependencies** (“if … then …”).
  - **Global semantics**, **discourse**, **world knowledge**.
  - Generalising to unseen contexts beyond smoothed interpolation.

**Pros / cons**

- **Pros**: Simple, interpretable, fast for small N, easy to compute.
- **Cons**: Data-hungry, sparse, poor at long-range syntax/semantics, doesn’t share parameters across similar contexts.

**Extra exam point (generative process)**

- **Generative story**:
  1. Pick first token(s) from start-of-sentence distribution.
  2. For each position (i), **sample $(w_i)$** from $P(w_i \mid w_{i-N+1}^{i-1})$.
  3. Continue until EOS token.
- Joint probability of a **sequence** (no latent vars in basic N-gram):  
   $P(w_1^T) = \prod_i P(w_i \mid w_{i-N+1}^{i-1})$.
- If the question mentions **latent variables**, you might be expected to **write a generic form**  
   $P(x,z) = P(z)P(x\mid z)$ and relate to any extended N-gram variant discussed (e.g. tags).

---

## 2️⃣ Logistic Regression (Binary)

**Core idea**

- **Linear classifier** with **sigmoid** output:  
   $P(y=1 \mid x) = \sigma(w^\top x + b)$.
- Used for **binary classification** (e.g. positive vs negative).

**Compute step**

- Given **weights**, **bias**, **feature vector**, compute:
  - **Score**: $s = w^\top x + b$.
  - **Probability**: $P(y=1 \mid x) = \dfrac{1}{1+e^{-s}}$.
- In exam: you may **not** have to compute $e^s$, but must:
  - **Set up** the expression
  - Say which class has **higher probability** based on comparing scores.

**Parameters**

- For input with **d features**: **d weights + 1 bias**.
- If you treat it as logistic for $y∈{0,1}$, that’s it.

**Training**

- **Data**: pairs $(x^{(i)}, y^{(i)})$.
- **Objective**: maximise **log-likelihood** (equiv. minimise **cross-entropy loss**).
- **Training**: gradient descent / variants, with **regularisation** (L2, etc.).

**Regularisation**

- To prevent **overfitting**, especially with **many features** (e.g. bag-of-words).
- Common: **L2** penalty $\lambda |w|^2$.
- Most important when number of features ≫ number of examples.

**Typical tasks**

- Binary **sentiment analysis** (pos/neg).
- Binary **spam detection**.
- **Any** yes/no classification where features are interpretable.

**What it can / can’t capture**

- **Can**: linear decision boundaries in feature space; works well with **good features**.
- **Cannot**: model **interactions** or **nonlinear** relations unless features encode them; no sequence structure.

**Pros / cons**

- **Pros**: Interpretable, convex training objective, relatively easy.
- **Cons**: Limited expressivity; relies heavily on **feature engineering**.

---

## 3️⃣ Multinomial Logistic Regression (Softmax Regression)

**Core idea**

- Generalises logistic regression to **K > 2** classes with **softmax**:  
   $P(y=k \mid x) = \dfrac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)}$.

**Compute step**

- Given features ($x$) and class weight vectors ($w_k$):
  - Compute **scores** $s_k = w_k^\top x + b_k$.
  - Compute **softmax probabilities** using those scores.

**Parameters**

- For **d features** and **K classes**:
  - Weights: $K \times d$, biases: $K$.
  - Total params: $K \cdot d + K$.

**Training & regularisation**

- Same story as logistic, but with **multiclass cross-entropy loss**.
- Regularisation again important when many features.

**Tasks & features**

- **POS tagging per token** (if you treat each token independently).
- **Topic classification**, **intent classification**, etc.
- Features: bag-of-words, n-gram counts, lexical features, etc.

**Extra exam requirements**

- **You must be able to**:
  - **Write the softmax formula** clearly.
  - Given weights/features, **identify most probable class** (highest score or highest unnormalised logit).
  - Reason about how **changing a weight** would affect class probabilities.

---

## 4️⃣ Skip-gram with Negative Sampling (Word2Vec)

**Core idea**

- A **neural model** to learn **word embeddings** by predicting **context words** from a **target word** (Skip-gram).
- Negative sampling approximates full softmax by contrasting **true context words** vs **sampled negatives**.

**Architecture**

- **Input**: one-hot word → **embedding lookup** (vector $v_w$).
- **Output**: separate **context embedding** $u_c$.
- **Scoring**: $u_c^\top v_w$ → passed through sigmoid for positive vs negative pairs.

**Compute step**

- Given **target embedding** $v_w$ and **context embedding** $u_c$:
  - Score: $s = u_c^\top v_w$.
  - Positive pair probability (binary logistic): $\sigma(s)$.

**Parameters**

- Two embedding matrices (often):
  - **Input embeddings**: $|V| \times d$.
  - **Output/context embeddings**: $|V| \times d$.
- Total: $≈ 2|V|d$ params.

**Training**

- **Data**: text → pairs (target, context) within a window.
- **Objective**: for each positive pair:
  - maximise $\log \sigma(u_c^\top v_t) + \sum_{neg} \log \sigma(-u_{n}^\top v_t)$.
- Uses **stochastic gradient descent**.

**Regularisation**

- Mainly **implicit** via:
  - Limited embedding size
  - Negative sampling distribution.
- You can also use **L2** on embeddings, but often not emphasised.

**Tasks**

- Not a task model per se; it **learns representations** for use in:
  - Downstream classifiers (sentiment, NER, etc.).
  - Similarity tasks, analogies, nearest neighbours.

**What it can / can’t capture**

- Captures **distributional semantics** (“you know a word by the company it keeps”).
- Fails at:
  - Sentence/sequence structure; it’s bag-of-context windows.
  - Word sense disambiguation (single vector per type).

**Pros / cons**

- **Pros**: Fast, effective embeddings, simple.
- **Cons**: No contextualisation, static word meaning, limited to co-occurrence.

---

## 5️⃣ Multilayer Perceptron (Feed-forward Network)

**Core idea**

- Fully connected **layers** with **nonlinear activations** (e.g. ReLU, tanh):
  - $h = f(Wx + b)$,
  - $y = g(Uh + c)$ (for classification, g=softmax/sigmoid).

**Compute step**

- Given small dimensions, you should be able to:
  - Multiply input by weight matrix, add bias.
  - Apply nonlinearity (ReLU, tanh).
  - Apply final linear + softmax for class probs.

**Parameters**

- For each layer with **input dim in**, **hidden dim out**:
  - Weights: in × out, biases: out.
- Total params = **sum over layers**.

**Training & regularisation**

- **Objective**: cross-entropy for classification; MSE or others for regression.
- **Training**: backpropagation + gradient descent.
- Regularisation: **L2**, **dropout**, **early stopping**, etc.

**Tasks**

- **Classification** from fixed-size features (e.g. sentence embeddings → sentiment).
- **Regression** tasks (e.g. scoring).

**What it can / can’t capture**

- **Can**: complex **nonlinear mappings** from features to labels.
- **Cannot**: handle **variable-length sequences** directly (unless you summarise first); no explicit temporal structure.

**Pros / cons**

- **Pros**: flexible, universal approximator.
- **Cons**: needs good features, no sequence inductive bias.

---

## 6️⃣ Recurrent Neural Network (RNN)

**Core idea**

- Processes **sequences** step-by-step, maintaining a **hidden state** $h_t$:  
   $h_t = f(W_x x_t + W_h h_{t-1} + b)$.

**Compute step**

- Given small dimensions, you must be able to:
  - Start with $h_0$ (often zeros).
  - Compute $h_1$ from $x_1, h_0$;
  - Then $h_2$, etc.
  - Optionally compute output $y_t = g(W_y h_t + c)$ (e.g. softmax over vocab).

**Parameters**

- For vanilla RNN:
  - Input→hidden: $W_x$ (dim: $d_{in} \times d_h$).
  - Hidden→hidden: $W_h$ ($d_h \times d_h$).
  - Hidden→output: $W_y$ ($d_h \times d_{out}$).
  - Plus biases.
- Total = sum of these matrices + biases.

**Training & regularisation**

- **Objective**: e.g. next-token cross-entropy for LM.
- **Training**: backpropagation through time (BPTT).
- Regularisation: dropout on hidden states, gradient clipping, etc.

**Tasks**

- **Language modelling**, sequence classification, tagging, simple MT (with encoder-decoder RNN variants).

**What it can / can’t capture**

- **Can**: some **longer context** than N-grams, sequential patterns.
- **Cannot (well)**: very long dependencies (vanishing/exploding gradients); parallelisation is hard.

**Pros / cons**

- **Pros**: sequence-aware, more expressive than N-grams.
- **Cons**: training difficulties, slower than Transformers for long sequences.

---

## 7️⃣ RNN with Attention (Seq2Seq + Attention)

**Core idea**

- **Encoder RNN** reads input → sequence of hidden states.
- **Decoder RNN** generates output, at each step **attending** to encoder states via attention mechanism.

**Attention computation**

- For decoder state $s_t$ and encoder states $h_1,\dots,h_T$:
  - Scores: $e_{t,i} = \text{score}(s_t, h_i)$ (dot, MLP, etc.).
  - Weights: $\alpha_{t,i} = \text{softmax}_i(e_{t,i})$.
  - Context: $c_t = \sum_i \alpha_{t,i} h_i$.
  - Decoder uses $[s_t ; c_t]$ to predict next token.

**Parameters**

- Encoder RNN params + decoder RNN params + **attention parameters** (for score function).
- You should be able to count these for a small example.

**Tasks**

- **Seq2seq**: machine translation, summarisation, etc.

**What it can / can’t capture**

- **Can**: explicitly model **alignments**, focus on specific input tokens.
- **Still limited by**: sequential processing, training stability, long sequences vs Transformer.

---

Gotcha — let’s clean this up so the LaTeX is actually readable.

I’ll redo the **Transformer notes with correct MathJax**, keeping:

- Your **structure** (8️⃣ / 9️⃣ / 🔟, A–F sections, etc.)
- The **worked examples** in the style you showed
- All the **math properly typeset**

---

# 8️⃣ **Transformer Encoder-Only (BERT-style)**

---

## **Core idea**

- Stack of **self-attention + feedforward** blocks with **residual connections** and **LayerNorm**.
- Processes the whole sequence **in parallel** → outputs **contextualised representations** for all tokens.

---

## **A. Computational / Forward Step**

Let input be a matrix
$X \in \mathbb{R}^{N \times d}$ (N tokens, model dimension $d$).

### 1. Linear projections

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V
$$

where

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}.
$$

### 2. Self-attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Call the result $A \in \mathbb{R}^{N \times d_k}$

### 2.5 Output Projection

$$A_{proj} = AW_O$$
Where $A_{proj} \in \mathbb{R}^{N \times d}$ and $W_O \in \mathbb{R}^{d_k \times d}$

Each head has a dimension $d_k$ which $= \frac{d}{h}$ with $h$ heads, the concatenated output is $hd_k = d$. $W_O$ mixes head outputs back to the correct shape to be able to have layerNorm + Residuals applied.

### 3. Residual + LayerNorm (attention block)

$$
X' = \text{LayerNorm}(X + A_{proj}).
$$

### 4. Feed-forward network (FFN, per token)

$$
H = \phi(X' W_1 + b_1) W_2 + b_2,
$$

where

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$,
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$.

### 5. Residual + LayerNorm (FFN block)

$$
X_{\text{out}} = \text{LayerNorm}(X' + H).
$$

You should be able to do this **for tiny toy shapes** (e.g. $N=2, d=2, d_k=2$).

---

## **B. Parameter Counting (Per Layer)**

Let:

- model dim: $d$
- head dim: $d_k$
- FFN inner dim: $d_{\text{ff}}$
- single head for simplicity.

### Attention projections

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k},\quad
W_O \in \mathbb{R}^{d_k \times d}.
$$

Number of **attention weights**:

$$
3 \cdot d \cdot d_k + d_k \cdot d = 4 d d_k.
$$

### Feed-forward network

$$
W_1 \in \mathbb{R}^{d \times d_{\text{ff}}},\quad
W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}.
$$

Number of **FFN weights**:

$$
d \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d
= 2 d d_{\text{ff}}.
$$

Biases:

- $b_1 \in \mathbb{R}^{d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d}$ → add $d_{\text{ff}} + d$ if needed.

### LayerNorm

Two LayerNorms per layer, each with scale and bias:

- each LayerNorm: $2d$ params
- per layer: $4d$.

### Embeddings (global, not per-layer)

- Token embeddings: $|V| \times d$.
- Positional embeddings (if learned): $N_{\max} \times d$.

---

## **C. Training + Regularisation**

### Training (BERT-style)

- **Masked Language Modelling (MLM)**:
  mask some tokens and predict them from **bidirectional context**.
- Sometimes **Next Sentence Prediction** (NSP) objective.

Optimisation:

- Cross-entropy loss on masked positions.
- Backprop through the whole network.
- Optimiser: some variant of SGD (e.g. AdamW).

### Regularisation

- **Dropout** on attention outputs and FFN outputs.
- **LayerNorm** for stabilising activations.
- **Weight decay** (L2-style) on parameters.

---

## **D. Typical Tasks**

- **Sentence/document classification** (use [CLS] token).
- **Token classification** (NER, POS, chunking).
- **Span-based QA**.
- **NLI, paraphrase detection, similarity**.

All framed as **“encode input → add small head → predict labels”**.

---

## **E. What It Can / Can’t Capture**

### Can

- **Bidirectional context** for each token.
- **Long-distance dependencies** via global self-attention.
- Rich semantic representations.

### Can’t

- Autoregressive generation (without adding a generative head / procedure).
- Very long sequences efficiently (attention is $O(N^2)$ in sequence length).

### Typical failure modes

- On very long sequences, attention may become **diffuse**.
- Positional encodings trained for max length $L$ may **not extrapolate** beyond $L$.
- Still struggles with **deep hierarchical syntax** (e.g. many nested clauses).

---

## **F. Pros / Cons**

**Pros**

- Excellent for **understanding** tasks.
- Parallel over tokens (good GPU utilisation).
- Powerful and expressive.

**Cons**

- Quadratic time/memory in sequence length.
- Not naturally generative.
- Pretraining is compute- and data-heavy.

---

## **G. Positional Encodings (Exam-Specific)**

You should know:

### Absolute positional encodings

- **Sinusoidal** (fixed):
  deterministic functions of the position $p$ and dimension index $i$.
- **Learned absolute** (BERT):
  trainable embedding $P_p \in \mathbb{R}^d$ added to token embeddings.

### Relative positional encodings

- Encode **offset** between positions (e.g. $j-i$), not their absolute indices.
- Often implemented via **added biases** to attention scores or shifted embeddings.
- Help generalise to **different sequence lengths** and capture relative order.

---

## **H. Scaling Laws (Exam-Specific)**

High-level ideas:

- **Parameter scaling**:
  For $L$ layers, model dimension $d$, feedforward dim $d_{\text{ff}}$,

  $$
  \text{params} \approx L \cdot (c_1 d^2 + c_2 d d_{\text{ff}})
  $$

  (for constants $c_1, c_2$ depending on heads etc.).

- **Performance scaling**:
  Empirically, loss often follows a **power law** in **data size**, **model size**, **compute**.

- Doubling $d$ tends to **quadruple** param count; doubling $L$ increases linearly.

You don’t need exact exponents, just this **qualitative scaling story**.

---

## **I. Interpreting Attention Weights (Exam-Specific)**

Possible interpretations:

- Some heads focus on **syntactic relations** (e.g. subject ↔ verb).
- Others track **coreference** (pronouns ↔ antecedents).
- Some track **relative positions** or punctuation.

Caveats:

- Attention weights do **not necessarily equal explanations**.
- Many heads are **diffuse** or encode information that’s not human-interpretable.

---

## **Worked Example 1 — Self-Attention (One Query, Full Weights)**

This is exactly in the style you gave.

We take 3 tokens, dim = 2, and assume projections already done:

$$
Q = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix},\quad
K = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix},\quad
V = \begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}.
$$

We compute attention for **token 1**, with query
$q_1 = [1, 0]$.

### Scores

$$
e_i = q_1 \cdot k_i
$$

- $e_1 = [1,0] \cdot [1,0] = 1$
- $e_2 = [1,0] \cdot [0,1] = 0$
- $e_3 = [1,0] \cdot [1,1] = 1$

So the score vector is:

$$
z = [1, 0, 1].
$$

### Softmax

$$
\alpha_i = \frac{\exp(z_i)}{\exp(1) + \exp(0) + \exp(1)}
= \frac{\exp(z_i)}{2e + 1}.
$$

So:

$$
\alpha_1 = \alpha_3 = \frac{e}{2e + 1},\quad
\alpha_2 = \frac{1}{2e + 1}.
$$

### Attended vector

$$
\text{attn}(q_1) = \sum_i \alpha_i v_i
= \alpha_1 [1,0] + \alpha_2 [0,1] + \alpha_3 [1,1].
$$

Compute component-wise:

$$
\text{attn}(q_1)
= \begin{bmatrix}
\alpha_1 + \alpha_3 \\
\alpha_2 + \alpha_3
\end{bmatrix}
= \begin{bmatrix}
\frac{2e}{2e+1} \\
\frac{1 + e}{2e+1}
\end{bmatrix}.
$$

No need for numeric approximation — the **structure** is what matters.

---

## **Worked Example 2 — Encoder Param Count (1 Layer)**

Exactly in your requested style.

> **Q:** Approximate parameter count for a 1-layer encoder with
> • vocab size $V = 10$
> • model dim $d_{\text{model}} = 4$
> • feed-forward dim $d_{\text{ff}} = 8$
> • 1 attention head

### Embedding layer

Token embeddings:

$$
V \times d_{\text{model}} = 10 \times 4 = 40.
$$

(We’ll ignore positional embeddings here.)

### Self-attention (single head)

- $W_Q: 4 \times 4 = 16$
- $W_K: 4 \times 4 = 16$
- $W_V: 4 \times 4 = 16$

So

$$
3 \times 16 = 48.
$$

Output projection:

$$
W_O: 4 \times 4 = 16.
$$

Total attention weights:

$$
48 + 16 = 64.
$$

### Feed-forward

- $W_1: 4 \times 8 = 32$
- $W_2: 8 \times 4 = 32$

Total FFN:

$$
32 + 32 = 64.
$$

### Total (ignoring biases & LayerNorm)

$$
40\ (\text{emb}) + 64\ (\text{attn}) + 64\ (\text{FFN}) = 168\ \text{parameters}.
$$

---

Below are **fully expanded, encoder-level versions** of both **decoder-only** and **encoder–decoder** transformers.
Format, depth, and sectioning **exactly match** your encoder template.

---

# 9️⃣ **Transformer Decoder-Only (GPT-style)**

(Updated to be as detailed as encoder)

---

# **Core idea**

- Same building blocks as encoder: **self-attention + FFN**, each wrapped with **residuals + LayerNorm**.
- BUT with a **causal mask** so token _t_ only attends to tokens ≤ _t_.
- Trained on **autoregressive LM objective**:

$$
P(x) = \prod_{t=1}^N P(x_t \mid x_{<t}).
$$

---

# **A. Computational / Forward Step**

Input sequence (with positions added):

$$
X \in \mathbb{R}^{N \times d}.
$$

---

## **1. Linear projections**

Exactly as in encoder:

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V,
$$

with

$$
W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}.
$$

---

## **2. Causal self-attention**

Compute raw attention scores:

$$
S = \frac{QK^\top}{\sqrt{d_k}}.
$$

Apply **causal mask**:

$$
S_{ij} =
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

Softmax row-wise:

$$
A = \text{softmax}(S)V
\quad\in\mathbb{R}^{N \times d_k}.
$$

Interpretation:
**Row i** attends only to **columns 1..i**.

---

## **2.5 Output projection**

$$
A_{\text{proj}} = A W_O
\quad \text{where } W_O \in \mathbb{R}^{d_k \times d}.
$$

---

## **3. Residual + LayerNorm**

$$
X' = \text{LayerNorm}(X + A_{\text{proj}}).
$$

---

## **4. Feed-Forward Network (FFN)**

Same as encoder:

$$
H = \phi(X' W_1 + b_1) W_2 + b_2,
$$

with

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$

---

## **5. Residual + LayerNorm**

$$
X_{\text{out}} = \text{LayerNorm}(X' + H).
$$

This is final hidden state for each token.

---

## **6. LM head for generation**

Take **last token hidden state** ($h_t$):

$$
\text{logits} = h_t W_E^\top,
$$

where embeddings are often **tied**:

$$
W_E \in \mathbb{R}^{|V| \times d}.
$$

Softmax → probability over next token.

---

# **B. Parameter Counting (Per Layer)**

Identical to encoder except **no cross-attention**.

### Attention projections

$$
W_Q, W_K, W_V: d \times d_k,\quad
W_O: d_k \times d
$$

Total:

$$
4d d_k.
$$

### Feed-forward

$$
2d d_{\text{ff}}.
$$

### LayerNorm

Two LayerNorms →

$$
4d.
$$

### Embeddings

- Token: $|V| \times d$
- Positional: $N_{\max} \times d$

### LM Head

Often **tied** → no extra params.

---

# **C. Training + Regularisation**

### **Training objective**

$$
\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t})
$$

- True past tokens fed in (“**teacher forcing**”).
- Standard cross-entropy over vocabulary.

### **Regularisation**

- Dropout (attention weights, FFN output).
- Weight decay (L2).
- LayerNorm stabilisation.

---

# **D. Typical Tasks**

Everything framed as **next-token prediction**:

- Text continuation / generation.
- Summarisation (as generation).
- Translation (as generation).
- Retrieval-augmented tasks.
- Classification via instruction prompting.

---

# **E. What It Can / Can’t Capture**

### **Can**

- Strong generative expressivity.
- Long-range context (within window).
- Few-shot / zero-shot via prompting.

### **Can’t**

- Bidirectional context in a single forward pass.
- Efficient very-long-sequence computation.

---

# **F. Pros / Cons**

### **Pros**

- Amazing generative capabilities.
- Simplest architecture (one stack).
- Natural for in-context learning.

### **Cons**

- No backward context.
- Quadratic attention cost.
- Pretraining compute cost huge.

---

# 1️⃣0️⃣ **Transformer Encoder–Decoder (T5 / Seq2Seq)**

(Updated to match full encoder detail)

---

# **Core idea**

Two stacks:

- **Encoder**: exactly like encoder-only BERT (bidirectional self-attn).
- **Decoder**: GPT-style masked self-attn **plus cross-attention** to encoder output.

Used for sequence-to-sequence mapping:

$$
x_{1..N} \to y_{1..M}.
$$

---

# **A. Computational / Forward Step**

We track shapes explicitly.

---

## **1. Encoder stack**

Input:

$$
X \in \mathbb{R}^{N \times d}.
$$

For each layer:

1. **Self-attn** (bidirectional):

   $$
   Q = XW_Q,\quad K = XW_K,\quad V = XW_V
   $$

   $$
   A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
   $$

2. **Residual + LayerNorm**:

   $$
   X' = \text{LayerNorm}(X + A W_O).
   $$

3. **FFN**:

   $$
   H = \phi(X'W_1 + b_1)W_2 + b_2.
   $$

4. **Residual + LayerNorm**:
   $$
   H_{\text{enc}} = \text{LayerNorm}(X' + H).
   $$

Output of encoder:

$$
H_{\text{enc}} \in \mathbb{R}^{N \times d}.
$$

---

## **2. Decoder stack (per layer)**

Input decoder tokens:

$$
S \in \mathbb{R}^{M \times d}.
$$

---

### **2.1 Masked self-attention (decoder)**

Same as GPT:

$$
Q_s = S W_Q^{(\text{self})},\quad
K_s = S W_K^{(\text{self})},\quad
V_s = S W_V^{(\text{self})}.
$$

Causal mask on:

$$
S_s = \frac{Q_s K_s^\top}{\sqrt{d_k}}
\quad\text{with } j > i \text{ masked}.
$$

Then:

$$
A_s = \text{softmax}(S_s)V_s.
$$

Residual + LN:

$$
S' = \text{LayerNorm}(S + A_s W_O^{(\text{self})}).
$$

---

### **2.2 Cross-attention (key difference)**

Queries from decoder; keys/values from encoder.

$$
Q_c = S' W_Q^{(\text{cross})},\quad
K_c = H_{\text{enc}} W_K^{(\text{cross})},\quad
V_c = H_{\text{enc}} W_V^{(\text{cross})}.
$$

Attention:

$$
A_c = \text{softmax}\left(\frac{Q_c K_c^\top}{\sqrt{d_k}}\right)V_c.
$$

Residual + LN:

$$
S'' = \text{LayerNorm}(S' + A_c W_O^{(\text{cross})}).
$$

---

### **2.3 Feed-Forward Network**

Same FFN structure:

$$
F = \phi(S''W_1 + b_1)W_2 + b_2.
$$

Residual + LN:

$$
H_{\text{dec}} = \text{LayerNorm}(S'' + F)
\quad \in \mathbb{R}^{M \times d}.
$$

This is final decoder output.

---

## **3. Output head**

$$
\text{logits} = H_{\text{dec}} W_E^\top.
$$

Softmax gives $P(y_t \mid y_{<t}, x)$.

---

# **B. Parameter Counting**

Total params include:

---

## **1. Encoder (per layer)**

Same as encoder-only:

$$
4d d_k + 2d d_{\text{ff}} + 4d.
$$

---

## **2. Decoder (per layer)**

Has **three** attention modules:

1. Masked self-attn:
   $4d d_k$

2. Cross-attn (Q from decoder, K/V from encoder):
   also $4d d_k$

3. FFN:
   $2d d_{\text{ff}}$

4. 3 LayerNorms →
   $6d$

**Total per decoder layer:**

$$
8d d_k + 2d d_{\text{ff}} + 6d.
$$

---

## **3. Embeddings**

- Input token embeddings: $|V_{\text{in}}| \times d$

- Output token embeddings: $|V_{\text{out}}| \times d$
  (Often tied in text-to-text models.)

- Positional embeddings for input + output.

---

# **C. Training + Regularisation**

### **Training objective**

Teacher-forced seq2seq loss:

$$
\mathcal{L} = -\sum_t \log P(y_t \mid y_{<t}, x).
$$

### **Pretraining (T5)**

- **Span corruption** (mask random spans with sentinel tokens).
- Predict the missing text with decoder.

### Regularisation

- Dropout
- LayerNorm
- Weight decay

---

# **D. Typical Tasks**

- Machine translation (canonical use).
- Abstractive summarisation.
- Paraphrasing.
- Question answering.
- Text-to-text unification (T5 slogan: **“Everything is text-to-text.”**)

---

# **E. What It Can / Can’t Capture**

### **Can**

- Rich input understanding (encoder).
- Strong generation conditioned on input (decoder).
- Flexible seq2seq modelling structure.

### **Can’t**

- Avoid quadratic attention cost.
- Efficiently handle very long inputs and outputs.

---

# **F. Pros / Cons**

### **Pros**

- Best architecture for **supervised seq2seq**.
- Encoder specialises in reading; decoder specialises in writing.
- Excellent controllability.

### **Cons**

- Heaviest architecture (two stacks).
- More complex attention.
- Slower training/inference than pure encoder or pure decoder models.

## 1️⃣ BAYES’ RULE

### **Formula**

$$P(A \mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

### **Meaning**

Infer the probability of a **hidden cause** $A$ (class, tag, topic) from an **observed signal** $B$ (words, features).

---

### **Strengths**

- **Uses priors** $P(A)$ → robust when data is sparse.
- Natural fit for **generative classifiers** (Naive Bayes).
- Gives **posterior** $P(A\mid B)$, which is what we actually want for classification.

### **Weaknesses**

- Needs $P(B)$, often computed via **law of total probability** (can be intractable with many classes).
- Sensitive to **incorrect priors** and assumptions (e.g. Naive Bayes independence).

---

### **Uses in NLP**

- **Naive Bayes** (sentiment, topic classification).
- **WSD**: sense as hidden variable.
- **HMMs**: interpreting hidden states given observations.

---

### **Worked Example (Sentiment classification)**

Given:

- $P(\text{positive}) = 0.4$
- $P(\text{negative}) = 0.6$
- $P(\text{“excellent”} \mid \text{positive}) = 0.1$
- $P(\text{“excellent”} \mid \text{negative}) = 0.01$

Compute $P(\text{positive} \mid \text{“excellent”})$.

1. **Denominator** via law of total probability:

$$
P(\text{“excellent”}) =
0.1 \cdot 0.4 + 0.01 \cdot 0.6
= 0.04 + 0.006
= 0.046
$$

2. **Posterior**:

$$
P(\text{positive} \mid \text{“excellent”})
= \frac{0.1 \cdot 0.4}{0.046}
= \frac{0.04}{0.046}
\approx 0.87
$$

**Interpretation:** Seeing “excellent” makes the text **very likely positive**, even though positive is not the majority class.

---

## 2️⃣ CONDITIONAL PROBABILITY

### **Formula**

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

### **Meaning**

Probability of **A** happening given that **B** has happened — the basic building block of **sequence models**.

---

### **Strengths**

- Lets us **factor** complex joint distributions into manageable pieces:
  $P(w_1,\dots,w_T)=\prod_t P(w_t\mid w_{<t})$.
- Underlies **N-grams**, **HMMs**, and **autoregressive LMs**.

### **Weaknesses**

- Accurate estimation needs large data; suffers badly from **sparse counts**.
- Motivates **smoothing** and more powerful models (RNNs, Transformers).

---

### **Uses in NLP**

- **N-gram LMs**: $P(w_i\mid w_{i-n+1}^{i-1})$.
- **HMMs**: $P(\text{tag}_t \mid \text{tag}_{t-1})$, $P(\text{word}\_t \mid \text{tag}\_t)$.
- Conceptually: **attention** defines a conditional distribution over positions.

---

### **Worked Example (Corpus counts)**

A corpus has 200 sentences:

- 40 sentences contain _cat_
- 10 sentences contain both _cat_ and _chases_

Approximate sentence-level probabilities:

- $P(\text{cat}) = 40 / 200 = 0.2$
- $P(\text{chases}, \text{cat}) = 10 / 200 = 0.05$

Then:

$$
P(\text{chases} \mid \text{cat})
= \frac{0.05}{0.2}
= 0.25
$$

**Interpretation:** Among sentences with _cat_, **25%** also contain _chases_.

---

## 2️⃣.2️⃣ JOINT PROBABILITY

### **Formula**

$P(A,B) = P(A \cap B)$

and its key factorisations:

$P(A,B) = P(A\mid B)P(B) = P(B\mid A)P(A)$

### **Meaning**

Probability that **A and B happen together**.  
It is the core building block that **Bayes’ rule** and **conditional probability** are derived from.

---

### **Strengths**

- Fundamental to **generative models**: entire sequence probabilities are joint probabilities.
- Allows factorisation via the **chain rule**:
  $P(w_1,\dots,w_T)=\prod_t P(w_t \mid w_{<t})$

### **Weaknesses**

- Direct estimation of high-dimensional joints is impossible with sparse data → must factorise into conditionals and use **smoothing** or **neural models**.
- Co-occurrence alone doesn’t tell you direction of dependence.

---

### **Uses in NLP**

- **N-gram LMs**: joint over sequences via product of conditionals.
- **HMMs**: joint over tags + words
  $P(z_{1:T}, x_{1:T}) = P(z_1)\prod_t P(z_t\mid z_{t-1})P(x_t\mid z_t)$
- **Co-occurrence matrices** used by SGNS / distributional semantics.

---

## 3️⃣ LAW OF TOTAL PROBABILITY

### **Formula**

$$P(B)=\sum_i P(B\mid A_i)P(A_i)$$

### **Meaning**

Total probability of **B** is the sum over contributions from all **latent causes** $A_i$.

---

### **Strengths**

- Connects **latent-variable models** (like HMMs) to observed probabilities.
- Provides denominator in **Bayes’ Rule**.

### **Weaknesses**

- Requires a **complete, disjoint** set of $A_i$ (often unrealistic).
- Can be intractable if there are many possible hidden states.

---

### **Uses in NLP**

- Computing $P(\text{word})$ from POS-tag-conditioned distributions.
- Marginalising hidden **HMM states**.
- Normalisation sums in generative models.

---

### **Worked Example (Noun vs verb)**

Let $C \in {\text{noun}, \text{verb}}$. Suppose:

- $P(\text{noun}) = 0.6$, $P(\text{verb}) = 0.4$
- $P(w \mid \text{noun}) = 0.1$
- $P(w \mid \text{verb}) = 0.02$

Then:

$$
P(w)
= P(w\mid \text{noun})P(\text{noun}) + P(w\mid \text{verb})P(\text{verb})
= 0.1\cdot 0.6 + 0.02\cdot 0.4
= 0.06 + 0.008
= 0.068
$$

**Interpretation:** Overall, **6.8%** of tokens are this word, aggregating across noun/verb uses.

---

## 4️⃣ ADD-ONE / ADD-ALPHA SMOOTHING

### **Formula (unigram)**

$$P(w)=\frac{C(w)+\alpha}{N+\alpha |V|}$$

For conditional (e.g. N-gram):

$$P(w_i\mid h)=\frac{C(h,w_i)+\alpha}{C(h)+\alpha |V|}$$

---

### **Meaning**

Adds a small **pseudo-count** $\alpha$ to every event so unseen events have **non-zero** probability.

---

### **Strengths**

- Simple, closed-form, easy to compute in exam.
- Prevents zero probabilities → critical for **N-gram products**.

### **Weaknesses**

- **Over-smooths**, especially with large vocabularies.
- Unrealistic distributions → replaced in practice by **Kneser–Ney**.

---

### **Uses in NLP**

- Textbook **N-gram LMs**.
- Naive Bayes when many features are unseen in a class.

---

### **Worked Example (Unigram)**

Vocabulary size: $|V| = 5$
Total tokens: $N = 100$
Word $w$ appears $C(w)=3$ times.
Let $\alpha = 1$ (add-one).

$$P(w)=\frac{3+1}{100+1\cdot 5}=\frac{4}{105}\approx 0.0381$$

**Interpretation:** The smoothed probability is slightly **higher** than raw MLE (3/100=0.03) because we expanded the denominator and added pseudo-counts.

---

## 5️⃣ DOT PRODUCT

### **Formula**

$$u \cdot v = \sum_i u_i v_i$$

### **Meaning**

Measures alignment scaled by magnitude.

- Large if vectors point in the same direction and are long.
- Small if vectors are short or orthogonal.

---

### **Strengths**

- Very fast to compute.
- Used directly in attention mechanisms (dot-product attention).
- Captures both magnitude and orientation → useful when vector length carries semantic information (e.g., SGNS frequency effects).

### **Weaknesses**

- Magnitude-sensitive: longer vectors yield larger dot products even if direction is not highly aligned.
- Cannot be used to compare vectors when norms differ widely (common in word embeddings).

---

### **Uses in NLP**

- **Self-attention / cross-attention**: $\text{score}(q,k) = q^\top k$
- **SGNS objective**: $u_{\text{pos}}^\top v_t$
- Encoding co-occurrence or similarity in simpler models.

---

### **Worked Example**

Two embeddings:

- $u = (1, 2)$
- $v = (3, 4)$

Compute:

$$u \cdot v = 1 \cdot 3 + 2 \cdot 4 = 3 + 8 = 11$$

**Interpretation:** A large positive dot product → vectors point in a broadly similar direction and have sizable magnitudes.

## 5️⃣.2️⃣ COSINE SIMILARITY

### **Formula**

$$\cos(u,v) = \frac{u \cdot v}{\|u\| \|v\|}$$

### **Meaning**

Measures pure directional alignment, ignoring magnitude.

- $= 1$: same direction
- $= 0$: orthogonal
- $= -1$: opposite direction

---

### **Strengths**

- Norm-invariant → frequency effects do not distort similarity.
- Performs well in high-dimensional embedding spaces.
- Better for semantic similarity where direction matters more than length.

### **Weaknesses**

- Affected by global geometry issues such as embedding anisotropy.
- Cannot distinguish antonyms (e.g., hot and cold may have similar directions).
- Does not incorporate magnitude information when that might be meaningful.

---

### **Uses in NLP**

- Similar word retrieval (synonyms, paraphrases).
- Document/sentence similarity.
- Clustering embeddings.
- Intrinsic evaluation of word vectors (e.g. SimLex, WordSim-353).

---

### **Worked Example**

Vectors:

- $u = (1, 2)$
- $v = (3, 4)$

Dot product (from earlier):

$$u \cdot v = 11$$

Norms:

$$\|u\| = \sqrt{1^2 + 2^2} = \sqrt{5}, \quad \|v\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

Cosine similarity:

$$\cos(u,v) = \frac{11}{\sqrt{5} \cdot 5} = \frac{11}{5\sqrt{5}} \approx 0.984$$

**Interpretation:** Near 1 → vectors point in almost the same direction, indicating strong semantic similarity.

---

## 6️⃣ EUCLIDEAN DISTANCE

### **Formula (2D)**

$$d(x,y)=\sqrt{(x_1-y_1)^2 + (x_2-y_2)^2}$$

### **Meaning**

Geometric distance between two embeddings in space.

---

### **Strengths**

- Intuitive interpretation as “how far apart”.

### **Weaknesses**

- In high dimensions, distances tend to **concentrate** (curse of dimensionality).
- Less useful than cosine for semantic similarity.

---

### **Uses**

- KNN classification.
- Clustering.
- Detecting embedding outliers.

---

### **Worked Example**

Word embeddings:

- $x = (1,2)$ (cat)
- $y = (3,4)$ (dog)

$$
d(x,y)=\sqrt{(1-3)^2+(2-4)^2}
= \sqrt{(-2)^2 + (-2)^2}
= \sqrt{4+4}
= \sqrt{8} \approx 2.828
$$

**Interpretation:** Larger distance → less similar; here they’re moderately far apart.

---

## 7️⃣ L2 REGULARISATION

### **Formula**

$$L' = L + \lambda |w|^2$$

where $|w|^2 = \sum_j w_j^2$.

---

### **Meaning**

Adds penalty for **large weights**, encouraging smaller, smoother models.

---

### **Strengths**

- Reduces **overfitting**.
- Makes optimisation more stable; discourages extreme weights.

### **Weaknesses**

- Does **not** create sparsity (unlike L1).
- Too large $\lambda$ → **underfitting** (weights shrunk too much).

---

### **Uses**

- Logistic / softmax regression for text.
- Neural networks, weight decay.

---

### **Worked Example**

Suppose:

- Original loss $L = 0.40$
- Weight vector $w = [2,1]$
- $\lambda = 0.1$

1. Compute squared norm:

$$|w|^2 = 2^2 + 1^2 = 4 + 1 = 5$$

2. New loss:

$$L' = 0.40 + 0.1 \cdot 5 = 0.40 + 0.5 = 0.90$$

**Interpretation:** The model is penalised for large weights; training will prefer smaller weights if they don’t hurt performance too much.

---

## 8️⃣ PRECISION, RECALL, F1

### **Formulas**

$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = \frac{2PR}{P+R}$$

---

### **Meaning**

- **Precision**: among predicted positives, how many are correct?
- **Recall**: among true positives, how many did we find?
- **F1**: harmonic mean, balances both.

---

### **Strengths**

- Handle **class imbalance** better than raw accuracy.
- F1 good when both FP and FN matter.

### **Weaknesses**

- F1 ignores **true negatives**.
- Precision alone ignores missed positives; recall alone ignores false positives.

---

### **Uses**

- NER, hate-speech detection, toxicity detection, IE.
- Any **classification / sequence labelling** tasks.

---

### **Worked Example (Spam)**

Given:

- TP = 20
- FP = 5
- FN = 10

1. Precision:

$$P = \frac{20}{20 + 5} = \frac{20}{25} = 0.8$$

2. Recall:

$$R = \frac{20}{20 + 10} = \frac{20}{30} \approx 0.667$$

3. F1:

$$
F_1 = \frac{2 \cdot 0.8 \cdot 0.667}{0.8 + 0.667}
\approx \frac{1.067}{1.467} \approx 0.727
$$

**Interpretation:** Model is strong but misses some spam (recall < precision).

**Metric that penalises false negatives most directly:** **Recall**.

---

## 9️⃣ CROSS-ENTROPY

### 1️⃣ Entropy

#### What Entropy Measures

Entropy $H(X)$ quantifies unpredictability of a random variable.

- If outcomes are evenly distributed → high entropy (high uncertainty).
- If one outcome is dominant → low entropy (predictable).

#### Intuition

Measured in bits. “How many yes/no questions do I need, on average, to identify the outcome?”

#### Formula

$$H(X) = -\sum_x P(x) \log_2 P(x)$$

#### Interpretation

- Uniform distribution over $N$ outcomes:
  $$H = \log_2 N$$
- Skewed distribution: Entropy drops because uncertainty drops.
- English entropy: ~1.3 bits/character (strong frequency biases → less uncertainty).

### 2️⃣ Cross-Entropy

#### General Formula

For gold distribution $p(y)$ and model distribution $q(y)$:

$$H(p,q) = -\sum_y p(y) \log q(y)$$

#### One-Hot Case

If the true class is $c$ - $p(c) = 1$ for correct $c$ and $p(y) = 0$ for all others:

$$H(p,q) = -\log q(c)$$

Given $q(c) = P(y_i = c | x_i; \theta)$

for each training example:

$$\text{Cross-Entropy} = -logP(y_i | x_i; \theta)$$

Which is equivalent to **Negative log-likelihood**

#### Meaning

Expected **surprise** of the **true labels** under the model’s predicted distribution.

Equivalent to the average negative log-probability assigned to the correct outputs.

- **High Cross-Entropy** (High Surprise) = predictions are far from their labels
- **Low Cross-Entropy** (Low Surprise) = Prediction close to labels

#### Strengths

- Directly corresponds to **maximum likelihood training**.
- Smooth gradient → works well with backprop.
- Standard for classification, LM token prediction, SFT.

#### Weaknesses

- Punishes confidently wrong predictions very strongly.
- Not always aligned with human quality metrics in generation tasks.

#### Uses in NLP

- Logistic regression / softmax regression.
- RNN / Transformer LMs (token-level training loss).
- SFT for LLMs and general classification tasks.

#### Worked Example (Word Prediction)

Given:

- Model predicts:

  | word | prob |
  | ---- | ---- |
  | the  | 0.5  |
  | a    | 0.3  |
  | cat  | 0.2  |

- True word = "cat" → one-hot: $p(\text{cat}) = 1$.

#### Cross-entropy:

$$H(p,q) = -\log q(\text{cat}) = -\log(0.2) \approx 1.609$$

Interpretation:

- Somewhat surprised.
- Higher probability on the true class → lower cross-entropy.

### 3️⃣ Perplexity

#### Definition

$$PP = 2^{H_M}$$

where $H_M$ is the model's cross-entropy on the data.

#### Intuition

- Perplexity ≈ model’s **effective number of choices** at each prediction step.
- Lower = better.

#### Interpretation Scale

- $PP \approx 2$: model is very confident (≈ 2 plausible next tokens).
- $PP \approx 100$: highly uncertain (many plausible next tokens).

#### Why Logs Appear

- Avoid numerical underflow when multiplying many small probabilities.
- Cross-entropy uses logs; perplexity simply exponentiates.

#### Example

If cross-entropy = 3 bits/word:

$$PP = 2^3 = 8$$

→ model behaves as if it has 8 effective choices per word.

#### Model Comparison

- Unigram: high perplexity (no context)
- Bigram: lower.
- Trigram: lower still.

More context → more certainty → lower perplexity.

### 4️⃣ How These Quantities Fit Together

How These Quantities Fit Together

- Entropy: Intrinsic unpredictability of the data distribution.
- Cross-Entropy: Model’s estimate of that unpredictability.
- Equals entropy only if model = true distribution.
- Perplexity: A readability-friendly transformation of cross-entropy. “How confused is the model?”

# **1. Maximum Likelihood Estimation (MLE)**

**Definition**

- Choose parameters $\theta$ that **maximise the likelihood** of observed corpus data.
- For an N-gram:
  $$P(w_i | h) = \frac{\text{count}(h, w_i)}{\text{count}(h)}$$

**Pros**

- **Unbiased** estimator (given enough data).
- **Simple**, closed-form for many models.
- **Interpretable**.

**Cons / Characteristic Errors**

- **Zero-probabilities** for unseen events ⇒ catastrophic failure in generative models.
- **Overfitting** on small corpora.
- Poor generalisation beyond observed contexts.

**When acceptable?**

- Very **large corpora**.
- Tasks where **exact probabilities matter less** (e.g., ranking with smoothing fallback).

**When unacceptable?**

- Small corpora, rare events, OOV-heavy domains.

---

# **2. Add-One / Add-Alpha (Laplace) Smoothing**

**Definition**

- Add a small constant ($\alpha$) to counts to avoid zero probabilities.
- Formula:
  $$P(w | h) = \frac{\text{count}(h,w) + \alpha}{\text{count}(h) + \alpha \cdot |V|}$$

**Pros**

- Eliminates **zero-probabilities**.
- Very **simple**.

**Cons / Errors**

- **Over-smooths**: probability mass redistributed too uniformly.
- Degrades performance on high-frequency items.

**Acceptable?**

- Toy examples, demonstrations, extremely low-resource settings.

**Unacceptable?**

- Real NLP tasks where accuracy matters (prefer Kneser–Ney, Good–Turing).

---

# **3. Cross-Entropy Loss**

**Definition**

- Measures how well predicted distribution $q$ approximates true distribution $p$.
- $$H(p,q) = -\sum p(w) \log q(w)$$
- For supervised training: use target labels as $p$ (one-hot).

$p(w) = \begin{cases} 1 & \text{if } w = y \\ 0 & \text{otherwise}. \end{cases}$

Since $p(w) = 0$ for all classes except $w=y$

$H(p,q) = -\Big( p(y)\log q(y) + \sum_{w\neq y} p(w)\log q(w) \Big)$

But since $p(y)=1$ and $p(w\neq y) = 0$:

$H(p,q) = - \log q(y)$

**Used for:**

- Training classifiers, language models, seq2seq models.

**Strengths**

- Directly optimises model likelihood.
- Differentiable, compatible with SGD.

**Weaknesses**

- Sensitive to **miscalibrated probabilities**.
- Encourages overconfidence.

**Alternatives**

- KL divergence (equivalent up to constant).
- Margin-based losses.

---

# **4. Teacher Forcing**

**Definition**

- During training, RNN/Transformer decoder receives **gold previous token** instead of its own prediction.

**Pros**

- Faster, more stable training.
- Helps model learn correct conditional distributions.

**Cons / Errors**

- **Exposure bias**: model never sees own prediction errors during training.
- At inference, small mistakes can **cascade**.

**Mitigations**

- Scheduled sampling.
- Sequence-level training (e.g., RL, minimum-risk training).

---

# **5. Stochastic Gradient Descent (SGD)**

**Definition**

- Update parameters using **gradient estimates** from minibatches.

**Pros**

- Scales to huge datasets.
- Good exploration of parameter space due to noise.
- Fast convergence with variants (Adam, RMSProp).

**Cons / Errors**

- Highly sensitive to **learning rate**.
- Can get stuck in bad minima/saddle points.
- Noisy gradients ⇒ unstable without tuning.

**Acceptable?**

- Always the default for neural models.

**Unacceptable?**

- When data is tiny (closed-form solutions preferable).

---

# **6. Backpropagation**

**Definition**

- Compute gradients via the **chain rule** through all layers.

**Core idea**

- Local gradient × upstream gradient.
- Enables training deep networks.

**Failure modes**

- Vanishing/exploding gradients (mitigated by LSTMs, residual connections).
- Requires differentiable components.

---

# **7. Negative Sampling**

**Definition**

- Approximate softmax by contrasting **true pairs** with a small set of **noise samples**.
- Common in **word2vec skip-gram**.

**Pros**

- Huge speed improvements.
- Good embeddings with limited computation.

**Cons / Errors**

- Choice of **negative distribution** strongly affects performance.
- Estimates only **relative** probabilities (not full normalised softmax).

**Typical errors**

- Rare words poorly learned when insufficient negatives.

---

# **8. Contrastive Learning**

**Definition**

- Learn representations by **bringing positives closer** and **pushing negatives apart**.
- Often uses **InfoNCE** loss.

**Examples**

- Sentence embedding models.
- Image-text alignment (CLIP).

**Pros**

- Strong generalisation.
- No need for labelled data.

**Cons**

- Requires large batch sizes / memory.
- Negative selection crucial.

**Sources of error**

- False negatives (different sentences with same meaning).

---

# **9. Transfer Learning**

**Definition**

- Use parameters trained on one task/domain for a different task.

**Forms**

- **Feature-based** (use frozen embeddings).
- **Fine-tuning** (update all weights).
- **LoRA/adapter layers**.

**Pros**

- Massive performance boost on small datasets.
- Leverages world knowledge.

**Cons**

- **Catastrophic forgetting**.
- Domain mismatch and biases propagate.

---

# **10. In-Context Learning (ICL)**

**Definition**

- Model learns behaviour from **examples in the prompt**, not parameter updates.

**Pros**

- No training required.
- Flexibility across tasks.
- Data Efficient
- Immediate Deployment

**Cons / Errors**

- Very sensitive to **prompt order**, formatting, bias.
- Task performance unstable for small LMs.
- No guarantee of generalisation.

---

# **11. Zero-Shot and Few-Shot Learning**

**Zero-shot**

- Provide only natural-language task description.

**Few-shot**

- Provide small set of labelled examples in context.

**Pros**

- Extremely efficient.
- Works best with LLMs trained on broad mixtures.

**Cons**

- Erratic for structured tasks.
- Relies heavily on pretraining prior.

**When unacceptable?**

- Safety-critical or high-precision tasks.

---

# **12. Cross-Lingual Knowledge Transfer**

### **Definition**

- Using a model trained on **source languages** to perform tasks in a **different target language**, without requiring large labelled datasets in the target language.

### **Mechanisms**

- **Multilingual embeddings** → map words across languages into a shared semantic space.
- **Shared subword vocabularies (BPE/SentencePiece)** → overlapping morphology/scripts supports transfer.
- **Emergent alignment** → shared layers naturally align languages even _without_ parallel data.

### **Transfer Strategies**

- **Zero-Shot Transfer** → train on EN/DE, test directly on FR (no FR labels).
- **Translate-Test** → translate FR input → EN; use a monolingual EN model at inference.
- **Translate-Train** → translate EN training data → FR; fine-tune on synthetic FR data.

**Pros**

- **Helps low-resource languages**.
- Enables **zero-shot multilingual tasks**.
- **Single large model** vs thousands of monolingual
- Wider **global access to NLP** systems

**Cons / Errors**

- Transfer depends on **linguistic similarity**. (i.e shared morphology - latin, germanic)
- **Script Mismatch** (latin vs Arabic) -> weak transfer
- **Curse of Multilinguality** -> too many languages dilute parameter capacity per language
- **Tokenisation bias** ("fertility") -> Low resource languages get more subwords -> longer sequences -> higher compuite cost
- **Cultural shift issues:** models fail on concepts absent/different in high resoruce training data

---

# **13. Pretraining Objectives**

## **a) Causal Language Modelling (CLM)**

- Predict next token using only **left context**.
- Used in GPT-type models.
  **Strength:** excels at generation.
  **Weakness:** poorer at bidirectional understanding.

## **b) Masked Language Modelling (MLM)**

- Predict masked tokens using **both sides of context**.
- Used in BERT.
  **Strength:** strong encoding representations.
  **Weakness:** not generative.

## **c) Denoising Language Modelling**

- Corrupt input (shuffling, dropping, masking) and reconstruct.
- Used in T5, BART.
  **Strength:** robust representations.
  **Weakness:** expensive pretraining.

---

# **14. Post-Training Objectives (Expanded)**

Post-training modifies a **pre-trained next-token predictor** into a model that **follows instructions**, **aligns with human preferences**, and **avoids harmful behaviour**.  
Three dominant families of objectives are used: **SFT**, **RLHF**, and **DPO/RLVR-family objectives**.

---

# **a) Supervised Fine-Tuning (SFT)**

## **What It Is**

- Fine-tuning on curated **(instruction → response)** pairs.
- Examples come from **human demonstrations**, synthetic Teacher–Student pipelines, or multi-task instruction datasets (e.g., **Natural Instructions**, SNI).

## **How It Works**

- Input: formatted prompt, often with `<|user|>` and `<|assistant|>` delimiters.
- Model predicts output tokens.
- **Loss:** cross-entropy only on **assistant tokens**.

## **What It Teaches**

- **Task grounding:** recognising “what task is being asked”.
- **Behavioural priors:** politeness, format, safety scaffolding.
- **Generalisation:** across tasks via multitask training.

## **Pros**

- **Stable** (pure gradient descent).
- **Controllable behaviour** through curated demonstrations.
- No negative examples needed.

## **Cons**

- **Imitation only** → no notion of “better vs worse”.
- Strongly affected by **dataset quality, biases, demographic skew**.
- Can make models **overly verbose** or **pattern-copying**.

## **Typical Errors**

- **Shallow compliance:** follows templates without deep understanding.
- **Hallucination style-bias:** repeats demo patterns even when wrong.
- **Limited safety:** cannot resolve ambiguous / harmful user intents.

---

# **b) RLHF (Reinforcement Learning from Human Feedback)**

## **Why RLHF Exists**

- SFT cannot encode **preferences**, only behaviour.
- We need a mechanism to **compare outputs**:
  - “This answer is better than that one.”

## **Pipeline Components**

1. **Preference Data**

   - Human raters pick **preferred** output from pairs.

2. **Reward Model (RM)**

   - Trained to predict which output humans prefer.
   - Implements a **Bradley–Terry** comparison:
     - $\sigma(R(o^+) - R(o^-))$

3. **RL Optimisation (PPO)**

   - Policy (the LLM) is updated to **maximise RM score**  
      while staying close to the SFT model via **KL penalty**.

## **What It Teaches**

- **Helpful** (follows preferred behaviours).
- **Honest** (doesn’t fabricate confident lies when penalised).
- **Harmless** (avoids dangerous instructions).

## **Pros**

- Embeds **preference learning**, not mere imitation.
- Captures **nuanced human signals** (tone, respectfulness).
- Helps avoid pathological or unsafe behaviours.

## **Cons / Characteristic Errors**

- **Reward hacking:**
  - Model finds outputs that trick the RM but degrade truthfulness.
- **Over-optimisation / collapse:**
  - Becoming verbose, generic, or stylistically “safe”.
- **Drift:**
  - If KL penalty too small → catastrophic loss of base abilities.
- **Expensive:**
  - Must load **policy + reference + reward model** simultaneously.

## **Why PPO?**

- Stabilises updates by **clipping** large policy shifts.
- Needed because language-generation RL is notoriously unstable.

---

# **c) RLVR / DPO-family (Direct Preference Optimisation)**

These methods aim to **avoid full RL**, keeping the alignment signal but simplifying optimisation.

---

## **RLVR (RL with Verifiable Rewards)**

### **Core Idea**

- Replace Reward Model with a **programmatic verifier**:
  - Math tasks: exact numerical answer.
  - Code tasks: runs + passes tests.
  - Safety tasks: rule-based validators.

### **Pros**

- **Cheaper & simpler:**
  - Only Policy + Reference model required.
- **Stable:** No PPO instability.
- Works well for **objective-verifiable** tasks.

### **Cons**

- **Reasoning mismatch:**
  - Model may reach correct answer via flawed logic.
- **Reward gaming:**
  - Exploiting quirks in verifiers (e.g., formatting hacks).
- **Limited applicability:**
  - Not useful for open-ended dialogue, ethics, writing style.

---

# **TLDR SUMMARY**

- **MLE**: simple but brittle → fails on unseen events.
- **Smoothing (Add-α)**: prevents zeros, over-smooths.
- **Cross-entropy**: core loss for all LMs.
- **Teacher forcing**: stable training but exposure bias.
- **SGD/Backprop**: optimisation backbone; sensitive to tuning.
- **Negative sampling / contrastive**: efficient learning of embeddings.
- **Transfer / ICL / few-shot**: leverage pretrained knowledge.
- **Pretraining objectives**: CLM (generation), MLM (encoding), denoising (robust).
- **Post-training**: SFT, RLHF, RLVR align models to human preferences.

## **1. Core Idea**

- At inference, the model outputs a **probability distribution** over the vocabulary at each timestep.
- Decoding strategies determine **which token to choose** from this distribution.
- Trade-off: **determinism vs. diversity**, **efficiency vs. quality**, **search width vs. computational cost**.

---

# **2. Greedy Decoding**

### **What it is**

- Always pick the **argmax** token at each timestep.

### **How it works**

1. Model outputs probabilities: e.g. `{"cat": 0.4, "dog": 0.3, "sat": 0.2, ...}`
2. Choose the **highest-probability** token.
3. Feed it back into the model and repeat.

### **Strengths**

- **Fast**, **deterministic**, **low computation**.

### **Weaknesses**

- Gets stuck in **local optima**.
- Produces **generic / repetitive** text.
- Often misses globally better sequences.

### **Typical exam phrasing**

- “Given these output probabilities, apply _greedy_ decoding to produce the next token.”

---

# **3. Beam Search**

### **What it is**

- A **heuristic search** that keeps multiple candidate sequences (**beam width = k**) at each timestep.

### **How it works**

1. For each partial hypothesis, expand with **all possible next tokens**.
2. **Score** each sequence (usually sum of log-probs).
3. Keep the **top k** sequences.
4. Continue until EOS.

### **Strengths**

- Explores more of the search space than greedy.
- Produces **higher probability** sequences.
- Standard for machine translation / summarisation.

### **Weaknesses**

- **Computationally expensive** (≈ k× slower).
- Still approximate.
- Larger beam can produce **short, generic, or repetitive** outputs (length-bias issue).

### **Exam must-knows**

- You must be able to **do one step**:  
   Expand candidates → compute **sequence log-probabilities** → keep best **k**.

---

# **4. Sampling (Stochastic Decoding)**

### **What it is**

- Choose the next token by **sampling** from the full probability distribution.

### **How it works**

1. Model outputs probabilities.
2. Treat them as a **categorical distribution**.
3. Draw a random sample.

### **Strengths**

- Introduces **diversity** / variation.
- Avoids deterministic repetition.

### **Weaknesses**

- Can sample **very low-quality** or incoherent tokens from the long tail.
- High variance; outputs unpredictable.

---

# **5. Top-k Sampling**

### **What it is**

- Restrict sampling to the **k most probable tokens**, renormalise, then sample.

### **How it works**

1. Sort tokens by probability.
2. Keep top **k** (e.g. k=40).
3. Renormalise probabilities.
4. Sample from the reduced set.

### **Strengths**

- Removes low-probability noise.
- Keeps some diversity while maintaining stability.

### **Weaknesses**

- Fixed k does not adapt to distribution shape (e.g. whether tail is flat or steep).

### **Exam skill**

- Given probabilities and a **k**, you must show:
  - the truncated list
  - renormalised probabilities
  - sampled token.

---

# **6. Top-p Sampling (Nucleus Sampling)**

### **What it is**

- Select the **smallest** subset of tokens whose cumulative probability ≥ **p**, then sample.

### **How it works**

1. Sort tokens by probability.
2. Compute cumulative sum.
3. Take the **nucleus** where cumulative ≥ p (e.g. p=0.9).
4. Renormalise and sample.

### **Strengths**

- **Adaptive** to distribution shape.
- Avoids both:
  - long-tail noise
  - overly restrictive fixed-k behaviour.

### **Weaknesses**

- May still allow unpredictable outputs if distribution is flat.
- More difficult to compute manually (but still required in exams).

### **Exam requirement**

- Show **which tokens enter the nucleus**, renormalise, then sample.

---

# **7. Summary Table (Useful for Exam Comparisons)**

| Method          | Deterministic?    | Diversity   | Computation | Typical Failure                 |
| --------------- | ----------------- | ----------- | ----------- | ------------------------------- |
| **Greedy**      | Yes               | Low         | Very low    | Local optimum; repetition       |
| **Beam search** | Yes (for fixed k) | Low–Med     | High        | Length bias; generic outputs    |
| **Sampling**    | No                | High        | Low         | Chaotic, incoherent outputs     |
| **Top-k**       | No                | Medium      | Low         | k not adaptive                  |
| **Top-p**       | No                | Medium–High | Low         | Sensitive to distribution shape |

---

# **TLDR**

- **Greedy:** take argmax. Fast but gets stuck.
- **Beam search:** keep top-k sequences. Higher quality but expensive.
- **Sampling:** draw from full distribution. Diverse but noisy.
- **Top-k:** sample from k best tokens. Controls noise.
- **Top-p:** sample from smallest cumulative-p set. Adaptive + diverse.

# **1. Byte-Pair Encoding (BPE)**

## **What BPE is used for**

- A **subword tokenisation algorithm**.
- Solves the **unknown word** problem by breaking rare words into frequent subword units.
- Reduces vocabulary size while keeping expressive power.
- Used in GPT, RoBERTa, many modern LLMs.

---

## **How the BPE algorithm works (steps)**

### **Training phase**

1. Start with training text split into **characters** (plus a word boundary symbol, often `</w>`).
2. Count **all symbol pairs** (adjacent characters/subwords).
3. Find the **most frequent pair**.
4. **Merge** that pair into a new symbol.
5. Replace all occurrences in the corpus.
6. Repeat **N merges** (hyperparameter).

### **Inference (encoding)**

- Apply stored merge rules **in order**, merging greedily until no further merges apply.

---

## **Hand-simulation example (exam essential)**

### **Corpus**

`low lowish`

### **Initial segmentation**

`l o w </w> l o w i s h </w>`

### **Step 1: Count pairs**

- `l o`: 2
- `o w`: 2
- `w </w>`: 1
- `w i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent pair: **`l o`** or **`o w`** (tie; either is acceptable in exam).  
Assume we merge **`l o → lo`**.

### **After merge**

`lo w </w> lo w i s h </w>`

### **Step 2: Recount pairs**

- `lo w`: 2
- `w </w>`: 1
- `w i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent: **`lo w` → low**.

### **After merge**

`low </w> low i s h </w>`

### **Step 3: Recount**

- `low </w>`: 1
- `low i`: 1
- `i s`: 1
- `s h`: 1
- `h </w>`: 1

Most frequent: **`i s` → is**.

### **After merge**

`low </w> low is h </w>`

### **Step 4: Recount**

- `is h`: 1
- `h </w>`: 1

Most frequent: **`is h` → ish**.

### **Final segmentation of words**

- **low**
- **low + ish**

### **Key exam point**

- You must show **pair counts**, **merge selection**, and **updated corpus** at each step.
- **Initial vocab = all characters + end-of-word.**
- **Each merge adds one new symbol.**
- **Stop when |V| reaches target.**

---

# **2. Backpropagation**

## **What backpropagation is used for**

- Computes **gradients** of the loss w.r.t. all model parameters in a neural network.
- Enables **Stochastic Gradient Descent (SGD)** or Adam to update weights.
- Essential for training any neural model: MLPs, CNNs, RNNs, Transformers.

---

## **Core steps of backpropagation**

### **1. Forward pass**

- Compute outputs layer by layer.
- Compute loss $L$ (e.g. cross-entropy).

### **2. Backward pass**

For each layer from top to bottom:

#### **a. Compute local derivatives**

- For linear layer:  
   $z = Wx + b$,  
   $\frac{\partial z}{\partial W} = x^\top$,  
   $\frac{\partial z}{\partial x} = W^\top$.
- For activation $a = f(z)$:  
   Multiply by **f'(z)**.

#### **b. Apply chain rule**

- **upstream gradient x local derivative = downstream gradient**
- Gradient to propagate further.

#### **c. Accumulate gradients w.r.t. parameters**

- Store $\partial L / \partial W$ and $\partial L / \partial b$.

### **3. Update step**

For each parameter $\theta$:
$$\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$$
where $\eta$ is the learning rate.

---

## **Minimal worked example (exam-ready)**

For a 1-layer network:

### **Forward**

- $z = Wx + b$
- $y = \text{softmax}(z)$
- Loss: $L = -\log y_{target}$

### **Backprop**

1. $\frac{\partial L}{\partial z} = y - t$ (where $t$ is one-hot target)
2. $\frac{\partial L}{\partial W} = (y - t)\,x^\top$
3. $\frac{\partial L}{\partial b} = y - t$
4. $\frac{\partial L}{\partial x} = W^\top (y - t)$

Because (chain rule)

$\frac{\partial L}{\partial z} = y - t$

$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}} = \delta_i \cdot x_j$

$\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial b_i} = \delta_i$

$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j} = (W^\top \delta)_j$

#### **Example**

Input: $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

Weights:

$W = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}$

Bias:

$b = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$

True label:

$\text{t or y} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

Forward pass:

$z = Wx = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
$y=softmax(z)=[0.730.27]y = \text{softmax}(z) = \begin{bmatrix} 0.73 \\ 0.27 \end{bmatrix}$
$L = -\log 0.73$

---

### ⭐ 1. **Compute $\frac{\partial L}{\partial z}$​** (error signal)

Softmax + cross-entropy has the known closed-form:

$\frac{\partial L}{\partial z_i} = y_i - t_i$

#### **Plug in values**

$\frac{\partial L}{\partial z} = \begin{bmatrix} 0.73 - 1 \\ 0.27 - 0 \end{bmatrix} = \begin{bmatrix} -0.27 \\ 0.27 \end{bmatrix}$

We call this:

$\delta = \frac{\partial L}{\partial z}$

---

### ⭐ 2. **Compute $\frac{\partial L}{\partial W}$​** (using chain rule)

Each weight affects the loss only through **z**, so:

$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial W_{ij}}$

We already know:

- $\frac{\partial L}{\partial z_i} = \delta_i$
- $z_i = \sum_j W_{ij} x_j \Rightarrow \frac{\partial z_i}{\partial W_{ij}} = x_j$

So:

$\frac{\partial L}{\partial W_{ij}} = \delta_i x_j$

#### **Now compute each number manually**

$\partial L/\partial W = \begin{bmatrix} 0.73 \\ 0.27 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

$\frac{\partial L}{\partial W} = \begin{bmatrix} -0.27 & -0.54 \\ 0.27 & 0.54 \end{bmatrix}$

---

### ⭐ 3. **Compute $\frac{\partial L}{\partial b}$​**

Bias affects z additively:

$z_i = W_i x + b_i \Rightarrow \frac{\partial z_i}{\partial b_i} = 1$

So:

$\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i} \cdot 1 = \delta_i$

### With numbers:

$\frac{\partial L}{\partial b} = \begin{bmatrix} -0.27 \\ 0.27 \end{bmatrix}$

---

#### ⭐ 4. **Compute $\frac{\partial L}{\partial x}$​**

Again use chain rule:

$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j}$

But:

$z_i = W_{i1} x_1 + W_{i2} x_2 \Rightarrow \frac{\partial z_i}{\partial x_j} = W_{ij}$

So in vector form:

$\frac{\partial L}{\partial x} = W^\top \delta$

#### **Compute numerically**

Transpose:

$W^\top = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}$

Now multiply:

$\frac{\partial L}{\partial x} = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix} \begin{bmatrix}-0.27 \\ 0.27\end{bmatrix}$

$\frac{\partial L}{\partial x} = \begin{bmatrix} 0.27 \\ -0.27 \end{bmatrix}$

Exam versions often give you a **tiny input vector**, **tiny weight matrix**, and ask for one forward + backward step.

---

**Some key notes:**

- _“Why does BPTT cause memory issues?”_
  - BPTT is for RNNs: storing every intermediate hidden state and activation for every timesept grows linearly with sequence length.
- Backprop for Transformers
  - For sequence N, memory for attention is $O(N^2L)$
  - QKV, softmax weights also scales with Nxd
- _“How can you avoid exploding gradients?”_
  **Solutions**: gradient clipping, truncated BPTT, gating units (LSTM/GRU).
- _“What happens to gradients through repeated multiplication?”_
  - **Vanishing**: repeated multiplication by values <1 shrinks gradients.
  - **Exploding**: repeated multiplication by values >1 grows gradients.

# **Summary**

## **BPE**

- Used for **subword tokenisation**.
- Steps: **count pairs → merge most frequent → repeat**.
- Must be able to **simulate manually**.

## **Backpropagation**

- Used for **gradient computation** in neural networks.
- Steps: **forward pass → compute loss → backward pass using chain rule → update**.

---

# **TLDR**

- **BPE:** iterative pair-merging subword algorithm; know how to run it by hand.
- **Backprop:** chain rule applied layer-by-layer to compute gradients; central to training neural nets.

---

# **ADDITIONAL MATHEMATICAL & COMPUTATIONAL CONCEPTS — REVISION NOTES**

---

## **1. Zipf’s Law & Sparse Data**

### **Zipf’s Law**

- **Definition:** In natural language, the **frequency** of a word is **inversely proportional to its rank**:
  $$f(r) \propto \frac{1}{r}$$
- **Implication:**
  - A **few words** occur extremely often (e.g., _the, of, and_).
  - **Most words** occur **very rarely** → **long tail** distribution.
- **Why it matters:**
  - Models trained with MLE will **underfit** or produce **0-probability** for rare words.
  - Forces need for **smoothing**, **subword models** (BPE), and **large corpora**.

### **Sparse Data**

- **Meaning:** Many valid word combinations **never appear in training**, even in huge corpora.
- **Consequences for NLP tasks:**
  - **Language modelling:** N-grams fail for unseen sequences → smoothing needed.
  - **Parsing:** Rare constructions weaken model probability estimates.
  - **Classification:** Rare features produce unstable estimates.

---

## **2. Training / Development / Test Sets**

### **Purpose**

- **Training set:** Fit model parameters.
- **Development (validation) set:**
  - Tune hyperparameters (learning rate, smoothing constant, model depth).
  - Early stopping.
- **Test set:**
  - Final **unbiased estimate** of generalisation.
  - Must **never** influence model decisions.

### **Why they matter**

- **Avoid overfitting**
  - Distinct splits ensure improvements are not just memorisation.
- **Hyperparameter control**
  - Dev set provides feedback without contaminating test performance.
- **Fair model comparison**
  - Competitors must evaluate on the **same untouched test distribution**.
- **Detect domain shift**
  - If dev ≠ test domain, tuning may generalise poorly — common in NLP (e.g., Wikipedia → Twitter).

### **Task examples**

- **Language modelling:** Dev set used to monitor **perplexity**.
- **Classification:** Dev set guides regularisation strength.
- **LLM finetuning:** Dev set prevents model collapse during SFT.

- **Using test set for hyperparameter tuning**  
   → **data leakage**, inflated scores, invalid evaluation.
- **Randomly mixing genres across splits** (e.g., training on Wikipedia, testing on Twitter)  
   → leads to unexpected ↑ perplexity / ↓ accuracy.
- **Too small dev set**  
   → unstable hyperparameter decisions.
- **Training and dev from different distributions**  
   → hyperparameters tuned for the wrong domain.
- **Overfitting visible on dev before test**  
   → solved via early stopping.

---

## **3. LLM Development Phases**

### **Pre-training**

- **Objective:** Learn broad linguistic and world knowledge via
  - **Causal LM (next-token prediction)**
  - **Masked LM (MLM)**
  - **Denoising**
- **Data:** Massive, diverse, often web-scale.

### **Post-training**

1. **SFT (Supervised Fine-Tuning):**

   - Learn to follow instructions; dataset contains **input → desired output**.

2. **RLHF (Reinforcement Learning from Human Feedback):**

   - Model generates responses → human/ranker provides preference → **reward model** → policy optimisation (PPO-like).

3. **RLVR (Reinforcement Learning from Verifiable Rewards):**

   - Similar to RLHF but uses **automatically verifiable signals** (e.g., maths correctness, constraint satisfaction).

### **Fine-tuning (task-specific)**

- Smaller, focused datasets.
- Examples: sentiment classifier, legal summarisation model, domain-specific chatbot.

---

## **4. LLM Inference Phases**

### **Initial KV Cache Creation (“prefill”)**

- Input prompt is passed through the model **once** to initialise **Key/Value activations** for attention.
- Speeds up generation: future tokens only attend to cached vectors, not recompute entire sequence.

### **Auto-regressive Decoding**

- At each step, the model:
  1. Uses KV cache
  2. Computes next-token logits
  3. Applies **decoding strategy** (greedy, sampling, top-k, top-p).
- Continues until EOS token or length limit.

---

# **5. Regular Expressions (Regex)**

### **Definition**

A compact notation for specifying **patterns over strings**.

### **Examples**

- `\d{3}-\d{2}-\d{4}` → US-style number pattern.
- `^[A-Z][a-z]+$` → Capitalised word.

### **Relevant NLP tasks**

- **Tokenisation**
- **Preprocessing / cleaning**
- **Pattern extraction** (dates, emails)
- **Rule-based NER**

### **Why important**

Still widely used despite deep learning; fundamental to text pipelines.

---

# **6. Sparse vs. Dense Word Representations**

## **Sparse Representations**

- **Definition:** High-dimensional vectors with mostly **zero** entries.
  - Example: **one-hot vectors**, **bag-of-words**.
- **Pros:**
  - Simple, interpretable.
- **Cons:**
  - No notion of **similarity** or **semantic structure**.
  - Huge dimensionality → inefficient.
  - Sparse data problem persists.

## **Dense Representations (Embeddings)**

- **Definition:** Low-dimensional, continuous vectors learned by models.
- **Properties:**
  - Capture **semantic similarity** (e.g., _king_ and _queen_ close in vector space).
  - Enable downstream models to generalise.

### **Relevant NLP tasks**

- **LMs**, **classification**, **translation**, **similarity search**, **retrieval**.

---

# **7. Vector-Based Similarity Measures**

### **Dot Product**

- Measures alignment: $\vec{a} \cdot \vec{b}$.
- Large positive = similar direction.

### **Cosine Similarity**

- **Scale-invariant** measure of angle:
  $$
  \cos(\theta) = \frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}
  $$
- Most common for embeddings.

### **Euclidean Distance**

- Measures absolute distance in space.
- Sensitive to embedding magnitude.

### **Task examples**

- **Nearest-neighbour word similarity**
- **Sentence retrieval**
- **Clustering** (e.g., semantic categories)

---

# **8. Sentence-level Embeddings**

### **Definition**

Represent entire sentences as **single dense vectors**.

### **Methods**

- **Pooling over token embeddings** (mean/max).
- **Sentence Transformers** (e.g., SBERT).
- **LLM hidden-state averaging**.

### **Uses**

- **Semantic similarity**
- **Clustering / retrieval**
- **Information retrieval (IR)**
- **Document classification**
- **Duplicate question detection** (e.g., Quora dataset)

### **Why important**

They make long-text comparison **computationally tractable** without expensive cross-attention.

---

# **TL;DR**

- **Zipf’s Law** → rare events dominate; leads to **sparse data**, motivates **smoothing** & **subword models**.
- **Train/dev/test** → prevent overfitting; evaluate generalisation.
- **LLM development** = _pre-training → SFT → RLHF → RLVR → fine-tuning_.
- **Inference** = _KV cache prefill → autoregressive decoding_.
- **Regex** → rule-based string patterns, still essential.
- **Sparse vs dense embeddings** → dense vectors encode semantics; sparse are simple but limited.
- **Similarity metrics** → cosine sim is dominant.
- **Sentence embeddings** → enable semantic search, retrieval, classification.

## **1. Ambiguity (multiple forms)**

**Core idea:** A single surface form can map to multiple interpretations.  
**Why it matters:** NLP systems must resolve ambiguity to correctly parse, classify, translate, or answer questions.

### **Types**

- - **Lexical ambiguity (word sense):**  
    _bank_ = _financial institution_ vs _river edge_.  
    Relevant to: **WSD**, MT, IR.
- **Parts of speech ambiguity:**  
   _flies_ = noun (“insects”) or verb (“he flies”).  
   Relevant to: **tagging**, morphological analyzers.
- Morphological ambiguity:
  - "Untiable" = able to be untied OR not able to be tied?
- **Syntactic (structural) ambiguity:**  
   _I saw an elephant in my pyjamas._  
   (Attachment of PP unclear)
- **Attachment ambiguity:**  
   _I saw the man with the telescope._  
   Relevant to: **parsing**, semantic role labelling.
- **Word-order ambiguity:**  
   _Old men and women._  
   Relevant to: **parsing**, MT.
- **Referential ambiguity**
  _Mary told Jane she should leave._ (“she” unclear)
- **Idiomatic ambiguity:**  
   _kick the bucket_ → literal vs idiomatic.

---

## **2. Agreement**

**Core idea:** Certain grammatical features must match across elements.

### **Types of agreement**

- **Verb–argument agreement:**
  - _She runs_ vs _They run_.
  - Relevant to: **syntactic parsing**, LM scoring.
- **Co-reference agreement:**
  - Pronouns must match antecedent number/gender: _Mary said she…_
  - Relevant to: **coreference resolution**.
- **Language-specific agreement features:**
  - **Case** (German, Russian), **gender** (Spanish), **number**.
  - Relevant to: MT, parsing, morphology modelling.

### **Long-distance dependencies**

- Agreement can span clauses:
  - _The bouquet of roses **was** lovely._
  - Relevant to: assessing **context windows**, identifying **limitations of n-gram models**.

---

## **3. Word Types vs Tokens & Tokenization**

- **Tokens:** individual surface occurrences in text.
  - _“the cat sat”_ → 3 tokens.
- **Types:** unique word forms in a corpus.
  - In that sentence, 3 types (all unique).

**Tokenization issues:**

- Hyphens, contractions (_don’t → do + n’t_), multiword expressions.
- Relevant to: **BPE**, vocabulary construction, embeddings.

---

## **4. Stems, Affixes, Root, Lemma**

- **Root:** irreducible lexical core (_scrib-_).
- **Stem:** form to which affixes attach (_scrib(e)_).
- **Affixes:** prefixes/suffixes (_un-_, _-ing_).
- **Lemma:** dictionary form (_run_ vs _running_, _ran_).

**Relevant to:** morphological analyzers, MT, IR normalisation.

---

## **5. Inflectional vs Derivational Morphology**

- **Inflectional:** alters grammatical features; **does not change word class**.
  - _walk → walked_ (tense), _cat → cats_ (number).
  - Relevant to: POS tagging, parsing.
- **Derivational:** forms **new lexemes**, often **changes word class**.
  - _happy → happiness_, _teach → teacher_.
  - Relevant to: vocabulary growth, embeddings, MT.

**Case, gender, number marking:**

- Languages vary: Romance (gender), Slavic (case), English (mostly number/tense).

Note: Morphological richness → **data sparsity**, more word forms → harder for n-grams, embeddings, tagging.

---

## **6. Dialects**

- Variation in vocabulary, syntax, spelling, morphology.
- Variety by region, class or culture. typically mutually intelligable.
- Examples: _colour/color_, _you all/y’all_.
- Relevant to: LM robustness, MT, speech-to-text, fairness.

---

## **7. Part-of-Speech (POS)**

- Grammatical categories (noun, verb, adj…).
- Important because POS constrains syntactic structure.
- Ambiguity common: _book_ (noun/verb).
- Relevant to: tagging, parsing, downstream tasks.

---

## **8. Open-class vs Closed-class Words**

- **Open-class:** nouns, verbs, adjectives, adverbs.
  - Semantically rich; new words appear.
  - Important for embeddings, semantics.
- **Closed-class:** determiners, prepositions, pronouns, conjunctions.
  - Rarely grow; grammatical glue.
  - Important for syntax modelling.

---

## **9. Long-Distance Dependencies**

- Dependencies spanning large distances or intervening material.
- Examples:
  - Subject–verb agreement: _The key to the cabinets **is** missing._
  - Wh-movement: _What did John say Mary bought \_\_?_

**Relevance:**

- N-grams fail (limited context).
- Transformers handle via self-attention.

---

## **10. Syntactic Roles**

- **Subject:** agent/performer.
- **Object:** undergoer/patient.
- **Indirect object:** recipient or benefactive.

**Why needed:** semantic role labelling, parsing, MT disambiguation.

---

## **11. Word Senses & Semantic Relations**

- **Synonym:** _big / large_.
- **Hypernym:** _animal_ (hypernym of _dog_).
- **Hyponym:** _poodle_ (hyponym of _dog_).
- **Similarity:** distributional or conceptual closeness.

**Relevance:** WSD, MT, IR, embeddings.

---

## **12. Distributional Hypothesis**

**Definition:**  
_Words that occur in similar contexts tend to have similar meanings._

- Forms basis of **word embeddings**, co-occurrence matrices, skip-gram/CBOW.
- Relevant to: semantics, clustering, similarity tasks.

---

## **13. Static vs Contextualized Embeddings**

### **Static embeddings**

- One vector per word type (e.g., **word2vec**, **GloVe**).
- Cannot capture polysemy (_bank_ has one vector).
- Trained using distributional context.

### **Contextual embeddings**

- One vector per **token** in context (e.g., **ELMo**, **BERT**, **GPT**).
- Capture polysemy, subtle syntactic/semantic relations.
- Generated via **deep language models** during inference.

**Relevance:** nearly all modern NLP (NER, MT, QA, sentiment).

---

# **TLDR**

- Ambiguity drives most NLP difficulty; know types and examples.
- Agreement involves number/gender/case dependencies; can be long-distance.
- Tokens ≠ types; tokenization matters for BPE and embeddings.
- Morphology: roots, lemmas, inflection vs derivation.
- POS, syntactic roles, open/closed classes guide parsing.
- Word senses + semantic relations feed into WSD, MT, IR.
- Distributional hypothesis underpins embeddings.
- Static embeddings = type-level; contextual embeddings = token-level, context-sensitive.

# **1. Tokenization**

**What it is:**

- Splitting raw text into **tokens** (words, subwords, punctuation).
- Defines the _basic units_ of all downstream NLP models.

**Examples:**

- “don’t” → {“don”, “’”, “t”} (char-based)
- BPE: “unhappiness” → “un”, “happi”, “ness”

**Ambiguity / Difficulty:**

- **Multi-word expressions:** “New York”, “hot dog”.
- **Agglutinative languages:** very long words (e.g., Turkish).
- **No whitespace languages:** Chinese, Japanese → segmentation is non-trivial.

**Algorithms / Methods:**

- Rule-based tokenizers, whitespace tokenizers.
- **Subword models**: BPE, WordPiece, SentencePiece.

**Evaluation:**

- Intrinsically rare; typically evaluated indirectly via downstream task performance.

**Task Structure:**

- **Preprocessing step**, not a prediction task.

---

# **2. Language Modelling (LM)**

**What it is:**

- Predicting the **next token** given previous context:  
   **P(wₜ | w₁ … wₜ₋₁)**.

**Examples:**

- Predict the next word: “The cat sat on the \_\_\_”.

**Ambiguity / Difficulty:**

- **Long-distance dependencies:** LM must capture grammar agreement, discourse cues.
- **Sparsity:** Rare n-grams severely weaken count-based models.
- **Unbounded vocabulary / OOV:** mitigated by subword tokenization.

**Algorithms / Methods:**

- **N-gram models (+ smoothing).**
- **Neural LMs:** RNN, LSTM, GRU.
- **Transformers:** GPT-style causal LMs.

**Evaluation:**

- **Perplexity** (exp of average negative log-likelihood).
- Intrinsic, task-specific.

**Task Structure:**

- **Sequence prediction** (autoregressive).
- **Generative model.**

---

# **3. Text Categorization (Sentiment, Topic Classification, etc.)**

**What it is:**

- Assigning one label (or set of labels) to a text span.
- Examples: **Sentiment** (positive/negative), **Spam detection**, **Topic** (sports, politics).

**Ambiguity / Difficulty:**

- **Sarcasm / Irony:** “Great job…” (negative sentiment).
- **Domain shift:** model trained on movie reviews performs poorly on finance tweets.
- **Multi-label vs single-label:** deciding task structure affects modelling.

**Algorithms / Methods:**

- Logistic regression, linear SVMs.
- Bag-of-words, TF-IDF, embeddings → classifiers.
- Fine-tuned transformers (BERT, RoBERTa).

**Evaluation:**

- **Accuracy**, **Precision/Recall/F1**, **Confusion matrix**.
- Macro-F1 for imbalanced classes.

**Task Structure:**

- **Classification** (single label) or **multi-label classification**.

---

# **4. Word Sense Disambiguation (WSD)**

**What it is:**

- Assigning the correct **sense** of a polysemous word in context.
- Example: “bank” = _riverbank_ vs _financial institution_.

**Ambiguity / Difficulty:**

- **Lexical ambiguity** is large and context-dependent.
- Senses in resources (WordNet) may not align with real usage.
- Many senses are extremely rare → sparse data problem.

**Algorithms / Methods:**

- Knowledge-based: Lesk algorithm (overlap).
- Supervised classifiers using context windows.
- Transformer contextual embeddings (state-of-the-art).

**Evaluation:**

- Sense-level **accuracy** vs a gold dataset (e.g., SemCor).

**Task Structure:**

- **Classification** over senses of a single target word.

---

# **5. Sequence-to-Sequence Tasks**

_(Machine translation, summarization, data-to-text, style transfer)_

**What they are:**

- Input sequence → **output sequence**, often of different length.
- Examples:
  - **Machine Translation:** “Je suis fatigué” → “I am tired.”
  - **Summarization:** long article → short abstract.

**Ambiguity / Difficulty:**

- **Multiple correct outputs** → evaluation is hard.
- **Alignment** between source and target tokens is implicit and complex.
- **Long-range reasoning** required (especially summarization).
- **Hallucinations** in generative models.

**Algorithms / Methods:**

- Seq2Seq RNNs w/ attention (Bahdanau, Luong).
- Transformer encoder–decoder (e.g., T5, BART).
- Causal LMs with prompt-based decoding.

**Evaluation:**

- **BLEU**, **ROUGE**, **METEOR** (n-gram overlap).
- Increasingly: human evaluation, BERTScore.

**Task Structure:**

- **Sequence-to-sequence generation**, conditional generation.

---

# **6. Open-Ended Conversational AI**

**What it is:**

- Systems that generate **contextually appropriate, multi-turn dialogue**.
- Not a closed classification task—output space is unbounded.

**Examples:**

- Chatbots, customer support assistants, tutoring systems.

**Ambiguity / Difficulty:**

- **Ambiguous intent**: user messages underspecified.
- **Long context tracking**: maintaining state across turns.
- **Safety & grounding**: misinformation, hallucinations, harmful responses.
- **Evaluation** inherently subjective.

**Algorithms / Methods:**

- Large Language Models (GPT-style).
- Retrieval-augmented generation (RAG).
- Reinforcement learning (e.g., RLHF, RLVR).

**Evaluation:**

- Hard! Typically:
  - **Human judgments** (quality, relevance, safety).
  - **Automated metrics** (BLEU, embedding similarity) but imperfect.
  - **Task-specific** scoring (goal completion in task-oriented agents).

**Task Structure:**

- **Open-ended generation** (not finite-label).
- Often modeled as **sequence-to-sequence** or **autoregressive LM** with memory.

---

# **7. Identifying Task Structures (Meta-Skill)**

You should be able to recognise, for a **newly described task**, which structure it fits:

| Task                            | Structure                              |
| ------------------------------- | -------------------------------------- |
| POS tagging                     | **Sequence labelling**                 |
| Named entity recognition        | **Sequence labelling**                 |
| Spam detection                  | **Classification**                     |
| MT / Summarization              | **Seq2Seq**                            |
| Dialogue generation             | **Open-ended generation**              |
| Question answering (extractive) | **Span prediction**                    |
| Coreference resolution          | **Clustering / structured prediction** |

Key cues:

- **Is the output one label?** → classification.
- **Label per token?** → sequence labelling.
- **Output is a new sequence?** → seq2seq.
- **Output is unconstrained free text?** → generative LM task.

---

# **TLDR — Core Expectations**

- Know **what each task is**, with **examples**, **ambiguities**, **difficulty sources**.
- Know **typical algorithms**: N-grams, logistic regression, transformers, seq2seq.
- Know **evaluation metrics**: accuracy, F1, perplexity, BLEU/ROUGE.
- For **new tasks**, quickly identify if it's **classification / seq-labelling / seq2seq / open-gen**.
