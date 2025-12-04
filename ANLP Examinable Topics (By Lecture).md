
## W101 - Introduction

### Topics Covered:

- **Ambiguity** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why

---

## W102 - Words as Data

### Topics Covered:

- **Zipf's Law and sparse data** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain what Zipf's law is and its implications, discuss what "sparse data" refers to, discuss these with respect to specific tasks
- **Part-of-Speech** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why
- **Open-class Words, Closed-class Words** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain these concepts, give examples, identify examples, say what NLP tasks these are relevant to and why
- **Tokenization** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss cases of ambiguity or what makes the task difficult
- **Corpora: Issues in collection, annotation, distribution** (Section 10: Evaluation)
    - _You should be able to:_ identify issues involved in collection, annotation and distribution

---

## W103 - Morphology

### Topics Covered:

- **Stems, Affixes, Root, Lemma** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain these concepts, give examples, identify examples, say what NLP tasks these are relevant to and why
- **Inflectional and derivational Morphology** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain these concepts, give examples, identify examples, say what NLP tasks these are relevant to and why
- **Byte-Pair Encoding (BPE) algorithm** (Section 5: Algorithms)
    - _You should be able to:_ explain what it computes (input and output), what it is used for, hand simulate it
- **Tokenization** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss cases of ambiguity or what makes the task difficult

---

## W201 - Probability, Models, and Data

### Topics Covered:

- **N-gram models** (Section 2: Generative models)
    - _You should be able to:_
        - Describe the generative process and write down the formula for joint probability
        - Compute the probability of sequences (assuming known parameters)
        - Explain how the model is trained
        - Give examples of tasks it could be applied to and how
        - Say what it can and cannot capture about natural language, with failure mode examples
- **Bayes' Rule** (Section 4: Formulas)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately
- **Maximum Likelihood Estimation** (Section 4: Formulas & Section 6: Mathematical concepts)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately; understand the conceptual differences and motivation, use formulas if given
- **Prior and likelihood** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain what these refer to (in general and in specific models), explain their role in a probabilistic model

---

## W202 - N-Grams

### Topics Covered:

- **N-gram models** (Section 2: Generative models)
    - _You should be able to:_
        - Describe the generative process and write down the formula for joint probability
        - Compute the probability of sequences (assuming known parameters)
        - Explain how the model is trained
        - Give examples of tasks it could be applied to and how
        - Say what it can and cannot capture about natural language, with failure mode examples
- **Maximum Likelihood Estimation** (Section 4: Formulas)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately
- **Language modelling** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss difficulty; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used
- **Perplexity** (Section 10: Evaluation)
    - _You should be able to:_ explain what this measures, what tasks it would be appropriate for, and why

---

## W203 - Smoothing and Sampling

### Topics Covered:

- **Add-One / Add-Alpha Smoothing** (Section 4: Formulas & Section 6: Mathematical concepts)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately; understand pros and cons, characteristic errors, when methods are acceptable/unacceptable
- **Interpolation (for smoothing)** (Section 4: Formulas & Section 6: Mathematical concepts)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately; understand conceptual differences and motivation
- **Training, development, and test sets** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain how these are used and for what reason, explain their application to particular problems

---

## W301 - Text Classification with Logistic Regression

### Topics Covered:

- **Naive Bayes classifier** (Section 2: Generative models)
    - _You should be able to:_
        - Describe the generative process and write down the formula for joint probability
        - Compute the probability of sequences (assuming known parameters)
        - For models with latent variables, compute the most probable class for a particular input
        - Explain how the model is trained
        - Give examples of tasks it could be applied to and how
        - Say what it can and cannot capture about natural language
- **Logistic regression (multinomial)** (Section 3: Discriminative models)
    - _You should be able to:_
        - Understand the formula for computing conditional probability P(class|observations)
        - Apply the formula given features and weights
        - Determine which class is most probable
        - Give examples of tasks and what features might be useful
        - Explain at a high level what training aims to achieve
        - Explain how training differs from generative models
        - Explain the role of regularization and when it's most important
        - Discuss pros and cons of discriminative vs generative models
- **Bayes' Rule** (Section 4: Formulas)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately
- **Text categorization** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss difficulty; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used
- **Sentiment analysis** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss difficulty; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used

---

## W302 - Training Logistic Regression and Evaluation

### Topics Covered:

- **L2 regularisation** (Section 4: Formulas)
    - _You should be able to:_ know the formula, what it may be used for, apply it appropriately, discuss strengths and weaknesses
- **Precision, recall, and F-measure** (Section 4: Formulas & Section 10: Evaluation)
    - _You should be able to:_ know the formulas, what they may be used for, apply them appropriately; explain what each measures, what tasks they're appropriate for, and why
- **Cross-entropy loss** (Section 6: Mathematical concepts)
    - _You should be able to:_ understand the conceptual differences and motivation, use the formula if given
- **Stochastic Gradient Descent** (Section 6: Mathematical concepts)
    - _You should be able to:_ understand the conceptual differences and motivation, use the formula if given
- **Training, development, and test sets** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain how these are used and for what reason, explain their application to particular problems
- **Accuracy** (Section 10: Evaluation)
    - _You should be able to:_ explain what this measures, what tasks it would be appropriate for, and why
- **Intrinsic vs. extrinsic evaluation** (Section 10: Evaluation)
    - _You should be able to:_ explain the difference and give examples of each for particular tasks

---

## W303 - Lexical Semantics

### Topics Covered:

- **Word Senses and relations (synonym, hypernym, hyponym, similarity)** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain these concepts, give examples, identify examples, say what NLP tasks these are relevant to and why
- **Distributional hypothesis** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why
- **WordNet** (Section 9: Resources)
    - _You should be able to:_ describe what linguistic information is captured, how it might be used in an NLP system
- **Word sense disambiguation** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss cases of ambiguity or what makes the task difficult; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used

---

## W401 - Dense Word Embeddings

### Topics Covered:

- **Dot product, cosine similarity, Euclidean distance** (Section 4: Formulas)
    - _You should be able to:_ know the formulas, what they may be used for, apply them appropriately, discuss strengths and weaknesses
- **Sparse and dense vector representations / word embeddings** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain these concepts, give examples, say what NLP tasks these are relevant to and why
- **word2vec/skipgram** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain this concept, give examples, say what NLP tasks this is relevant to and why
- **Negative sampling** (Section 6: Mathematical concepts)
    - _You should be able to:_ understand the conceptual differences and motivation, use the formula if given
- **Vector-based similarity measures** (Section 6: Mathematical concepts)
    - _You should be able to:_ explain these concepts, give examples, say what NLP tasks these are relevant to and why
- **Distributional hypothesis** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why

---

## W402 - Multilayer Perceptrons (MLPs)

### Topics Covered:

- **MLP language models** (Section 2: Generative models)
    - _You should be able to:_
        - Describe the generative process and write down the formula for joint probability
        - Compute the probability of sequences (assuming known parameters)
        - Explain how the model is trained
        - Give examples of tasks it could be applied to and how
        - Say what it can and cannot capture about natural language, with failure mode examples
- **MLP classifiers** (Section 3: Discriminative models)
    - _You should be able to:_
        - Understand the formula for computing conditional probability P(class|observations)
        - Apply the formula given features and weights
        - Give examples of tasks and what features might be useful
        - Explain at a high level what training aims to achieve
        - Explain how training differs from generative models
        - Explain the role of regularization and when it's most important
        - Discuss pros and cons of discriminative vs generative models
- **Language modelling** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss difficulty; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used

---

## W403 - Training Neural Nets

### Topics Covered:

- **Backpropagation** (Section 6: Mathematical concepts)
    - _You should be able to:_ understand the conceptual differences and motivation, use the method if formulas/process are given

---

## W501 - RNNs

### Topics Covered:

- **RNN language models** (Section 2: Generative models)
    - _You should be able to:_
        - Describe the generative process and write down the formula for joint probability
        - Compute the probability of sequences (assuming known parameters)
        - Explain how the model is trained
        - Give examples of tasks it could be applied to and how
        - Say what it can and cannot capture about natural language, with failure mode examples
- **RNN classifiers** (Section 3: Discriminative models)
    - _You should be able to:_
        - Understand the formula for computing conditional probability P(class|observations)
        - Apply the formula given features and weights
        - Give examples of tasks and what features might be useful
        - Explain at a high level what training aims to achieve
        - Explain the role of regularization and when it's most important
- **Language modelling** (Section 8: Tasks)
    - _You should be able to:_ explain this task, give examples, discuss difficulty; say what algorithm(s) or method(s) can be used; what evaluation method(s) are typically used

---

## W502 - Dialect and Discrimination

### Topics Covered:

- **Dialects** (Section 7: Linguistic concepts)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why
- **Algorithmic bias** (Section 11: Ethical issues)
    - _You should be able to:_ identify and briefly discuss this issue, provide examples, discuss possible ways to mitigate it, say how it's relevant when presented with an example task
- **Direct vs. indirect discrimination** (Section 11: Ethical issues)
    - _You should be able to:_ identify and briefly discuss this issue, provide examples, discuss possible ways to mitigate it, say how it's relevant when presented with an example task

---

## W503 - Discrimination and Data Ethics

### Topics Covered:

- **Legal and ethical issues in resource creation and collection** (Section 9: Resources)
    - _You should be able to:_ identify legal and ethical issues in the creation and collection of linguistic resources
- **Corpora: Issues in collection, annotation, distribution** (Section 10: Evaluation)
    - _You should be able to:_ identify issues involved in collection, annotation and distribution

---

## W601 - Attention in Seq2Seq

_No exam topics explicitly covered (Attention mechanisms not in exam guidelines)_

---

## W602 - Self-Attention in Transformers

_No exam topics explicitly covered (Transformers not in exam guidelines)_

---

## W603 - Transformer Architecture

_No exam topics explicitly covered (Transformers not in exam guidelines)_

---

## W701 - Transformer Inputs and Outputs

_No exam topics explicitly covered (Transformers not in exam guidelines)_

---

## W702 - Transfer Learning and BERT

_No exam topics explicitly covered (BERT not in exam guidelines)_

---

## W703 - Architectures for Language Models

_No exam topics explicitly covered (T5/GPT not in exam guidelines)_

---

## W801 - In Context Learning

_No exam topics explicitly covered (In-context learning not in exam guidelines)_

---

## W802 - Scaling Laws and Evals

_No exam topics explicitly covered (Scaling laws not in exam guidelines)_

---

## W803 - Memory and Compression

_No exam topics explicitly covered (Memory/compression not in exam guidelines)_

---

## W901 - Multilingual LLMs

_No exam topics explicitly covered (Multilingual LLMs not in exam guidelines)_

---

## W902 - Instruction Tuning

_No exam topics explicitly covered (Instruction tuning not in exam guidelines)_

---

## W903 - RLHF

### Topics Covered:

- **Verifiability** (Section 7: Linguistic concepts - RLVR section)
    - _You should be able to:_ explain this concept, give examples, identify examples, say what NLP tasks this is relevant to and why

---

## W1010 - From Reference Book Topics
## Section 2: Generative Probabilistic Models

### Hidden Markov Model

- _You should be able to:_
    - Describe the generative process and write down the associated formula for the joint probability of latent and observed variables
    - Compute the probability of (say) a tag-word sequence (assuming you know the model parameters)
    - For models with latent variables, compute the most probable tag sequence for a particular input, hand-simulating any algorithms that might be needed
    - Explain how the model is trained
    - Give examples of tasks the model could be applied to, and how it would be applied
    - Say what the model can and cannot capture about natural language, ideally giving examples of its failure modes

### Probabilistic Context-Free Grammar

- _You should be able to:_
    - Describe the generative process and write down the associated formula for the joint probability of latent and observed variables
    - Compute the probability of a parse tree (assuming you know the model parameters)
    - For models with latent variables, compute the most probable parse tree for a particular input, hand-simulating any algorithms that might be needed
    - Explain how the model is trained
    - Give examples of tasks the model could be applied to, and how it would be applied
    - Say what the model can and cannot capture about natural language, ideally giving examples of its failure modes

---

## Section 5: Algorithms and Computational Methods

### Viterbi Algorithm

- _You should be able to:_
    - Explain what it computes (its input and output)
    - Explain what it is used for
    - Hand simulate the algorithm

### CKY Algorithm

- _You should be able to:_
    - Explain what it computes (its input and output)
    - Explain what it is used for
    - Hand simulate the algorithm

### Arc-Standard Transition-Based Parsing Algorithm

- _You should be able to:_
    - Explain what it computes (its input and output)
    - Explain what it is used for
    - Hand simulate the algorithm

### Finite State Automata and Transducers

- _You should be able to:_
    - Explain what each method computes (its input and output)
    - Explain what it is used for
    - Hand simulate each one

---

## Section 6: Additional Mathematical and Computational Concepts

### Contrastive Learning (explicit coverage)

- _You should be able to:_
    - Understand the conceptual differences and motivation behind this method
    - Should be able to use the formulas if they are given to you
    - _Note:_ Related concepts appear in W401 (negative sampling) but not explicitly as "contrastive learning"

### Pointwise Mutual Information

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### TF-IDF

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Regular Language and Regular Expressions

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

---

## Section 7: Linguistic and Representational Concepts

### Context-Free Grammar (detailed coverage)

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why
    - _Note:_ Mentioned briefly in lectures but not covered in detail

### Chomsky Normal Form

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Terminal and Non-Terminal (Phrasal) Categories

- _You should be able to:_
    - Explain these concepts
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks these are relevant to and why

### Dependency Syntax

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Projective and Non-Projective Dependencies

- _You should be able to:_
    - Explain these concepts
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks these are relevant to and why

### Lexical Head Words (in syntax)

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Meaning Representations (MR)

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### First Order Logic

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Compositionality

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

### Quantifiers and Quantifier Scoping

- _You should be able to:_
    - Explain these concepts
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks these are relevant to and why

### Coreference and Anaphora

- _You should be able to:_
    - Explain these concepts
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks these are relevant to and why

### Referring Expression

- _You should be able to:_
    - Explain this concept
    - Give one or two examples where appropriate
    - Be able to identify examples if given to you
    - Say what NLP tasks this is relevant to and why

---

## Section 8: Tasks

### Part-of-Speech Tagging (detailed algorithms)

- _You should be able to:_
    - Explain this task
    - Give one or two examples where appropriate
    - Discuss cases of ambiguity or what makes the task difficult
    - Say what algorithm(s) or general method(s) can be used to solve the task
    - Say what evaluation method(s) are typically used
    - _Note:_ Part-of-speech mentioned in W102, but tagging algorithms not covered

### Span Labeling and BIO Tagging

- _You should be able to:_
    - Explain this task
    - Give one or two examples where appropriate
    - Discuss cases of ambiguity or what makes the task difficult
    - Say what algorithm(s) or general method(s) can be used to solve the task
    - Say what evaluation method(s) are typically used

### Syntactic Parsing

- _You should be able to:_
    - Explain this task
    - Give one or two examples where appropriate
    - Discuss cases of ambiguity or what makes the task difficult
    - Say what algorithm(s) or general method(s) can be used to solve the task
    - Say what evaluation method(s) are typically used

---

## Section 9: Resources

### Penn Treebank

- _You should be able to:_
    - Describe what linguistic information is captured in this resource
    - Explain how it might be used in an NLP system

### Universal Dependencies

- _You should be able to:_
    - Describe what linguistic information is captured in this resource
    - Explain how it might be used in an NLP system

---

## Section 11: Ethical Issues

### Representational vs. Allocational Harm (explicit definition)

- _You should be able to:_
    - Identify and briefly discuss this ethical issue arising from developing or deploying NLP tools
    - Provide some examples of each
    - Discuss possible ways to mitigate them
    - Say how they might be relevant when presented with an example task
    - _Note:_ Related concepts discussed in W502 (Dialect and Discrimination) but not explicitly defined with this terminology