# ANLP Lecture Notes

## W101 - Introduction

### What is Natural Language Processing (NLP)?

- **Definition:** The processing of "natural" (human) language, distinct from computer languages.
- **Scope:**
  - Everyday language includes text, speech, and sign language.
  - This course specifically focuses on **text** processing.
- **System Pipeline:** A typical deployed system (e.g., a dialogue system) separates components:
  - Input (Speech) $\rightarrow$ Automatic Speech Recognition (ASR) $\rightarrow$ **NLP System (Text)** $\rightarrow$ Text-to-Speech (TTS) $\rightarrow$ Output.

### Challenges of Learning from Text

- **Ambiguity:** Languages contain ambiguities at many levels that require context to resolve.
- **Complexity:** Language follows rules, but these rules have many exceptions.
- **Infiniteness & Sparsity:**
  - Language is infinite; we cannot see examples of every possible sentence.
  - Most data we observe is rare (sparse).

### Ambiguity

- **Ambiguity in Humour:** Jokes often rely on double meanings.
  - _Example 1:_ "I hate all change." (Ambiguity: Coins vs. Transformation).
  - _Example 2:_ "I shot an elephant in my pyjamas." (Ambiguity: Who is wearing the pyjamas?).
- **Types of Ambiguity:**
  - **Lexical:** Ambiguity regarding the meaning of a specific word (e.g., "change").
  - **Syntactic (Structural):** Ambiguity in the relationship between words or the sentence structure (e.g., does "in my pyjamas" attach to "I" or "elephant"?).
  - **Referential:** Ambiguity regarding who/what a word refers to (e.g., "She said it was important" — who is "she"?).
  - **Idiomatic:** Confusion can arise between literal meanings and idiomatic expressions (e.g., "kick the bucket").
- **Resolving Ambiguity:**
  - Humans typically resolve ambiguity using context (sentence or discourse level) and world knowledge.
    - _Example:_ "I cooked the fish in the freezer." World knowledge clarifies the cooking did not happen _inside_ the freezer.

## W102 - Words as data

### Introduction & Challenges of Text Data

- **Recap of challenges:** Learning from text is difficult because natural language is **ambiguous**, has complex structures (rules with exceptions), and is effectively infinite.
- **Focus of this lecture:** Examining the statistical distribution of words and the internal structure of words (morphology).
- **Defining a "Word":**
  - **Simple definition**: Strings separated by spaces.
  - **Issues:** Punctuation, compounds (e.g., "high-risk"), and languages without spaces (e.g., Chinese) make this definition difficult.
  - **Tokenisation:** The process of splitting text into separate items (tokens) for processing. Modern systems often use sub-word units rather than whole words.

### Word Frequencies & Distribution

- **Europarl Corpus Example:** 24 million words from European Parliament proceedings.
- **Frequency Hierarchy:**
  - **Most frequent:** Function words (the, of, to, and).
  - **Most frequent nouns:** Topic-specific (European, Mr, Commission).
  - **The Long Tail:** Over 1/3 of distinct words occur only _once_ (hapax legomena). This includes rare words, specific entities, and errors (e.g., typos like "policyfor").
- **Key Terminology:**
  - **Word Type:** Unique strings/lexical items (vocabulary size).
  - **Word Token:** Total number of word instances (running text length).
    - _Example:_ "a cat and a brown dog chased a black dog" = 10 tokens, 7 types.

### Zipf's Law

- **Plotting Frequency:** When plotting Frequency vs. Rank on standard axes, the data is unreadable due to the massive drop-off.
  - **Rank:** the position of a word when you sort the vocabulary by decreasing frequency
- **Logarithmic Scale:** Plotting on log-log axes reveals a linear pattern across multiple languages (English, Spanish, Finnish, German).
- The Equation:
  $$f \times r \approx k$$
  Where $f$ is frequency, $r$ is rank, and $k$ is a constant.
  - In log space: $\log f = \log k - \log r$ (creates a line with a negative slope).
- **Implications for NLP:**
  - **Sparsity:** Most words are rare.
  - **Data Scaling:** Gathering more data does not solve sparsity; it just uncovers _more_ rare words.
  - **Shared Distribution:** distribution applies to other structures too (n-grams, syntax).

### Corpora (Data Sources)

- **Definition:** A corpus is a collection of text, often including metadata (author, date) and annotations (labels, syntax trees).
- **Variations in Text:**
  - **Language/Variety:** Dialects (e.g., Scottish English) and code-switching (mixing languages like Hindi/English).
  - **Genre:** Newswire vs. fiction vs. social media.
  - **Demographics:** Author age, gender, etc.
- **Corpus Types:**
  - **Classic:** Brown (balanced), WSJ (news), BNC (balanced written/spoken).
  - **New/Web:** Common Crawl (web scrape), Wikipedia, OpenSubtitles.
- **Best Practice:** **Corpus Datasheets** should document motivation, collection process, consent, and annotation details to understand biases and limitations.

### Word Classes

- **Function Words (Closed Class):**
  - Grammatical glue (prepositions, conjunctions, determiners, pronouns).
  - High frequency, stable over time (new ones rarely invented).
  - Similar across corpora of the same language.
- **Content Words (Open Class):**
  - Carry primary meaning (nouns, verbs, adjectives, adverbs).
  - Frequency depends heavily on topic/genre.
  - "Open" because new words are constantly created (e.g., "selfie").
- **Ambiguity:** Words can belong to multiple classes (e.g., "report" can be a noun or a verb).

### Cross-Linguistic Differences & Morphology

- **Comparison:** While Zipf's law shape is similar, the _slope_ differs by language.
  - _Example:_ Finnish has ~10x more distinct word types than English for the same number of sentences.
  - _Reason:_ **Morphology**.
- **Morphology Definition:** The study of word internal structure and how morphemes (smallest meaningful units) combine.
- **Why it matters for NLP:**
  - **Information Retrieval:** Matching related forms (play, played, player).
  - **Prediction:** Helping predict unseen words based on structure.
  - **Translation:** Crucial for expressing meaning correctly.

### Morphology in Action: Case and Gender

- **Case Marking (Role Indication):**
  - Some languages (e.g., Russian) use suffixes (morphology) to indicate who did what to whom.
  - _English:_ Uses word order/prepositions. Swapping noun positions changes meaning.
  - _Russian:_ Can swap word order while retaining meaning because the suffix marks the role (subject vs. object).
- **Grammatical Gender:**
  - **Agreement:** Verbs or adjectives must match the noun's gender.
  - **Translation Bias:** Translating "The doctor gave..." into Russian requires a gendered verb. Systems often default to male due to training data bias, even when context implies female (e.g., "gave _her_ patient").
- **Noun Classes:**
  - Linguistic "gender" means "type" or "class," not necessarily biological sex.
  - Some languages have 5+ genders/noun classes (e.g., Bantu languages like Swahili/Zulu).

## W103 - Morphology

### Tokenisation Challenges & Motivation

- **Word Tokenisation:** Standard word tokenisation faces the "sparse data problem" where systems encounter rare or unseen words regardless of training data size.
- **Character Tokenisation:** While using characters reduces the vocabulary size, it creates very long sequences (e.g., a 4-word sentence becomes 20 character tokens).
  - This causes computational difficulties in modelling.
- **Unicode:** Systems must handle a vast range of characters from different languages.
  - Proper encoding (Unicode) is essential for multilingual models.

### Morphology Basics

- **Morphemes:** The smallest meaningful units of speech.
  - Using morphemes as tokens is a potential middle ground between words and characters.
  - They are meaningful and occur more frequently than full words, helping with data sparsity.
- **Concatenative Morphology:** Most languages form words by sticking morphemes together.
  - **Stems:** Convey the core meaning (e.g., _small_, _cat_, _walk_).
  - **Affixes:** Modify the meaning (e.g., _-ed_, _un-_).
- **Affix Positions:**
  - **Suffix:** After the stem.
  - **Prefix:** Before the stem.
  - **Infix:** Inserted into the stem (rare in English).
  - **Circumfix:** Surrounds the stem (e.g., German _ge- -t_).

### Lemma vs. Stem

- **Lemma:** The canonical or dictionary form of a word.
  - Example: _fly_, _flies_, _flew_, and _flying_ all share the lemma **fly**.
  - _Walker_ has the lemma _walker_ (lemmas are words themselves).
- **Stem:** The part of the word shared by variants; may not be a complete word.
  - Example: _produc_ is the stem of _production_.
  - NLP tools include **Lemmatizers** (return lemmas) and **Stemmers** (return stems).

### Complex Morphology Types

- **Non-Concatenative Morphology:**
  - **Reduplication:** Repeating a word to change meaning (e.g., Bahasa Indonesian repeats a word to pluralise it).
  - **Root and Pattern:** Interleaving vowels and consonants (common in Semitic languages like Arabic).
- **Irregularities:**
  - Concatenation often involves spelling changes (e.g., _happy_ $\rightarrow$ _happiness_) or sound changes.
  - Pronunciation of plurals changes based on the preceding sound (e.g., _cats_ [s] vs. _dogs_ [z]).
  - Irregular forms (e.g., _run/ran_) are often the most frequent words in a language.

### Inflectional vs. Derivational Morphology

- **Inflectional Morphology:**
  - Does **not** change the basic meaning or part of speech.
  - Expresses grammatical features like agreement, number, or tense.
  - Generally applies to all words of a specific part of speech.
  - **Agreement Issues:** Agreement (e.g., verb matching subject) creates "long-distance dependencies" where the verb must agree with a subject far away in the sentence, ignoring intervening nouns.
  - **Examples:** `(cat(s))`,`(walk(ing))`, `(play(ed))`, `((run)s)`
- **Derivational Morphology:**
  - Can change the part of speech (e.g., _noun_ $\rightarrow$ _verb_) or meaning.
  - Is "picky": Applies to some words but not others (e.g., _intractability_ is preferred over _intractableness_).
  - Usually applies closer to the stem than inflectional affixes (e.g., _govern_ + _ment_ + _s_).
  - Allows for the creation of complex, valid words even if they haven't been seen before (e.g., _wordificationist_).
  - **Examples**: `(visual(ise))`, `(teach(er))`, `((govern(ment))s)`

### Ambiguity

- **Stem Ambiguity:** Same spelling, different meanings (e.g., _bank_ as finance vs. river).
- **Affix Ambiguity:**
  - _-s_ can be a plural noun marker (_dogs_) or a 3rd person verb marker (_swims_).
  - _'s_ can mean _is_ or _has_.
- **Structural Ambiguity:** The order of combination matters.
  - Example: _Untieable_ could mean "able to be untied" or "unable to be tied" depending on grouping.

### Cross-Lingual Variation

- Languages vary significantly in affix usage (prefixing vs. suffixing).
- **English:** Relatively morphologically poor; uses fixed word order or prepositions instead of complex cases.
- **Finnish:** Morphologically rich (15+ cases), resulting in many more unique word types compared to English
- **Bias Risk:** Multilingual systems trained primarily on English may struggle with languages that use gender/case agreement heavily.

### Sub-word Tokenisation (The Modern Solution)

- **The Problem with Morphemes:** Pure morphological segmentation is difficult to implement for every language and doesn't solve the issue of unseen stems.
- **Data-Driven Approach:** Modern systems use "sub-word" units found via algorithms, which may not strictly correspond to linguistic morphemes.
- **Byte Pair Encoding (BPE):** A popular algorithm for finding sub-word units.

#### BPE Algorithm: Learner Mode (Training)

1. **Initialise:** Start with a vocabulary of all individual characters and add an "end-of-word" symbol to data.
2. **Iterate:**
   - Count all adjacent symbol pairs in the data.
   - Merge the most frequent pair into a new single symbol.
   - Add this new symbol to the vocabulary.
3. **Stop:** Repeat until the vocabulary reaches a desired size.

#### BPE: Training Example

_Tiny corpus: “low lower lowest”; initial tokens = characters._

1. **Initialise:**  
   Vocabulary = `{▁, l, o, w, e, r, s, t}`.  
   Words: `▁ l o w`, `▁ l o w e r`, `▁ l o w e s t`.
2. **Iterate:** Example Merges 3. `l o` → `lo` → `▁ lo w ...` 4. `lo w` → `low` → `▁ low ...` 5. `low e` → `lowe` → `▁ lowe ...`
3. **Stop:** When vocab reaches target size.

#### BPE Algorithm: Segmenter Mode (Inference)

1. Take the ordered list of merges learned during training.
2. Apply merges to new data in that exact order.
3. **Note:** Segmentation relies on the fixed merge rules, not the frequency of tokens in the _new_ data.

#### BPE: Inference Example

**Given merges:**

1. `l o` → `lo`
2. `lo w` → `low`
3. `low e` → `lowe`

**Example 1 — “lowest”**  
Start: `▁ l o w e s t`  
→ `▁ lo w e s t`  
→ `▁ low e s t`  
→ `▁ lowe s t`

Final: `[▁, lowe, s, t]`

## W201 - Probability, Models, and Data

### Introduction: The "Probability of a Sentence"

- **Chomsky's Critique (1969):** Noam Chomsky famously claimed the notion of "probability of a sentence" is entirely useless under any known interpretation.
- **Modern Rebuttal:**
  - Sentence probabilities are the backbone of modern NLP (e.g., ChatGPT, Google Search).
  - They are useful for language engineering and psycholinguistics (modelling human performance).
- **The "Zero Probability" Fallacy:**
  - **Argument:** A grammatical but unseen sentence (e.g., about an Archaeopteryx) and a nonsense word salad (e.g., "jaggedly trees") both have zero probability if based solely on past occurrence.
  - **Flaw:** "Has never occurred" does not equal "zero probability".
  - **Example:** Events like "my hair turns blue" or "trip to Finland" may be unseen but have different, non-zero likelihoods.
  - **Challenge:** Most sentences are unique/unseen; this makes estimation tricky, not meaningless.

### Probability Theory vs. Estimation

- **Probability Theory:** Solves problems where parameters are known
  - _(e.g., drawing from a jar with a known count of 6 blue and 4 red marbles)_
- **Estimation Theory:** Solves problems where parameters are unknown and must be inferred from data
  - _(e.g., drawing marbles to guess the jar's contents)_
- **Language Modelling:**
  - **Analogy**: Like weather forecasting, we need past data and a model to predict future outcomes (e.g., rain or the next word).
  - **Goal**: Assign a probability to a sentence $x=w_{1}...w_{n}$ using a corpus.
- **Notation:**
  - $P(\vec{w})$: True probability (rarely known).
  - $\hat{P}(\vec{w})$: **_Estimated_** probability.

### Generative Models

- **Definition:** A probabilistic process describing how data is generated.
- **Example Scenario:** Flipping a coin 10 times results in 7 Tails (T) and 3 Heads (H). What is $\hat{P}(T)$?. Example: `T T T T T T T H H H`
- **Model Dependence:** The answer depends on the model assumptions:
  - **Model 1 (Fair Coin):** Assumes fairness regardless of data; $\hat{P}(T) = \hat{P}(H) = 0.5$.
  - **Model 2 (Biased Coin):** Relies purely on observed data; $\hat{P}(T) = \frac{7}{10} = 0.7$.
  - **Model 3 (Prior Knowledge):** Assumes selection from a mix of fair/unfair coins; result is between 0.5 and 0.7.

### Model Structure & Parameters

- **Model = Structure + Parameters**
  - **Structure:** The mathematical form (e.g., _“Assume data is Gaussian”_).
  - **Parameters:** The specific values inside the structure (e.g., mean **μ**, std **σ**).
- Gaussian Example (Heights)
  - **Structure:** $p(x|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)$
- **Parameters:**
  - Changing **μ** → shifts the curve left/right.
  - Changing **σ** → spreads or narrows the curve.
- **Model Criticism (When Structure is Wrong)**
  - If your data is **bimodal** (two peaks) but your structure is a **single Gaussian**, the model _cannot_ capture reality.
  - **Fix:** Use a better structure, e.g. a **Gaussian Mixture Model** with parameters _(μ₁, σ₁, μ₂, σ₂, mixing weight π)_.
  - **Trade-off:** Better realism → more parameters → more data needed.
- **Improving Parameter Estimates**
  - **More Data**: Reduces variance of estimates (but costs time/money).
  - **Better Model Form**: Matches real-world patterns more closely; limited by domain knowledge.
  - **Better Estimator**: More sophisticated optimisation/statistical methods (limited by compute).

### Relative Frequency and Maximum Likelihood

- **Relative Frequency (RF):**
  - **Formula**: $P_{RF}(x) = \frac{C(x)}{N}$
    - **Read:** “The probability of $x$ is the count of $x$ divided by the total number of items.”
  - **Law of Large Numbers:** As $N \to \infty$, _RF_ converges to the true probability.
  - **The Sparsity Problem:** If counts are small (e.g., 3 red M&Ms out of 20), estimates are inaccurate and unstable.
- **Maximum Likelihood Estimation (MLE):**
  - **RF estimation** is equivalent to **MLE**.
  - **Likelihood:** The probability of the observed data given a specific model and parameters, $P(d|\theta)$.
  - **Goal:** MLE chooses parameters $\theta$ that maximise the likelihood of the data observed.
- **MLE Example (Coins):**
  - Data: `HTTTHTHTTT` (3 Heads, 7 Tails).
  - Fair Model ($\theta=0.5$): Likelihood of sequence is $(0.5)^3 \cdot (0.5)^7 = 0.00097$.
  - MLE/Biased Model ($\theta=0.7$): Likelihood is $(0.3)^3 \cdot (0.7)^7 = 0.00222$.
  - Result: The MLE model assigns a higher probability to the observed data than the fair model.

## W202 - N-Grams

### The Sparse Data Problem

- **Objective:** Estimate $\hat{P}(\vec{w})=P(w_{1}...w_{n})$ for a full sentence.
  - **Read:** “The probability of the entire sentence, made up of all its words in order.”
- **Direct MLE Approach:** Trying to count the specific sentence occurrence divided by total sentences $\hat{P}(\vec{w})=\frac{C(\vec{w})}{N}$.
- **Failure:** This fails because most valid sentences never appear in the corpus, resulting in a count of 0 (sparse data). Nearly all $\hat{P}(\vec{w})$ will be 0.

### Unigram Models (Bag-of-Words)

- **Assumption:** Words are generated independently of one another.
- **Formula:** $\hat{P}(\vec{w})=\prod_{i=1}^{n}P(w_{i})$
  - **Read:** _“The probability of the whole sentence equals the product of the individual word probabilities.”_
- **Flaw:** It ignores context and word order. $P(\text{the cat slept}) = P(\text{cat slept the})$.
- **Utility:** Despite poor syntax modelling, useful for "aboutness," i.e - for topic classification and lexical semantics.

### Joint & Conditional Probability (Definitions)

- **Joint Probability:**  
   $P(A,B)$ — the probability that **A and B happen together**.
  - **Example:** $P(\text{the}, \text{cat})$ is the probability of seeing “the cat” as a two-word sequence.
- **Conditional Probability:**  
   $P(A \mid B)$ — the probability that **A happens given that B has already happened**.
  - **Example:** $P(\text{cat} \mid \text{the})$ is the probability that “cat” follows “the”.

### General N-Gram Models

- **Chain Rule:** The _joint probability_ is rewritten as a product of _conditional probabilities_:
  $P(\vec{w}) = P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)...$.
  - **Read:** _“The joint probability of the full sentence equals a product of conditional probabilities, each word conditioned on all earlier words.”_
- **Markov Assumption:** We assume only a finite history matters (limited context).
  - **Reason**: Full conditioning is impossible due to data sparsity.
- **Trigram Model** ($N=3$): Conditions on the previous two words.
  $P(w_i | w_{1}...w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})$.
  - **Read:** _“The probability of the next word depends only on the previous **two** words.”_
- **Bigram Model** ($N=2$): Conditions on the previous single word.
  $P(w_i | w_{i-1})$.
  - **Read:** _“The probability of the next word depends only on the **previous single** word.”_

### Estimating N-Gram Probabilities (MLE)

- **Maximum Likelihood Estimation:** Use relative frequencies from a corpus.
- Bigram Estimate:
  $P_{ML}(w_{2}|w_{1})=\frac{C(w_{1},w_{2})}{C(w_{1})}$.
  (Count of word pair divided by count of the history word).
- Trigram Estimate:
  $P_{ML}(w_{3}|w_{1},w_{2})=\frac{C(w_{1},w_{2},w_{3})}{C(w_{1},w_{2})}$.

### Tiny Corpus Example

- Using this toy corpus of 6 sentences:
  1.  `the cat sat on the mat`
  2.  `the cat slept on the mat`
  3.  `the dog sat on the rug`
  4.  `the dog slept on the rug`
  5.  `the cat sat on the rug`
  6.  `the dog sat`
- Total tokens = **33**.
- Compute the probability of the sentence: **S = “the cat sat on the rug”**

#### 1. Unigram Model

- **Counts:** `C(the) = 11`, `C(cat) = 3`, `C(sat) = 4`, `C(on) = 5`, `C(rug) = 3`, `N = 33`.
- **Unigram probabilities:**
  - $P(the) = \frac{11}{33} = 0.3333$
  - $P(cat) = \frac{3}{33} \approx 0.0909$
  - $P(sat) = \frac{4}{33} \approx 0.1212$
  - $P(on) = \frac{5}{33} \approx 0.1515$
  - $P(rug) = \frac{3}{33} \approx 0.0909$
- **Sentence probability (unigram, independence assumption):**
  $P_{\text{uni}}(\text{the cat sat on the rug})$
  $= P(the) \times P(cat) \times P(sat) \times P(on) \times P(the) \times P(rug)$
  $= 0.3333 \times 0.0909 \times 0.1212 \times 0.1515 \times 0.3333 \times 0.0909$
  $= 1.69 \times 10^{-5}$

#### 2. Bigram Model

- **We use:**
  $P(\text{the cat sat on the rug})$
  $= P(the) \times P(cat \mid the) \times P(sat \mid cat) \times P(on \mid sat) \times P(the \mid on) \times P(rug \mid the)$
- **Bigram counts:** `C(the, cat) = 3`, `C(cat, sat) = 2`, `C(sat, on) = 3`, `C(on, the) = 5`, `C(the, rug) = 3`.
  - `C(cat, sat) = 2`
- **Conditional probabilities:**
  - $P(sat \mid cat) = \frac{2}{3} \approx 0.6667$
  - $P(on \mid sat) = \frac{3}{4} = 0.75$
  - $P(the \mid on) = \frac{5}{5} = 1.0$
  - $P(rug \mid the) = \frac{3}{11} \approx 0.2727$
- **Putting it together:**
  $P_{\text{bi}}(\text{the cat sat on the rug})$
  $= P(the) \times P(cat \mid the) \times P(sat \mid cat) \times P(on \mid sat) \times P(the \mid on) \times P(rug \mid the)$
  $= 0.3333 \times 0.2727 \times 0.6667 \times 0.75 \times 1.0 \times 0.2727$
  $= 0.0124$

So the bigram model gives a **much higher** probability than the unigram model because it captures local word ordering.

#### 3. Trigram Model

- **We use:**
  $P(\text{the cat sat on the rug})$
  $= P(the) \times P(cat \mid the) \times P(sat \mid the, cat) \times P(on \mid cat, sat) \times P(the \mid sat, on) \times P(rug \mid on, the)$
- **Trigram counts:** `C(the, cat, sat) = 2`, `C(cat, sat, on) = 2`, `C(sat, on, the) = 3`, `C(on, the, rug) = 3`.
- **Conditional probabilities:**
  - $P(sat \mid the, cat) = \frac{2}{3} \approx 0.6667$
  - $P(on \mid cat, sat) = \frac{2}{2} = 1.0$
  - $P(the \mid sat, on) = \frac{3}{3} = 1.0$
  - $P(rug \mid on, the) = \frac{3}{5} = 0.6$
- **Putting it together:**
  $P_{\text{tri}}(\text{the cat sat on the rug})$
  $= 0.3333 \times 0.2727 \times 0.6667 \times 1.0 \times 1.0 \times 0.6$
  $= 0.0364$

The trigram model gives a **much higher** probability than the bigram model because it captures more context.

### Takeaways

- As we **add more context** (unigram → bigram → trigram), the model:
  - **Assigns higher probability** to sequences that match patterns seen in the corpus,
  - **Relies on more detailed counts**, which in real life quickly run into sparsity.

### Sentence Boundaries

- **Problem:** Standard N-grams don't inherently model sentence starts or ends (e.g., "feeds cats" is a valid bigram, but "feeds" is bad at the start of a sentence).
- **Solution:** Augment input with start ($<s>$) and end ($</s>$) tokens.
- Model structure: $w_0 = <s>$ and $w_{n+1} = </s>$.
  $P(\vec{w})=\prod_{i=1}^{n+1}P(w_{i}|w_{i-1})$.
  - **Read:** _“The probability of the whole sentence is the product, from i = 1 to n+1, of the probability of each word given the previous word.”_

### Evaluation: Entropy

- Entropy $H(X)$ tells you **how unpredictable a random variable is**.
- If you draw an outcome from some distribution:
  - If all outcomes are equally likely → **very unpredictable** → **high entropy**.
  - If one outcome is very likely → **more predictable** → **low entropy**.
- **Roughly**: How many yes/no questions do I need, on average, to figure out the outcome?
- **Formula:** $H(X)=\sum_{x}-P(x)log_{2}P(x)$, intuition:
  - If an outcome is **very likely** (e.g., $P(x)=0.99$), then $-P(x)\log_2P(x)$ is small → **not much uncertainty**.
  - If probabilities are **spread out** (e.g., 1/6 each for a die), the terms are larger → **more uncertainty**.
  - Entropy therefore increases when outcomes are more even.
- **Interpretation:**
  - **Measured in bits**: Higher entropy means you need more yes/no questions to identify the outcome.
  - **Uniform Distribution:**
    - Highest uncertainty (Entropy = $log_2 N$).
    - All outcomes equally likely: A fair 8-sided die: $H=3$ bits.)
  - **Skewed Distribution:**
    - Lower uncertainty (Entropy < $log_2 N$)
    - One outcome dominates → entropy smaller: (Biased coin above: $H \approx 0.47$ bits.)
- **English Entropy:** Estimated to be roughly 1.3 bits per character because some characters and patterns (e.g. “th”, “e”) are far more frequent.

### Evaluation: Perplexity

- **Cross-Entropy:** Measures how well a model predict the data you actually observed. It is the _average negative log-probability the model assigns to the true sequence_.
  - **Low cross-entropy → model assigns high probability = better predictions**
- **Perplexity Formula:** $PP_{M}(\vec{w})=2^{H_{M}(\vec{w})}$.
  - **Read:** “Perplexity is two raised to the cross-entropy.”
- **Intuition:**
  - Perplexity is the model’s **effective average number of choices** at each step.
  - Perplexity ≈ 2 → model is very confident (only ~2 likely next words).
  - Perplexity ≈ 100 → model is very uncertain (many plausible next words).
  - **Lower perplexity = better model (less confused).**
- **Example:** If a model has cross-entropy $H_M = 3$ bits per word, then $PP = 2^3 = 8$
  - “On average, the model behaves like it has about **8** plausible next-word options.”
- **Model comparison:**
  - Unigram → high perplexity (no context); Bigram → lower; Trigram → lower still
  - More context → fewer effective choices → **lower perplexity**.
- **Why logs appear:** We compute cross-entropy logs ($2^{-\frac{1}{n}\sum \dots}$) to avoid computational underflow (numbers becoming too small for the computer) when multiplying many tiny probabilities.

## W203 - Smoothing and Sampling

### The Problem: Zero Probabilities & Overfitting

- **Unseen N-Grams:** If a sequence (e.g., "shall commence consuming") is valid but never observed in the corpus, MLE assigns it a probability of 0.
  - **Consequence:** The entire sentence receives a probability of 0, regardless of the quality of other parts.
- **MLE Flaws:**
  - Assumes anything unseen is impossible.
  - Overestimates probability for rare events that _do_ occur (overfitting).
- **Solution:** **Smoothing**. This reassigns probability mass from observed (rich) to unobserved (poor) events.

### Add-One Smoothing (Laplace)

- **Concept:** Pretend we have seen every possible N-gram one time more than observed.
- **Normalisation:** To ensure probabilities sum to 1, the denominator must increase by the vocabulary size $v$.
- **Formula:** $P_{+1}(w_{i}|w_{i-1})=\frac{C(w_{i-1},w_{i})+1}{C(w_{i-1})+v}$.
  - **Read:** _“The smoothed probability of word $w_i$​ given $w_{i-1}$​ equals the bigram count plus one, divided by the total count of $w_{i-1}$​ plus the vocabulary size.”\_
- **Major Issue:** Vocabulary size ($v$) is often huge (e.g., Europarl $v \approx 86,000$). Adding $v$ to the denominator **significantly dilutes probabilities of frequent events**.
  - _Example:_ A probability of 1 can drop to ~1/1000;
  - This method "steals" too much mass from frequent events.

### Add-Alpha Smoothing

- **Concept:** Instead of adding 1, add a fraction $\alpha < 1$ to counts to disturb frequent events less.
- **Formula:** $P_{+\alpha}(w_{i}|w_{i-1})=\frac{C(w_{i-1},w_{i})+\alpha}{C(w_{i-1})+\alpha v}$.
- **Optimisation:** $\alpha$ is a **hyperparameter** that must be tuned.
  - Choose a good value of $\alpha$ that minimises perplexity on dev set
  - The model will not fit the training data as well as MLE, but will generalise better.

### Data Splits

To optimise parameters like $\alpha$, data is split into three sets to prevent overfitting.

1. **Training Set (80-90%):** Used to estimate the probabilities (train the model).
2. **Development/Held-out Set (5-10%):** Used to debug and **optimise hyperparameters**
3. **Test Set (5-10%):** Simulates real-world deployment. Used **only once** for final reporting.

### Interpolation & Advanced Smoothing

- **Limit of Add-Alpha:** It assigns equal probability to all unseen events.
  - _Example:_ If "Scottish beer drinkers" and "Scottish beer eaters" are both unseen, Add-Alpha gives them the same score, despite "drinkers" being more plausible.
- **Solution:** Use information from **lower-order N-grams** (shorter histories) via **Interpolation** or **Backoff**.
  - **High-order N-grams:** Provide **rich, specific context** when the sequence has been seen, but their counts are often **sparse** because long n-grams are rarely repeated.
  - **Low-order N-grams:** Provide **broad, general context.** Ignore most of the history so are less specific, but have more **reliable counts** as shorter patterns occur more often.
- **Interpolation**: Mix all N-gram levels
  - Combine probabilities from **all** N-gram orders at the same time using weights that sum to 1.
  - **Idea:** High-order models give specificity; low-order models add robustness.
  - $P_{I}(w_{3}|w_{1},w_{2})=\lambda_{1}P_{1}(w_{3}) + \lambda_{2}P_{2}(w_{3}|w_{2}) + \lambda_{3}P_{3}(w_{3}|w_{1},w_{2})$.
- **Backoff**: Use the most specific model that exists
  - Use the **highest-order** N-gram that has a non-zero count; if unavailable, **back off** to a lower-order model.
  - **Idea:** Prefer specific contexts, fall back only when needed.
- **Kneser-Ney Smoothing:** The standard smoothing method for non-neural N-grams.
  - **Core Idea:** The probability of a word appearing in a new context depends on how many _distinct_ contexts it has previously appeared in.
  - _Example:_ "Francisco" is frequent but only follows "San". It should have a low probability of appearing in new contexts.

### Fundamental Limitations of N-Grams

- **Lack of Similarity:** Words are treated as distinct symbols. The model does not know that "salmon" and "swordfish" are semantically similar.
  - _Solution:_ Neural models use **embeddings** (vector representations).
- **Long-Distance Dependencies:** N-grams have a fixed history length. They _fail to capture dependencies separated by many words_ (e.g., "The **dog** who... **barks**").
  - _Solution:_ Neural models with attention or tree-structured models.

### Generation & Sampling

- **Generative AI:** N-gram models are generative because they define a probability distribution over sequences, letting us **sample** new text
- **Sampling Process:** $x\sim P(X)$.
  - **Read:** “Draw a random outcome $x$ according to the distribution $P(X)$.”
  1. Start with $w_0 = BOS$ (Beginning of Sentence).
  2. Randomly sample next word $w_1$ from the distribution $P(w|BOS)$.
  3. Sample $w_2$ from $P(w|w_1)$, and continue until $EOS$.
- **VS Max Likelihood:** Sampling is different from always picking the highest probability word (which reduces diversity).
- **Advanced Sampling Techniques:** Used to balance quality vs. diversity.
  - **Top-k:** Sample only from the top $k$ most probable tokens.
  - **Top-p (Nucleus Sampling):** Choose the **smallest set of tokens** whose cumulative probability $\ge p$, then sample from that set.
  - **Temperature ($\tau$):** Rescales the probability distribution before sampling.
    - Low $\tau$ $(< 1)$: sharper distribution → more deterministic.
    - High $\tau$ $(> 1)$: flatter distribution → more diverse.

## W301 - Text Classification with Logistic Regression

### Introduction to Text Classification

- **Definition:** Classification (categorisation) is the task of assigning an object or text to a specific group or class.
- **Examples:**
  - **Object/Speech:** Identifying physical objects or recognising spoken words.
  - **Text:** Determining if an email is spam, assessing native language, or identifying a topic.
- **Formal Task:** $\hat{c} = \text{argmax}_{c \in C} P(c|d)$.
  - **Read**: “To get $\hat{c}$ choose the category $c$ in the set $C$ that has the highest probability given the document $d$.”
  - $\text{argmax}$ is a function that returns the argument (here, the class) that makes the expression largest.
- **Intuition**
  - The classifier computes a probability score for each class: $P(\text{spam} \mid d)$ and $P(\text{not spam} \mid d)$ then it **picks the one that is highest**.

### Types of Text Classification Tasks

- **Content Classification:**
  - **Undesirable Content:** Detection of spam, hate speech, or disinformation.
  - **Sentiment Analysis:** Binary (positive/negative) or multi-way (star ratings) for reviews.
  - **Topic Classification:** Categorising texts by subject (e.g., sport, finance).
- **Authorship Attribution:**
  - **Demographics:** Identifying gender, dialect, or native language.
  - **Clinical Diagnosis:** Identifying cognitive impairments or psychiatric conditions.
  - **Authentication:** Distinguishing between human and AI-generated text.

### Bayes’ Rule

- **Formula:** $P(c \mid d) = \frac{P(d \mid c)\,P(c)}{P(d)}$
  - **Read:** “The probability of class ccc given document ddd equals the likelihood of the document under ccc times the prior of ccc, divided by the overall probability of the document.”
- **Quick meanings**
  - $P(c)$ — prior (how common the class is)
  - $P(d \mid c)$ — likelihood (how well the class explains the document)
  - $P(c \mid d)$ — posterior (updated belief after seeing the document)
- **Why we use it**
  - It lets us update our belief about a class based on how well that class could have produced the document.
- **In classification, we use the proportional form:** $P(c \mid d) \propto P(d \mid c)P(c)$ because $P(d)$ is the same for all classes.

### Modeling Approaches: Generative

- When we classify text, we want: $P(c \mid d) \quad \text{“the probability of class \(c\) given document \(d\)”}$
- **Generative Models:** _Model how the text was produced_ (Model Joint Probability)
  - A generative model tries to **explain how the document $d$ could have been generated** if it belonged to class $c$.
  - It models:
    - $P(c)$ — how common the class is (e.g., spam vs not-spam)
    - $P(d \mid c)$ — how likely the document is _if_ it is in class $c$
    - $P(c,d) = P(c)\,P(d \mid c)$ - The joint probability
  - Then it uses Bayes’ rule: $P(c \mid d) \propto P(d \mid c)\, P(c)$
    - **Roughly:** _“Pick the class that makes the document most likely.”_
- Because they model the **joint probability** $P(c,d)$, they can also **generate new documents** by sampling the model. e.g. Naïve Bayes

### Naïve Bayes

- A simple generative classifier for text classification (spam, topic sentiment etc)
  - **Pros:** Fast, simple, works well with sparse data
  - **Cons:** Independence assumption limits performance on tasks with strong word dependencies
- Uses Bayes' rule to choose the class that makes the document most likely.
- **“Naïve” assumption:** All features (words) are conditionally independent given the class.
  - In reality, words are _not_ independent (“New” predicts “York”).
  - The assumption makes the model extremely simple and surprisingly effective.
- **How it works (bag of words):**
  - For each class $c$, estimate $P(w \mid c)$for every word $w$.
  - For a new document $d = [w_1, w_2, \dots]$: $P(d \mid c) = \prod_{i} P(w_i \mid c)$
  - Multiply by the prior $P(c)$
  - Pick the class with the highest score.

### Modelling Approaches: Discriminative

- **Discriminative Models:** Just draw the boundary between classes (_model conditional probability_)
  - Directly model the **conditional** $P(c \mid d)$
  - They never try to model $P(d)$ or $P(d \mid c)$
  - Focus on what differentiates classes rather than modelling the class itself.
  - **Examples:** Logistic Regression (probabilistic), SVMs, Decision Trees.

### Feature Extraction

- **Representation:** A document is represented as a feature vector $x = [x_1...x_n]$.
- **Bag-of-Words:** Features are simple counts of word occurrences.
- **Feature Selection Strategies:**
  - **Frequency:** Use most frequent words; exclude "stop words" (function words like _the, a, in_).
  - **Lexicons:** Use specific lists, such as sentiment dictionaries (e.g., _adorable_ vs. _abysmal_).
  - **Binary:** Track presence/absence rather than counts.
  - **Complexity:** Can use bigrams, syntactic, or morphological features.
- **Domain Importance:** Standard stop words (pronouns) can be predictive in clinical tasks (e.g., "I" for depression, "You" for schizophrenia).
- **Document Embeddings:** Learned feature vectors from neural networks (alternative to hand-crafted features).

### Binary Logistic Regression

- **Goal:** Classify text into two classes (e.g., Positive $1$ vs. Negative $0$).
- **Model Components:**
  - **Input:** Feature vector $\mathbf{x}$.
  - **Parameters $\theta$:** Weight vector $\mathbf{w}$ and bias $b$.
  - **Output:** $\hat{y} = P\bigl(y = 1 \mid \mathbf{x}; \boldsymbol{\theta}\bigr)$ Compute the probability of 1 given input x and parameters $\theta$. Simplified to $P(y \mid \mathbf{x})$
- **The Score (Logit):**
  - Compute a linear score $z$ via dot product: $z = \textbf{w} \cdot \textbf{x} + b$.
  - Positive weights increase the score (push toward positive class); negative weights decrease it.
- **The Logistic Function (Sigmoid):**
  - Normalises score $z$ into a probability between 0 and 1.
  - Formula: $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$.
    - $\sigma(z)$ is the sigmoid function with the logit at the input.
  - **Behavior:** large negative $z \approx 0$; $\sigma(0) = 0.5$; large positive $z \approx 1$;
- **Decision:** Classify as positive if $\hat{y} > 0.5$.
- **Training**: Choose the weights $\mathbf{w}$ and bias $b$ that make the models predictions match the training labels by _minimising the negative log-likelihood_, using **gradient descent**

### Multinomial Logistic Regression _(MLR/Softmax)_

- **Goal:** Classify text into $>2$ classes _(e.g., Topic Classification, Next word prediction)._
- **Structure:**
  - Requires a separate weight vector $\mathbf{w_k}$ and bias $b_k$ for _each_ class.
  - Outputs a vector $\mathbf{\hat{y}}$ of probabilities (one for each class).
- **Softmax Function:**
  - Generalises the logistic sigmoid function for multiple output classes.
  - **Formula**: $\hat{y}_k = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}$ where $z_j = \mathbf{w_j} \cdot \mathbf{x} + b_j$
  - Exaggerates differences between scores and normalises them to sum to 1.
- **Matrix View:** Can be calculated via matrix multiplication $\mathbf{\hat{y}} = \text{softmax}(\mathbf{Wx} + \mathbf{b})$.

### Alternative Views and Evaluation of MLR

- **Neural Network View:** Logistic regression is equivalent to a single-layer neural network.
- **Geometric View:**
  - Weights represent directions in feature space.
    - Moving in the direction of a class's weight vector makes the document more like that class.
  - Decision boundaries are linear hyperplanes (straight lines in 2D).
- **Pros:** Good baseline accuracy, data-efficient, fast to train, interpretable weights.
- **Cons:** **Linear boundaries** may be too simple; manual feature engineering is difficult and time-consuming.

## W302 - Training Logistic Regression and Evaluation

### Training Components

Training a model requires distinguishing between the _goal_ and the _method_:

- **Objective Function (The Goal):** A metric (**loss function**) used to evaluate if a set of weights is good or bad.
- **Learning Algorithm (The method):** The method used to move weights towards values that optimise the objective.
- **Comparison to N-grams:**
  - N-gram training is often thought of as just counting/smoothing (the method).
  - However, N-grams implicitly **maximise likelihood** (the Goal), and smoothing corresponds to Bayesian objectives.
- **Maximum-Likelihood Training**
  - Likelihood is defined $P(data|model)$
  - The model is parameterised by $\theta$
  - So, to maximise the likelihood: $\hat{\boldsymbol{\theta}} \;=\; \arg\max_{\boldsymbol{\theta}} \; P(\text{data} \mid \boldsymbol{\theta})$
    - **Read:** “Pick the parameter vector θ\thetaθ that gives the highest probability to the observed data.”
- In N-gram models, the parameters $\theta$ are just the **conditional probabilities** of each word given its history.
  - The **best** (maximum-likelihood) estimate of these probabilities is simply **how often** the N-gram occurs divided by **how often the history** occurs.
  - So MLE is achieved by **setting conditional probabilities equal to relative frequencies**.

### Discriminative Models: Use Conditional Maximum Likelihood

- **Goal:** Maximise the **Conditional Likelihood** of the true labels given the features - $P(\mathbf{labels} | \mathbf{features}, model)$.
- Unlike Generative models (N-grams), Discriminative models cannot compute $P(data|model)$ directly.
- **Objective Function (Loss):**
  - Instead of maximising the likelihood directly, we minimise the **Negative Log-Likelihood** $\mathcal{L} = -\sum_i \log P(y_i \mid \mathbf{x}_i;\boldsymbol{\theta})$
  - This is also called **Cross-Entropy Loss**!! $\text{Cross-Entropy Loss}=\text{Negative Log-Likelihood}$
  - The loss function $L(f(x; w), y)$
    - The loss $L$ evaluated at the models prediction $f(x;w)$ vs the true label $y$
  - **True Labels Only:** for each example, the loss uses **only** the model's probability for the **gold** class $y_i$. Probabilities for _incorrect_ classes are ignored.
  - **Intuition:**
    - High Probability for the true class -> Small Loss
    - Low Probability -> Large Loss
  - **Why Negative Log:**
    - Logs avoid underflow (turning products into sums)
    - Negating makes it a **minimisation** problem instead of maximisation _(since log-probabilities are negative)_

### Probability vs. Likelihood

- Same formula, different interpretation: $P(\text{data} \mid \theta)$
  - As a **probability** (data varies, parameters fixed):
    - It is “the probability of the data when the parameters are fixed.”
    - You know the machine → “How probable was this output?”
    - Read $P(\text{6 heads, 4 tails} \mid \theta = 0.5)$, what is the _probability_ of 6 heads, 4 tails given the probability of heads is 0.5
  - As a **likelihood** (parameters vary, data fixed)
    - It is “a function of $\theta$” telling us how good those parameters are at explaining the data.
    - You know the output → “Which machine settings most **likely** produced this?”
    - Read $L(\theta)=P(\text{6 heads, 4 tails} \mid \theta)$, which value of $\theta$ makes the observed data most plausible.

### Optimisation: Stochastic Gradient Descent (SGD)

- **Concept:** To minimise loss, we compute the **gradient** (slope) of the loss function with respect to the parameters $\frac{d}{dw} \, L\bigl(f(x; w),\, y\bigr)$
  - We update parameters by taking steps in the direction of the negative gradient.
- **Convexity:**
  - Logistic Regression loss is **convex** → One global minimum
  - So SGD will always find the same optimum (unlike neural networks, which have many local minima)
- **Gradient Calculation:**
  - For softmax or binary logistic regression, the gradient for a single example is **surprisingly simple** $\frac{\partial L_{\text{CE}}}{\partial w_{k,i}} = - (y_k - \hat{y}_k)\, x_i$.
    - **Read:** “Update weight $w_{k,i}$​ in proportion to the prediction error $(y_k - \hat{y}_k)$ and the feature value $x_i$​.”
- **Interpretation:**
  - Larger updates occur when:
    - The model is **very wrong** (big gap between $y_k$​ and $\hat{y}_k$​), or
    - The feature xix_ixi​ has a **large value**.
  - This is why logistic regression behaves like a “weighted correction”: **big mistakes → big updates; small mistakes → small updates.**
- **SGD Algorithm:**
  1. Pick a single training example $(x,y)$ at random ($x$ is the input, $y$ is the label)
  2. Compute prediction $\hat{y}$
  3. Compute gradient ($-(y-\hat{y})x$)
  4. Update parameters (weights/biases).
  5. Repeat until convergence.
- **Learning Rate ($\eta$):** Controls step size.
  - **Too high:** Risk of overshooting the minimum or diverging / unstable
  - **Too low:** Convergence is very slow.
  - Good practice: decay the learning rate over time
- **Mini-Batching:**
  - Instead of updating on one example at a time, use a **small batch**
  - Reduces variance in updates + much faster on modern hardware.

### Regularisation

- **Problem:** Pure Conditional Maximum Likelihood estimation tends to **overfit** training data **(e.g., assigning very high weights to rare features).**
- **Solution:** Add a **penalty term** to the objective function for having large weights.
- **Types:**
  - **L2 Regularisation:** Penalises "Euclidean distance" (sum of squared weights). Keeps weights generally small.
  - **L1 Regularisation:** Penalises "Manhattan distance" (sum of absolute weights). Encourages sparsity (setting weights to exactly 0), less penalty for large weights.
- **Handling Bias:**
  - Do **not** regularise the bias term.
  - Biases are needed to **capture overall distribution of data** (e.g., prior class probabilities and locations of classes, if not centred)
  - Limiting them can artificially shift decision boundaries away from the data.
  - Biases are **_active for every example, so they rarely cause overfitting_**.

### Intrinsic vs Extrinsic Evaluation

- **Intrinsic:** a measure inherent to the task
  - Typically automatic to compute
  - Language Modelling (Perplexity); Classification (accuracy, F-score)
- **Extrinsic:** measures effects on a downstream task
  - More complex (larger system, human evals)
  - Language modelling (Improve MT? ASR?) Classification (i.e Reduce user search time)

#### Metrics

- **Accuracy:** Percentage of correctly classified documents.
  - What % of documents did I classify correctly? 12/15 = 80%
  - _Flaw:_ Misleading if classes are **unbalanced** (e.g., a spam detector that always predicts "not spam" might be 90% accurate but useless).
- **Precision & Recall (Detection):**
  - **Precision:** measures the accuracy of positive predictions $\frac{\text{num items detected and was right}}{\text{num items detected}}$
  - **Recall:** measures the models ability to find all positive cases $\frac{\text{num items detected and was right}}{\text{num items should have been detected}}$
- **Thresholds:**
  - Classification usually involves a probability threshold (e.g., > 0.5).
  - **Raising threshold:** Increases Precision, decreases Recall (more conservative).
  - **Lowering threshold:** Decreases Precision, increases Recall (less conservative).
- **F-Measure ($F_1$):**
  - The harmonic mean of Precision and Recall.
  - Used to combine both metrics into a single score; **_penalises cases where one is high and the other is low._**

## W303 - Lexical Semantics

### Introduction and Orientation

- **Shift in Focus:** Previous lectures treated tokens as symbols to analyse structure and statistics.
- **New Goal:** We are moving toward understanding **meaning** (semantics), specifically **lexical semantics** (meanings of individual words).
- **Future Topics:** This foundation leads to word embeddings (non-symbolic representations) and sentential semantics.

### The Challenge of Meaning in AI

- **The Grand Goal:** Creating machines that understand data rather than mindlessly processing it.
- **Eliza (1969):**
  - An early chatbot simulating a psychotherapist.
  - Fooled users into inferring understanding by asking non-specific follow-up questions.
- **Pragmatic View:** We follow the Turing test spirit: Can the computer _behave_ as though it understands? (e.g., in Dialogue, Translation, or QA systems).

### Case Study: Question Answering (QA) Difficulties

- **Simple Matching Easy:** Questions like "When was Barack Obama born?" are easy if text available matches the pattern. Simply rephrase: "...was born on\_\_\_"
- Building a QA system reveals specific semantic challenges:
  - **Word Senses:** "Plants native to Scotland" vs. "Chemical plant." The system must disambiguate meanings.
  - **Synonyms:** "Vacation" vs. "Holiday." Different words can mean the same thing.
  - **Hyponyms/Hypernyms:** "Animals" vs. "Polar bears." One word refers to a subclass (hyponym) or superclass (hypernym) of another.
  - **Similarity:** "Good way to remove" vs. "Great way to eliminate." Words relate via similarity or gradation.
  - **Inference Challenge:** Requires combining multiple facts (e.g., knowing Poland is in Central Europe) to answer a question.

### Ontologies and WordNet

- **WordNet:** A hand-built English ontology containing ~117,000 **synsets**.
  - **Synset:** A set of synonymous words representing a single concept.
- **Defined Relations:**
  - **Hyponym/Hypernym:** IS-A relationship (e.g., chair is furniture).
  - **Meronym:** PART-WHOLE relationship (e.g., leg is part of a chair).
  - **Antonym:** Opposites (e.g., good vs. bad).

### Word Sense Ambiguity

- **Homonyms:** Totally different words spelled the same way.
  - _Example:_ "Bank" (financial) vs. "Bank" (river).
- **Polysemes:** Words with multiple _related_ meanings.
  - _Example:_ "Chicken" (the animal) vs. "Chicken" (the meat).
  - These often follow predictable patterns (e.g., Container for Contents: "drank the bottle").
- **The Boundary Problem:** Distinguishing polysemy from homonymy is difficult. The word "interest" has many senses (financial, curiosity, legal share) that are hard to count definitively.

### Testing for Distinct Senses

- **Conjunction Test (Zeugma):** Conjoining two uses of a word with "and." If it sounds wrong, the senses are likely distinct.
  - _Example:_ "?Does Midwest Express serve breakfast and Philadelphia?" (Invalid combination of senses).
- **Translation:** If a word translates to different words in another language, this indicates multiple senses.
  - _Example:_ English "river" becomes French "fleuve" (flows to sea) or "rivière" (smaller).
  - _Example:_ English "interest" splits into German "Zins" (financial) and "Anteil" (stake).

### Word Sense Disambiguation (WSD)

- **The Task:** Given a polysemous word, identify its specific sense in a specific context.
- **Relevance:** Important for search relevance (avoiding "chemical plant" when searching for biology) and humanities research.
- **Method:** Typically formulated as a classification task.

### Distributional Semantics

- **Limitations of Ontologies:** They are resource-intensive to build, often miss words, and lack fine-grained similarity measures.
  - **Ontology**: **hand-crafted, symbolic knowledge structure** that defines the _entities, concepts, categories,_ and _relations_ in a domain.
- **The Distributional Hypothesis:**
  - **_Meaning can be inferred from the contexts_** a word occurs in.
  - Words occurring in similar contexts have similar meanings.
- **Linguistic Distribution:**
  - The set of contexts in which a word appears.
- **Vector Space Models:**
  - Represent words as vectors of context co-occurrences.
  - **Example:** A matrix where rows are words ($w_i$) and columns are context features. If $w_i$ co-occurs with "bone," the value is 1; otherwise 0.
  - Real vectors are much sparser than toy examples.
- **Next Steps:** Moving from raw count-based vectors to prediction-based vectors (embeddings).

## W401 - Dense Word Embeddings

### Introduction

- **Distributional Semantics:** The goal is to learn word meanings automatically from text.
  - **Core Idea:** A word is represented as a vector of its contexts.
  - "Distributional Semantic Models" $=$ "vector-space models".

### Sparse vs. Dense Vectors:

- **Sparse (Count-based):** High dimensionality (e.g., 10k–50k vocabulary words), where most values are zero.
- **Dense (Embeddings):** Lower dimensionality (100s–1000s), where most values are non-zero.
- **Benefits of Dense:** Fewer parameters, less prone to overfitting, and easier to use as features in machine learning.

### Measuring Similarity in Vector Space

- **Motivation:**
  - **Information Retrieval:** Matching queries (e.g., "remove") to answers using synonyms (e.g., "eliminate").
  - **Language Modelling:** Generalising probabilities to unseen words (e.g., from "salmon" to "swordfish") based on similar contexts.
- **Options Explored**
  - **Euclidean Distance:** Straight-line distance between two points.
    - $Dist_{EUC}=\sqrt{\Sigma_{i=1}^{D}(\upsilon_{i}-w_{i})^{2}}$.
    - **Issue:** Intuitions fail in high-dimensional space; most points become roughly equidistant.
  - **Dot Product:**
    - Formula: $sim_{DP}(v,w)=v\cdot w = \sum_{i=1}^{D}v_{i}w_{i}$.
    - **Issue:**
      - Sensitive to vector length - gives a large value if both are large.
      - Frequent words have larger norms, resulting in high dot products regardless of actual similarity.
- **Cosine Similarity (Standard Metric):**
  - Measures the **angle** between vectors: 1 (same direction), 0 (orthogonal), -1 (opposite).
  - Angles measured with respect to the origin, so ideally vectors centred around origin.
  - Solution to dot product bias is normalisation by vector length.
    - Formula: $sim_{cos}(v,w)=\frac{v\cdot w}{||v||\cdot||w||}$

### Learning Dense Embeddings: Word2Vec

- **Concept:** Instead of counting co-occurrences, **train a classifier to predict which words co-occur**. The learned parameters become the embeddings.
- **Architectures:**
  - **CBOW (Continuous Bag of Words):** Predict central word from context.
  - **Skip-gram:** Predict context words from a central word (one at a time)

### Skip-gram

- **Task:** Given a **target word** $w_t$, predict its **context words** within a window of size _2m_
- **Two vectors per word:**
  - **Target vector:** $\mathbf{v}(w_i)$ — used when the word is the **centre** word.
  - **Context vector:** $\mathbf{c}(w_i)$ — used when the word appears in the **context** of another word.
  - Can use a Softmax Regression: $$P(w_{k}|w_{t})=\frac{1}{Z}exp(v(w_{t})\cdot c(w_{k}))$$
  - **Read:** “The probability of context word $w_k$​ given target word $w_t$​ is the softmax of the dot-product between the target vector and the context vector.”
  - **Where:**
    - $\mathbf{v}(w_t)$ = **target word vector** (features for regression)
    - $\mathbf{c}(w_k)$ = **context word vector** (weights for regression)
    - $Z$ = normalisation constant ensuring probabilities sum to 1
  - **Intuition:** Vectors for words that co-occur are "pushed" together to maximise dot product.
    - **co-occur** (1st order): $cute$ coocurs with $cat$, then $\text{v(cat)}$ should be similar to $\text{c(cute)}$
    - **share contexts** (2nd order) $cute$ coocurs with $puppy$ then $\text{v(puppy)}$ should be similar to $\text{c(cute)}$ and $\text{v(cat)}$

### Skip-gram with Negative Sampling (SGNS)

- **The Bottleneck:** The normalisation constant $Z$ in Softmax requires summing over the _entire_ vocabulary, which is computationally expensive.
- **The Solution (SGNS):** Switch from multi-class softmax classification to **binary classification** using sigmoid
  - **Task**: Given word $w_k$ and target $w_t$, is $w_k$ a real context word?
  - **Equation:** $P(+|w_{t},w_{k})=\sigma(v(w_{t})\cdot c(w_{k}))$ (using Sigmoid).
    - **Read:** “How likely is $w_k$​ to be a _real_ context word for $w_t$​?”
- **Training Data:**
  - **Positive Examples:** True context words from the corpus.
  - **Negative Examples:** Randomly sampled "noise" words (k = 2–5 per positive example).
  - **Sampling Distribution:** Rare words are oversampled (flattened distribution) using $P(w)^{0.75}$.
- **Objective Function:** Maximise probability of real context words, minimise probability of noise words.
- **Resulting Embeddings:** The target vector is kept as the final embedding; context vectors are discarded or concatenated as they are likely similar.

### Evaluating Embeddings

- **Extrinsic Evaluation:** Test embeddings by using them in a downstream task (e.g., classification).
- **Intrinsic Evaluation:**
  - **Similarity Datasets:** Compare model similarity (Cosine) to human similarity ratings (e.g., WordSim-353, SimLex-999).
    - **Complication**: Humans struggle to distinguish **similarity** (car-bicycle) from **relatedness** (car-driver).
  - **Analogies:**
    - Vector arithmetic captures semantic/morphological relations
    - Example: $v(king) - v(man) + v(woman) \approx v(queen)$.
- **Limitation:** SGNS produces **static** embeddings (one vector per word type), ignoring polysemy (e.g., "table" as furniture vs. data).

## W402 - Multilayer Perceptrons (MLPs)

### Introduction

- **Context:** The lecture moves from Multinomial Logistic Regression (MLR) to more powerful Multilayer Perceptrons (also known as Feedforward Neural Networks).
- **Classification Pipeline:**
  - Input text is processed into features.
  - Features are converted into a vector representation.
  - The classifier (MLR) outputs a probability distribution over classes (e.g., positive vs. negative sentiment).
- **MLR Mechanism:** Multiplies feature vectors by class weights, then applies a softmax function to get probabilities.
  - **Limitation 1**: Requires domain knowledge and trail & error to choose good features
  - **Limitation 2**: can only learn linear decision boundaries

### Addressing Limitation 1: Hand-designed features

- **The Limitation:** MLR relies on hand-crafted features requiring domain knowledge.
- **The Solution:** Replace manual features with word embeddings.
- **Embedding Lookup:**
  - A lookup table (dictionary) converts words to embeddings: mathematically represented as an **embedding matrix**.
    - Multiplying a one-hot vector (index of the word) by the embedding matrix retrieves the specific embedding row.
    - **Adv:** Matrix multiplication allows efficient looking up of all words in a document simultaneously (computationally efficient on GPUs)
- **From Matrix to Vector (Pooling):**
  - The input text results in an $N$ by $D$ matrix (N-words, D-dimensions).
  - **Problem:** Classifiers require a single input vector, not a matrix.
  - **Solution:** **Mean Pooling:**
    - A common method for classification is averaging all word embeddings together to create one vector.
    - _Issue:_ Pooling loses positional information (creates a "bag of words" model).

### Addressing Limitation 2: Linear Decision Boundaries

- **The Limitation:** MLR can only learn linear decision boundaries (straight lines) to separate classes.
- **The Solution (Neural Networks):**
  - Neural networks use **hidden layers** to learn useful features.
  - **Warping Space:** The network **warps** inputs to a "hidden space" where classes that were not linearly separable **become linearly separable**.
  - **Interaction Effects:** This allows the model to capture interactions between features (e.g., Feature 1 and Feature 2 together have an effect different from their sum).
- Non-linearities make MLP more flexible than MLR, **but:**
  - More flexible models **overfit more easily**
  - **More data is required** to generalise (not just memorise)
  - Require **'over parameterised'** models (more params than datapoints)
  - **Trickier to train** as non-convex loss functions

### Feedforward Neural Network Architecture

- **Single vs. Multi-layer:**
  - MLR is effectively a single-layer network.
  - A Feedforward Network (or MLP) adds **hidden layers** between the input and output.
- **The Hidden Unit:**
  - Each unit computes a dot product of weights and inputs, adds a bias, and applies an **activation function**.
  - **Activation Function**: $a = \sigma(w \cdot x + b)$.
    - Where $\sigma$ is an 'appropriate non-linear function'
- **Importance of Nonlinearity:**
  - The activation function ($\sigma$) must be **nonlinear** in order to **warp the input space.**
  - Without nonlinearity, a multi-layer network is mathematically equivalent to a single linear layer (collapses back to MLR).
  - With nonlinearity, a network with a single (arbitrarily wide) hidden layer is a **universal approximator** (can compute any function).
- **Deep vs. Wide:** Empirically, deeper networks (more layers) work better than very wide single-layer networks.
- **Types of Activation Functions:**
  - **Perceptron Unit:** Step-function, non-differentiable (so doesn't support gradient based learning)
  - **Sigmoid/Logistic:** Historically used, outputs 0 to 1 along a sigmoid
  - **Tanh (Hyperbolic Tangent):** Similar to sigmoid but ranges -1 to 1.
    - In Sigmoid and Tanh, if $|z|$ (absolute value of z) gets too large, gradient approaches 0, so hard to change weights.
  - **ReLU (Rectified Linear Unit):** Returns $max(0, z)$. Helps avoid "vanishing gradients" and saturation issues.

### The Full Classification Model

- **Architecture:**
  1. **Input:** One-hot vectors for words. $N \times |V|$
  2. **Embedding Layer:** Look up vectors in the embedding matrix. $E = |V| \times d$ -> $N \times d$
  3. **Pooling:** Average embeddings to get a single vector. $X = [1 \times d]$
  4. **Hidden Layer(s):** Apply weights $W [d\times d_h]$ and nonlinear activations. $h = [1 \times d_h$]
  5. **Output Layer:** Compute logits $U = [d_h \times d_o]$ and apply Softmax for probabilities. $d_o$
- **Training:** The entire pipeline (including embedding weights) acts as one network and is optimised together.
- **Forward Pass (Inference):** The process of calculating outputs from inputs by sequentially multiplying through layers.

### MLPs for Language Modelling

- **Goal:** Predict the probability of the _next_ word given the previous words.
- **Architecture Adjustments:**
  - **Context:** Uses the previous N tokens.
  - **Output:** The size of the output layer equals the vocabulary size $|V|$
  - **No Pooling:** Language modelling relies on word order, so mean pooling (averaging) is inappropriate.
    - **Concatenation:** Instead of pooling, input embeddings are **concatenated** into a long vector to preserve order.
- **Comparison to N-grams:**
  - Like N-grams, simple MLPs use a fixed history size.
  - Unlike N-grams, MLPs use word similarity (embeddings), leading to lower perplexity.
  - **Cost:** MLPs are computationally more expensive to train and run than N-grams.

### Generalisation and Hyperparameters

- **Overparameterisation:** Modern networks often have more parameters than data points but still generalise well (don't just memorise), challenging conventional wisdom.
- **Hyperparameters to Tune:**
  - Width of hidden layers (inc. embedding layer)
  - Number of layers (depth).
  - Pooling function type.
  - Nonlinear activation function choice.
- **Loss Function:** Unlike MLR (convex), Neural Networks have **non-convex** loss functions, making training more difficult.

## W403 - Training Neural Nets

### Introduction & Recap

- **Context:** We extend feed-forward networks from **forward pass (prediction)** to the **backward pass (training)**.
- **Forward Pass Steps:**
  - **Layer 1 logits:** $z^{[1]} = W^{[1]} a^{[0]} + b^{[1]}$
  - **Layer 1 activations:** $a^{[1]} = g^{[1]}(z^{[1]})$
  - **Layer 2 logits:** $z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}$
  - **Output probabilities:** $\hat{y} = \text{softmax}(z^{[2]})$
    **Read:** “Multiply activations by weights, add bias, apply activation function.”
- **Goal:**  
   Learn weights that make $\hat{y}$​ close to the true label $y$.

### Training Overview

- **Components:**
  - **Objective** (the goal)
  - **Algorithm** (how to achieve it).
  - Examples:
    - Model | Objective | Training Method
    - N-Gram Model | Maximise Likelihood | Count and Normalise
    - N-Gram Model | Bayesian Objectives | Add-$\alpha$ / Kneser-Ney Smoothing
    - Logistic Regression | Cross-entropy | Stochastic Gradient Descent (SDG)
    - Neural Network | Cross-Entropy | SDG + Back-propagation

### Gradient Descent Concept

- **Loss function:** $L(\theta, D)$ Measures how wrong the model is on dataset $D$.
- **Compute gradients:** $\frac{\partial L}{\partial \theta}$ “How much does changing a weight $\theta$ change the loss?”
- **Update Rule (Gradient Descent):** $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$ _“Move weights a small step opposite the gradient to reduce loss.”_
- **Learning Rate $\eta$:**
  - Controls step size.
  - Too high → unstable, overshooting.
  - Too low → slow convergence.
  - Modern optimisers (Adam, RMSProp) **adapt** the effective learning rate during training.

### The Gradient Problem

- **Complexity:** Unlike logistic regression, where the relationship between weights and error is direct, NNs are complex composite functions.
- **The Challenge:** We must determine how a small change in a weight deep in the network (e.g., an embedding or early layer) affects the final output probabilities.
- **Inefficiency:** Computing derivatives for every parameter individually without a strategy would result in massive redundant computation.

### Why Backpropagation?

- Neural networks contain **many parameters**.
- We need an efficient way to compute $\frac{\partial L}{\partial W^{[i]}}$​ and $\frac{\partial L}{\partial b^{[i]}}$ for **every layer**.
- Backprop uses the **chain rule** from calculus: $\frac{df}{dx} = \frac{du}{dv} \cdot \frac{dv}{dx}$ to propagate gradients **from output → backwards** through the network.
  - **Intuition:** Backprop computes “how much each weight contributed to the final error.”

### Backpropagation

- **The Algorithm:**
  1. **Compute Final Layer:** Calculate partial derivatives for the output (softmax) layer first.
  2. **Propagate Backward:** Use those results to compute gradients for the preceding layer.
  3. **Iterate:** Continue backward through the network, reusing stored computations to build gradients efficiently.

#### Computation Graphs

- **Definition:** A graph used to order and store intermediate computations for the forward and backward passes.
- **Forward Pass:** Inputs flow through operations (nodes) to compute loss; results are stored.
- **Backward Pass:**
  - Starts at the loss node and moves backward.
  - Multiplies the local gradient (partial derivative of the current operation) by the upstream gradient (flowing from the end).
  - _Example:_ If $L = e^2$ and $e = y - d$:
    - $\frac{\partial L}{\partial e} = 2e$.
    - $\frac{\partial L}{\partial d} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial d}$.

#### Memory vs. Compute Trade-off

- **Efficiency:** Backprop drastically reduces _computation_ time by avoiding recalculations.
- **Memory Cost:** It increases _memory_ usage because the entire graph (gradients and intermediate forward-pass values) must be stored until the update is complete.

### Optimisation and Batching

- **The Loop:** Sample a batch $\rightarrow$ Forward pass $\rightarrow$ Calculate loss $\rightarrow$ Backpropagate $\rightarrow$ Update parameters.
- **Batching:** Using "mini-batches" (subsets of data) rather than single examples:
  - Increases efficiency (optimised for matrix multiplication hardware).
  - Smoothes out random noise/variance in gradient updates.
- **Batch Size:** Too large causes memory overflows or delays updates; too small results in noisy training.

### The Training Pipeline

Training involves more than just optimisation. The full process includes:

1. **Hyper-parameters:** Choosing architecture, learning rate, and optimiser.
2. **Initialisation:**
   - Crucial for non-convex loss functions to avoid bad local minima.
   - Weights are typically initialised randomly (near zero, small variance).
   - _Pre-training:_ Can initialise specific layers (like embeddings) using pre-trained models (e.g., Word2Vec) and potentially freeze them initially _(to train only upper layers)_
3. **Regularisation:**
   - Required to prevent **overfitting** (good training performance, bad generalisation).
   - Common method: **L2 Regularisation** (Weight Decay), which penalises large weights.
4. **Troubleshooting:**
   - **Underfitting:** Poor performance on training _and_ test data. Check:
     - baselines, learning rates, data size, stopping condition

### Stopping Criteria

- **Epochs:** One epoch equals one full pass through the training data.
- **Convergence:** Stop when parameter updates become negligible (below a threshold).
- **Early Stopping:** Stop when the loss on the **development set** (not training set) stops decreasing.

### Interpreting Loss Curves (Examples)

- **General Behaviour:** Training loss usually drops quickly and levels off. Dev loss drops then may rise (overfitting).
- **Smoothness:** Curves are jagged/bumpy because mini-batch SGD is stochastic (noisy).
- **Typical Curve:** _Train loss drops quickly, then levels. Dev loss drops more slowly, then rises._
  - **Early Stopping:** Do not stop at the immediate first rise in dev loss; use a moving average or patience window (e.g., wait 3 epochs) to confirm the trend.
- **Dev Loss < Train Loss:**
  - It is possible for dev loss to be lower than training loss if the dev set contains "easier" examples or is small.
- **Train Loss Increases:**
  - This indicates a critical failure.
  - **Causes:** Learning rate too high, code bugs, or plotting unregularised loss vs. regularised optimisation.
- **Ambiguous Performance:**
  - If curves look "normal" but performance is low, compare against a simple baseline (like logistic regression) to detect underfitting.
- **Model Selection:**
  - When selecting between two models, choose the one with the lowest **Dev** loss. The gap between training and dev loss is irrelevant for selection.

## W501 - RNNs

### Recap and Motivation

- **Language Model Goal:** Provide probability estimates for a sequence $y=y_{1}...y_{n}$.
- **Previous Approach (N-gram/MLP):**
  - Uses a fixed window of history (N words).
  - Makes strict independence assumptions: $P(y_{i}|y_{i-N},...y_{i-1})$.
- **Limitations:**
  - Fixed windows struggle with long-distance dependencies.
  - **Examples:** Subject-verb agreement ("The roses... are"), or semantic context ("Moby Dick" $\rightarrow$ "whale" vs. "house").
- **RNN Solution:** Removes explicit independence assumptions to model arbitrary-length contexts.

### RNN Architecture

- **Core Concept:**
  - A Recurrent Neural Network processes sequences by carrying information forward through time.
  - It does this using a **hidden state** that acts as a memory of everything seen so far.
  - The network contains a "cell" with a recursive loop.
- 1. **Hidden State ($h_t$):**
  - $h_t$ represents the context of the sequence up to time step $t$.
  - It is updated using: the **current input** $x_t$ and the **previous hidden state** $h_{t-1}$
  - **Simple Recurrent Network (SRN) Equation:** $h_{t}=\sigma(Uh_{t-1}+We_{t}+b_{1})$.
    - **Read:** _“Combine the previous hidden state $h_t$ and the current word embedding $e_t$ with their weight matrices $W$ and $U$, apply a nonlinearity $\sigma$”_
- 2. **Output Computation:**
  - The hidden state $h_t$ is passed to a softmax layer to predict the next token.
    - $\hat{y}=\text{softmax}(Vh_{t}+b_{2})$.
      - _“Turn the current hidden state into a probability distribution over the vocabulary.”_
- 3. **Unrolling:**
  - Although an RNN contains a **single recurrent cell**, we imagine it being **unrolled** to show how it processes sequences step-by-step
  - Unrolling turns the recurrence into a chain: $h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \dots \rightarrow h_T$
  - This looks like a deep feed-forward network with **one layer per time step**.
- 4. **Intuition Summary**
  - RNNs “remember” previous inputs using their hidden state.
  - The hidden state flows forward through time like a memory buffer.
  - The same computation is applied repeatedly for each position in the sequence.
  - Outputs at each step depend on both the **current input** and the **past context**.

### Probability Estimation

- Feed words one at a time ($x_1, x_2...x_T$).
- At each timestep, the model produces an output distribution $\hat{y_t}$ over the _next_ possible token $\mathbf{\hat{y_t}} = P(Y_t|x_{<t})$
- The probability of the full sequence is the **product** of these step-wise probabilities.
  - $P(x_1...x_T) = \prod_{t=1}^{T} P(Y_t = x_{t+1} | x_{<t})$

### Generation

- Sample an output $y_t$ from the distribution $\mathbf{\hat{y}_{t}}$.
- Feed the generated token $y_t$ back into the network as the input $x_{t+1}$ for the next step.
- We sample from the full distribution using top-k, top-p or temperature.

### Training

- **Method:** Back-propagation through time (BPTT) on the unrolled network.
  - Each timestep becomes a hidden layer: $time = depth$
- **Loss Calculation:** Cross-entropy loss is computed at each step by comparing the predicted distribution $\hat{y}_t$ with the true next token $x_{t+1}$.
- **Teacher Forcing:** During training, use the _actual_ ground-truth words as inputs for the next step, not the model's own predictions!
- **Batching:** Sequences of different lengths are padded (e.g., with zeros) to allow matrix operations.

### The Vanishing Gradient Problem

- **The Issue:** To learn long dependencies, gradients must propagate back through many time steps.
- **The Math:** Gradients of activation functions (sigmoid, tanh) are typically $< 1$.
  - Multiplying many small numbers causes the gradient to shrink toward zero.
  - Result: Weights for early inputs receive very little update signal.
- **Activation Functions:**
  - **Sigmoid:** Derivative is often much less than 1; causes rapid vanishing.
  - **Tanh:** Often preferred as its derivative is closer to 1 in the linear range.
- **Solutions:**
  - **Skip connections:** Linear activation = information to pass through unchanged.
  - **Complex RNN Cells:** Gated Recurrent Units (GRUs) or Long Short-Term Memory (LSTM)
    - Specialised cells that explicitly learn what to remember or forget, routing information to avoid vanishing gradients.

### Interpretability

- Analysis of individual neurons in character-level RNNs reveals they can learn specific structures.
- **Observed Neurons:**
  - **Quote Detection:** Activates inside quotation marks.
  - **Code Structure:** Activates inside `if` statements.
  - **Sentiment:** Tracks positive vs. negative sentiment (verified by intervention).

### Sequence-to-Sequence (Seq2Seq) & Translation

- **Challenge:** Translation is not one-to-one; word order and sentence lengths differ between languages.
- **Encoder-Decoder Framework:**
  - **Encoder RNN:** Reads source text and encodes it into a fixed-dimension context vector (the final hidden state).
  - **Decoder RNN:** Takes the context vector and generates the target translation.
- **Training Data Format:**
  - `Source Sentence <sep> Target Sentence`.
  - The `<sep>` token signals the model to switch from encoding to decoding.
- **Inference:**
  - Encode source $\rightarrow$ Get context vector.
  - Use context vector to initialise the decoder.
  - Generate tokens until the end-of-sequence symbol.

## W502 - Dialect and Discrimination

### Subjectivity in Science

- Science and engineering are often viewed as objective (e.g., measuring parsing speed or accuracy).
- **Subjectivity exists in research design**: researchers must decide **which questions are interesting** and **which metrics (speed vs. accuracy) matter** most.
- **Ethical Trade-offs:** Decisions involve weighing benefits against risks (e.g., improving user experience vs. violating privacy).
- There is rarely a single "right" answer; decisions are shaped by culture, experience, and background.

### Ethics and Historical Context

- **Lack of History:** Unlike medicine or psychology, Computer Science lacks a long-standing tradition of ethical training.
- **The Belmont Report (1978):** Established ethical principles for human subjects in response to exploitative studies (e.g., Tuskegee).
  - **Respect for persons:** Requires informed consent.
  - **Beneficence:** Maximise benefits while minimising risks.
  - **Justice:** Ensure fair and non-exploitative treatment of participants.
- **Modern CS Risks:** Recent issues include privacy leaks, toxic chatbots, disinformation via deepfakes, and AI encouraging self-harm.

### Algorithmic Bias

- **Definition:** Occurs when an algorithm's outputs differ systematically and unfairly between different groups of people.
- **Case Study: Face Recognition:**
  - US-developed systems performed worse on Black faces compared to White faces and had higher false-positive match rates for Black faces.
  - **Causes:**
    - **Training Data:** Lack of diversity in training sets or skewed data sourcing (e.g., mugshots vs. university graduates).
    - **Feature Selection:** Features designed for one group may not discriminate well for others.
  - **Amplification:** AI systems often amplify human biases rather than just mirroring them.

### Legal Implications (UK Context)

- **The Equality Act (2010):**
  - Prohibits discrimination based on 9 **protected characteristics**: age, disability, gender reassignment, marriage/civil partnership, pregnancy/maternity, race, religion/belief, sex, and sexual orientation.
  - Applies to "services," meaning many AI products are legally covered.
- **Types of Discrimination:**
  - **Direct Discrimination:** Treating someone differently specifically because they belong to a protected group (e.g., refusing to hire transgender workers).
  - **Indirect Discrimination:** Applying a policy equally to everyone that inadvertently disadvantages a protected group (e.g., mandatory attendance disadvantaging those with disabilities).
  - **Legality Exception:** Indirect discrimination may be legal if it is a "proportionate means to achieving a legitimate aim" (e.g., safety requirements).

### Dialects and Language Variation

- **Definition of Dialect:** A variety of language varying by region, socioeconomic class, or culture.
  - Includes differences in accent (pronunciation), vocabulary, morphology, and syntax.
  - Dialects are typically mutually intelligible, though degrees vary.
- **Linguistic Perspective:**
  - Everyone speaks a dialect; there is no "neutral" version of a language.
  - Social prestige is attached to "standard" dialects (e.g., General American, Received Pronunciation), but no dialect is linguistically superior to another.
  - The distinction between "language" and "dialect" is often political rather than linguistic.

### Specific Language Examples

- **Arabic:**
  - **Modern Standard Arabic (MSA):** Used in writing/formal settings; taught in schools but not spoken as a native dialect.
  - **Spoken Dialects:** Regional varieties (e.g., Egyptian, Levantine) used in daily life; some are not mutually intelligible and may be considered distinct languages by linguists.
- **Chinese:**
  - **Mandarin:** The standard variety in mainland China; spoken natively by many and taught in schools.
  - **Other varieties:** Roughly 7 other groups (e.g., Cantonese) often called "dialects" culturally, but linguists view them as distinct languages or language families due to lack of mutual intelligibility.
- **English:**
  - Varieties have varying degrees of prestige depending on the listener's context (e.g., an Indian English accent might signal elite status in Kenya but be perceived differently in the US).

### Sociolinguistics and Social Media

- **Sociolinguistics:** The study of language as a social device used to signal identity, achieve goals, or exert power.
- **Social Media Impact:**
  - Represents a new form of communication: written but informal, immediate, and persistent.
  - Forces the "writing down" of dialects that were previously only spoken (e.g., Egyptian Arabic, African-American Vernacular English).
- **NLP Consequence:**
  - NLP tools trained on standard edited text (e.g., news) struggle with social media text and non-standard dialects.
  - **Discrimination:** Marginalised groups often speak less standard dialects; therefore, NLP tools perform worse for them, leading to disadvantage.

### Q&A Key Points

- **Evaluation:** Using separate evaluation metrics (e.g., F1 scores) for different dialects is ideal in principle but difficult in practice due to the challenges of Dialect ID and data collection.
- **Word Embeddings:** Research into dialect-specific word embeddings exists but is currently an understudied area due to data scarcity.

## W503 - Discrimination and Data Ethics

### Context and Background

- **Subject:** The study examines Language Identification (LID) tools and African American Vernacular English (AAVE).
  - AAVE developed originally in the Southern United States among enslaved people and is now spoken across North America.
  - It is not spoken by all African Americans, and usage varies by region and distance from Standard American English.
  - The dialect features characteristic sound patterns (phonology), often appearing as non-standard spelling on social media, as well as distinct vocabulary and syntax.
- **Research Question:** Do off-the-shelf LID tools disadvantage AAVE speakers by failing to identify their tweets as English?
  - _Consequence:_ If tweets are not identified as English, these populations may be filtered out of downstream processing like sentiment analysis.

### Methodology

- **Data Collection:** Researchers used geotagged tweets and US Census data to identify the racial makeup of neighbourhoods.
- **Modelling:**
  - They built language models aligned with African American and White populations based on the assumption that language usage correlates with neighbourhood demographics.
  - Tweets were categorised as "AA-aligned" or "White-aligned" based on which language model generated the majority of their words.

### Analysis and Results

- **Initial Finding:** There was a difference in accuracy, with the system being less accurate on AA-aligned tweets than White-aligned tweets.
- **Interpreting Causation:**
  - A potential confound was identified: AA-aligned tweets in the sample were shorter on average, and LID systems generally perform worse on short text.
- **Controlled Comparison:**
  - Researchers compared tweets of similar lengths to isolate the dialect factor.
  - _Result:_ White-aligned tweets still had higher accuracy across almost all length comparisons.
  - The disparity was particularly large for the shortest tweets.
- **Significance:**
  - The results were statistically significant (not due to random chance).
  - The results are practically significant because a large proportion of the tweets fell into the short category where the difference was largest.

### Modern Implications (LLMs)

- Recent studies indicate that Large Language Models (LLMs) still exhibit "covert" discrimination.
- While trained to avoid overt racism, LLMs may associate AAVE text with negative stereotypes or negative judgments regarding employability.

### Intellectual Property (IP)

- **Annotated Data:** Often requires licenses as it is expensive to produce.
  - _Paid Licenses:_ The Linguistic Data Consortium (LDC) distributes corpora like the Penn Treebank; these cannot be redistributed outside the university.
  - _Free Corpora:_ The Child Language Data Exchange (CHILDES) is free but requires citing the database and contributors.
- **Action:** Always check license agreements before using or redistributing data.

### Privacy and Consent

- **Key Questions:** When using spontaneous data (e.g., social media), researchers must consider if individuals are identifiable and what consent was obtained.
- **Legal/Ethical Bodies:**
  - **Laws:** The UK Data Protection Act 2018 governs personal data.
  - **Oversight:** University ethics panels and scientific bodies (like the ACL) impose guidelines.
- **Vulnerable Groups:** Research involving children or people with disabilities requires heightened ethical scrutiny.

### The Myth of "Public" Data

- **Social Media Reality:** Raw social media data almost always contains "personal data" (identifiable information).
- **User Expectations:**
  - Studies found most users were unaware their tweets were used for research.
  - Users generally expected to be asked for consent and for their data to be anonymised.
  - _Conflict:_ Previous platform policies sometimes required publishing the username, conflicting with user desires for anonymity.

### Examples of Ethical Dilemmas

- **Human Evaluation:** Asking people to rate machine translation output requires ethics approval as it involves human participants.
- **Anti-Spambot:** A bot designed to waste spammers' time requires approval because it interacts with humans without informed consent.
- **Audio Localisation:** Recording one's own phone audio for research requires approval if it captures the voices of others, which is considered personal data.

### Harmful Content

- Web data often contains sexually explicit content or hate speech, which poses risks when training general-purpose models.

## W601 - Attention in Seq2Seq

### The Sequence-to-Sequence Framework

- **Definition:** A general framework mapping an input sequence to an output sequence.
- **Versatility:** Most NLP tasks fit this structure:
  - **Translation:** Source language sentence $\rightarrow$ Target language sentence.
  - **Summarisation:** Long document $\rightarrow$ Short summary.
  - **Dialogue:** History of interaction $\rightarrow$ Next response.
  - **Multimodal:** Image $\rightarrow$ Text caption (or Speech $\rightarrow$ Text).
- **Goal:** Train a model to learn the mapping between these variable-length sequences.

### The Information Bottleneck Problem

- **Vanilla Encoder-Decoder:**
  - An RNN encoder processes the source and produces a **final hidden state** $h_f$.
  - This final state initialises the RNN decoder.
- Why this fails:
  - **Compression Bottleneck:**
    - The entire source sequence content must be compressed into a single, fixed-size vector.
    - This vector has **limited representational capacity**, regardless of input length (e.g., a sentence vs. a book).
  - **Recency Biases:** RNNs naturally emphasise later tokens because the final state is dominated by recent updates → Early information is forgotten.
  - **Decoder Limitation:** Every decoding step relies on the **same static vector**, even though different output tokens **need different parts of the input**.
- **Intuition:** It's like reading an entire book, making _one summary vector_, and then trying to answer detailed questions using only that one vector.

### The Solution: Attention Mechanism

- **Core Idea:** Instead of using one fixed vector, the decoder looks back at **all encoder states** and chooses which parts are relevant **at each decoding step**.
- **Dynamic Context Vector:** At output step $t$, compute a context vector $c_t$​ that summarises the input **specifically for predicting the next token**.
- **Weighted Average of Encoder States:** $c_t = \sum_{i} \alpha_{t,i}\, h_i$
  - **Read:** “Combine encoder states using weights $\alpha_{t,i}$​ that tell the model which source positions matter now.”
  - The summary is a weighted average of all encoder states from the last layer.
- **End-to-End Learning:** The model learns the weighting function jointly with the rest of the network.
- **What Attention Fixes**
  - **Removes bottleneck:** Decoder no longer depends on one compressed vector.
  - **Access to full input:** At each step, the model retrieves the relevant information.
  - **Better long-range behaviour:** Early and late tokens remain accessible.
  - **End-to-end training:** The model learns when and where to attend automatically.

### Attention Computation: Step-by-Step

At decoder step $t$, attention receives:

- the **current decoder state** $h_t$​
- **all encoder states** $s_1, \dots, s_m$

#### 1. Calculate Attention Scores

For each encoder state $s_k$​, compute a **scalar score** measuring _how relevant_ that source position is for generating the next output token: $\text{score}(h_t, s_k)$

- These scores are **logits** (un-normalised values).
- They encode raw importance before softmax.

#### 2. Normalise Scores → Attention Weights

- Apply **softmax** to the scores to obtain a **probability distribution** over encoder positions:
- Equation: $a_{k}^{(t)}=\frac{exp(score(h_{t},s_{k}))}{\sum_{i=1}^{m}exp(score(h_{t},s_{i}))}$.
- Convert importance scores to weights sum to 1 and fall between 0 and 1.
- Introduces a non-linearity to attention

#### 3. Compute Context Vector

- Form a **weighted sum** of encoder states:
  - Equation: $c^{(t)} = \sum_{k=1}^{m} a_{k}^{(t)} s_{k}$.
- This vector $c^{(t)}$ represents the source input specifically tailored for decoder step $t$.

### Motivation & Intuition

#### Optimisation Motivation: Better Gradient Flow

- Without attention:
  - The gradient from output token yty*tyt​ must travel \_through all earlier decoder states*, and then back through the encoder.
  - Leads to **vanishing gradients** and poor “credit assignment.
- With attention:
  - There is a **direct, short gradient path** from output $y_t$​ to each encoder token $x_k$​.
  - Improves stability and lets the model assign blame/credit to the _right_ input words.
- **Intuition:** The model no longer must remember everything in a single vector; it can directly “reach back” to relevant inputs.

#### Linguistic Motivation: Alignment & Reordering

- Different languages have different word orders
  - Alignments are often **non-monotonic** (e.g., Japanese ↔ English).
  - Decoder cannot rely on positional correspondence.
- **Attention advantage:** Enables the decoder to “look at” whichever source words matter, regardless of distance or order.

### Attention Score Functions

There are different ways to calculate the relevance score between decoder state $h_t$ and encoder state $s_k$:

- **Dot Product:** $score(h_t, s_k) = h_t^T s_k$.
  - Simple ad fast.
  - _Issue:_ As dimension $d$ grows, variance grows -> softmax saturate -> tine gradients
  - _Fix:_ Scale by $\sqrt{d}$ (used in Transformers).
- **Bilinear (Luong, 2015):** $score(h_t, s_k) = h_t^T W s_k$.
  - Uses a learned weight matrix $W$.
  - More flexible than dot product.
- **Additive / MLP (Bahdanau, 2015):** $score(h_t, s_k) = v^T \tanh(W [h_t; s_k])$.
  - Uses a non-linear multilayer perceptron.
  - Handles scaling well; widely used by RNN-base NMT.

### Combining Context with Decoder

- Once the context vector $c^{(t)}$ is generated, it must be combined with the decoder state $h_t$.
- Common approach: Concatenate $c^{(t)}$ and $h_t$ - $\tilde{h}_t = \tanh(W_c [h_t; c^{(t)}])$
- Send $\tilde{h}_t$​ to the next softmax prediction step.
- **Purpose:** Incorporate both the decoder’s internal dynamics and the attended source information.

### Interpretability

- **Heatmaps:** Attention weights can be visualised as a matrix (Source x Target), acting as a "soft alignment" between words.
- **Caveats:**
  - Attention weights **look like alignments**, but:
    - Encoder/decoder states are already **contextualised**.
  - **"Sinks":** Special tokens (like `<eos>`) may receive high attention not because they are meaningful, but because they aggregate global information across layers.
  - **Contextualisation:** By the last layer, encoder states already contain mixed information from the whole sequence.
- **Usage:** Despite limitations, it remains a useful analytical tool for inspecting model focus (reference paper: "Attention is not not Explanation").

### Generalisation

- Attention is **modality-agnostic**. It works for any sequence-to-sequence task.
- **Example - Image Captioning:** The model attends to specific pixels or patches of an image while generating the corresponding word in the caption (e.g., focusing on a frisbee when generating the word "frisbee").

## W602 - Self-Attention in Transformers

### Context and Motivation

- **Recap: Seq2Seq Components**
  - Classical encoder–decoder models rely on **RNNs** to encode sequences, decode sequences, and model encoder–decoder interactions.
- **Limitations of RNNs**
  - **Training Instability:** Prone to vanishing and exploding gradients due to back-propagation through time.
  - **Long-distance Dependencies:** Hard for RNNs to maintain information over long spans.
  - **Sequential Bottleneck:** Recurrence requires sequential processing (one token at a time). This prevents parallelisation and scaling to web-sized datasets.
  - **Information Bottleneck**: Encoder compresses entire sequence input to single vector
- **The Transformer Solution**
  - Replaces RNNs entirely with attention mechanisms for the encoder, decoder, and their interaction.
  - Leads to parallelisation, stability, better long-distance modelling and scalability.

### Core Concept: Self-Attention

- **Definition**
  - **Cross-Attention:** Decoder attends to encoder states (two different sequences).
  - **Self-Attention:** A sequence attends to **itself** (encoder self-attention, decoder self-attention).
  - **Parallelism:** Unlike RNNs, all tokens are processed **simultaneously**, not sequentially.
- **Function: Contextualisation** Each token representation is updated by aggregating information from **all** other tokens. Self-attention allows such dynamic resolution:
  - “The animal didn’t cross the road because **it** was tired.” → **it ↦ animal**
  - “...because **it** was too wide.” → **it ↦ road**

### Q,K,V Mechanism: Query, Key, and Value

Transformers project input tokens into three specific representations using learnable weights.

- **The Roles** Each input embedding is projected into:
  - **Query ($Q$):** "What am I looking for?" Represents the token looking for information
  - **Key ($K$):** "How should other tokens match me?" Represents the token offering information
  - **Value ($V$):** "Here is my information" The actual content to be extracted/averaged
- **Attention Computation**
  1. **Score**: $QK^T$ Calculate the dot product of the Query and Key
  - Measures similarity / relevance between tokens
  - $^T$ means take the transpose of this matrix.
  - $Q$ shape is ($N \times d$), $K$ shape is ($N \times d$ )
    - $N$ = number of tokens
    - $d$ = embedding dimension
    - $K^T$ has shape ($d \times N$) -> $QK^T$ has shape $(N \times d)(d \times N) = (N \times N)$
  - This $(N \times N)$ **attention score matrix** is one score for every pair of tokens
    - Each row of Q is a token asking, "which other tokens are relevant to me" and each column of $K^T$ is a token answering: here's how I identity
    - $QK^T$ computes **all pairwise similarities** between queries and keys.
  2. **Scaling:** Divide by $\sqrt{d_k}$, so $\frac{QK^T}{\sqrt{d_k}}$(square root of key dimensionality). This prevents large dot products and softmax saturation, thus stabilises variance and gradients
  3. **Softmax:** Apply softmax to normalise scores into a probability distribution (attention weights) where weights sum to 1.
  4. **Weighted Sum:** Multiply the probabilities by the Values ($V$) to get the final output representation.
- The Formula
- **Weighted Sum:**
  - $\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
  - **Read:** “Weight each Value vector by how relevant its Key is to the current Query.”

### Parallelisation and Complexity

- **Parallel Training**
  - Inputs are stacked into a matrix $X$ ($N \times d$), where $N$ is sequence length and $d$ is embedding dimension.
  - Q, K, V are computed via multiplying the learned weight matrices.
    $Q = XW_Q,\quad K = XW_K,\quad V = XW_V$
  - Compute **all pairwise attentions** at once → $N\times N$ matrix.
- **Complexity**
  - Because every token attends to every other token, attention has **quadratic complexity** $O(N^2)$ in sequence length.
  - This is the main memory bottleneck for very long sequences.

### Masking Strategies

There is a mismatch between parallel training and autoregressive inference. Masks are used to handle this.

- **Causal (Look-ahead) Mask - Decoder**
  - **Purpose:** Simulates test-time conditions during training. The decoder cannot "see" or attend to future tokens it hasn't generated yet.
  - **Method:** In the attention score matrix, set values for future positions (upper right triangle) to $-\infty$ so softmax gives weight of 0, preventing information flow from future.
  - **Note:** The diagonal is _not_ masked; a token can attend to itself.
- **Padding Masking**
  - **Purpose:** Handles variable sequence lengths in a batch.
  - **Method:** Mask out `<pad>` tokens by setting to $-\infty$ so the model ignores them during attention calculation.

### KV Cache (Inference-time optimisation)

- **The Problem**
  - During autoregressive decoding (generation), the decoder produces **one token at a time**.
  - Recomputing all Keys and Values for all previous tokens at every step is wasteful.
- **The Solution**
  - Compute $K$ and $V$ for the current token once and store them in a cache (memory).
  - At the next step, only compute the Query for the new token and retrieve previous $K$ and $V$ from memory.
  - **Trade-off:** Much faster (reducing computation), but cache grows linearly with output length.

### Multi-Head Attention

- **Concept**
  - Run multiple self-attention computations (heads) in parallel
  - Outputs are concatenated and projected linearly to form the final result.
- **Motivation**
  - Natural language contains diverse relationships (e.g., syntactic dependencies, local context).
  - Single-head attention struggles to capture all these simultaneously.
- **Interpretability**
  - Different heads often specialise in specific tasks.
    - **Position Heads:** Track relative positions (e.g., looking at the previous token).
    - **Syntactic Heads:** Track grammatical relations (e.g., subject-verb, verb-object).

## W603 - Transformer Architecture

### Transformer Layer = Attention + FFN

A standard layer comprises four main components.

1. **Multi-Head Attention (Sequence Mixing)** Mixes information _between token_
2. **Feed Forward Networks (FFN)** Mixes information _within a tokens vector_
   - A single attention layer is limited to the "convex hull" of value representations (it's just a weighted average). It lacks sufficient expressivity to re-map token representations into richer spaces.
   - FFN introduces a non-linearity and allows richer representations.
   - A Multi-Layer Perceptron (MLP) applied to each token independently:
     - $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$
   - Operates on individual tokens to mix their internal dimensions.
3. **Residual Connections (Highway Connections)**
   - Adds the input of a transformer block directly to its output
     - $Output = Input + Block(Input)$.
   - Functions as a **highway channel** through which information flows unchanged - bypass modules preventing vanishing gradients.
   - Enables the **deep stacking of layers** with stable gradients. Allowing gradients to flow through the "residual stream" to lower layers.
   - The model **learns small, additive edits** to the token representation rather than rewriting it entirely at every layer.
4. **Layer Normalisation (LayerNorm)**
   - Stabilises training by keeping representation scales consistent.
   - Subtracts the mean and divides by the standard deviation (computed across the _dimensions_ of a single token).
   - Applies **learnable scale and bias parameters**.
   - Pre-LN vs. Post-LN: Normalisation pre/post attention/FFN blocks
     - **Post-LN (Original)**: Unstable in very deep modules
     - **Pre-LN**: Improves gradient flow; each block becomes a "small correction to identity"

### Token Representations across layers

Visualising how token embeddings (e.g., "is", "are", "was") change as they move up through the layers.

**Empirical observation via t-SNE:**

- **Lower Layers (Input):** Representations cluster strictly by word type (e.g., all "was" tokens group together).
- **Encoder Higher Layers (e.g., Translation Source):**
  - Clusters remain distinct.
  - **Reason:** The encoder must retain the specific identity of the input word to allow the decoder to translate it correctly (e.g., preserving singular/plural or gender information).
- **Decoder Higher Layers (e.g., Language Model):**
  - Clusters merge and disappear; representations become "messy" and mixed.
  - **Reason:** The goal shifts from representing the _current_ token to predicting the _next_ token. The representation is heavily contextualised to serve the prediction task.

**Roles of token representation**

- Help predict the next output
- Help build representations for other tokens
- These tradeoff differently in encoder vs. decoder

### The Three Transformer Architectures

Stacking layers creates three main model types.

#### Encoder-Only

- **Example:** BERT
- **Uses:** Understanding/Classification/Encoding
- **Nature:** Non-generative; outputs contextualised embeddings for input labels.
- **Structure:** Unmasked self-attention only.

#### Decoder-Only (Causal / Autoregressive)

- **Example:** GPT series
- **Uses:** Language Modelling
- **Nature:** Generative; Predicts next token auto-regressively; models input and output as one continuous sequence.
- **Structure:** Contains only Masked Self-Attention and FFNs. No cross-attention.
- The field has moved heavily toward this architecture for Large Language Models (LLMs).

#### Encoder-Decoder (Seq2Seq)

- **Example:** Original Transformer
- **Uses:** Machine Translation
- **Encoder:** Unmasked self-attention (sees full context).
- **Decoder:** masked self-attention + cross-attention (decoder queries attend to encoder keys/values)

## W701 - Transformer Inputs and Outputs

### Inputs: Token Embeddings

- A sentence is tokenised into a sequence of indices from the vocabulary $V$
  - **Vocabulary ($V$):** Consists of words or subwords (e.g., BPE, Unigram).
- **Embedding Matrix:**
  - Each token / index retrieves a **row** from the embedding matrix $E \in \mathbb{R}^{|V|\times d}$
    - Matrix shape is $\text{Vocabulary Size} |V| \times \text{Embedding Dimension} \quad d$
- Output shapes: $\text{Batch} \times \text{SeqLen} \times d$
- **Function:**
  - This vector encodes the **type-level semantics** of a token (independent of context).
  - Transformers operate on vectors; embeddings turn symbolic tokens into continuous vectors suitable for attention operations.

### Inputs: Positional Information

- **The Problem:** Transformers are **permutation-invariant**.
  - Self-attention depends only on dot products between Q/K/V — **position plays no role**.
  - FFN operates **position-wise** and cannot inject order.
  - Therefore, **scrambling the tokens produces identical internal activations** → We must inject **position** externally.
- **Method 1: Learned Absolute Position Embeddings**
  - Each absolute position $i$ (1,2,3,…) has a learned vector $p_i$
  - We add this to our embedding, so input becomes: $x_i = e_i + p_i$
  - **Limitations:**
    - **Poor Extrapolation:** Cannot handle sequences longer than the maximum seen during training.
    - **Undertraining:** High-index positions (e.g., position 512) appear rarely so are poorly learned
    - **Wrong Bias:** Natural language relies on **relative** positions (syntax), not absolute indices.
- **Method 2: Sinusoidal Periodic Encoding**
  - Uses fixed Sine/Cosine waves at different frequencies to generate fixed position vectors.
  - Like binary encoding: lower-frequency waves track coarse structure; high-frequency waves track fine-grained positions.
  - **Advantages:**
    - **Extrapolates** to unseen lengths.
    - **Preserves locality** as nearby positions share similar patterns.
    - **Relative Distances** are naturally **encoded** (phase differences)
    - **No parameters to learn**.
- **Method 3: Relative Position Encoding (Modern Approach)**
  - Encodes the distance _between_ tokens directly into the attention mechanism.
  - **RoPE (Rotary Positional Embeddings):** Rotate Q and K vectors in a way that naturally encodes relative positions.
  - **ALiBi (Attention with Linear Biases)**: Adds a position-depended bias directly to the attention scores.
    - Encourages attending more to recent tokens; extends to arbitrarily long sequences
  - Relative distances generalise better across sequence lengths and task

### Outputs: The Language Modelling Head (unembedding layer)

- **Definition:** The final linear layer that **maps hidden states** back to **vocabulary logits**
- **Mechanism:**
  - A linear projection mapping the hidden dimension $d$ back to the vocabulary size $|V|$.
  - Often shares weights (transposed) with the input embedding matrix.
  - Produces **logits** (scores) for each token in the vocabulary.
  - Mathematically:
    - Hidden state at final layer: $h^L_i \in \mathbb{R}^d$
    - Unembedding matrix $E^\top \in \mathbb{R}^{d \times |V|}$
    - Logits: $l_i = h^L_i E^\top$
- **Softmax:** Converts logits into a probability distribution over the vocabulary

### Generation (Decoding) Strategies

- **Greedy Decoding:**
  - Selects the single most probable token at each step $argmax$
  - Fast, deterministic.
  - Often suboptimal because it cannot plan ahead for the full sequence. (_i.e - locally optimal, not globally optimal)_
- **Beam Search:**
  - Tracks top $k$ (beam size) partial hypothesis at each step.
  - Expands each and keeps only best $k$ (Prunes least likely branches to maintain a fixed number of candidates.)
  - Balances exploration and explotiation
- **Sampling:**
  - Used when diversity is desired (creative tasks)
  - Selects the next token randomly based on the probability distribution to balance **Quality** vs. **Diversity**.
  - **Top-K:** Samples from the top $k$ highest logits only. Simple but inflexible:
    - For flat distributions → may discard too much
    - For peaky ones → may keep many tiny-probability tokens.
  - **Top-P (Nucleus):** Samples from the smallest set of tokens whose cumulative probability $\ge p$.
    - Dynamic: adapts dynamically to the distribution shape.

### Training Generative Transformers

- **Forward Pass:**
  - Input Sequence $\rightarrow$ Embeddings $\rightarrow$ Transformer Layers $\rightarrow$ Logits
  - Compute predicted distribution for each next token
  - Apply cross-entropy loss
- **Teacher Forcing:**
  - At step $t$, the model is provided with the **true** tokens $x_{1..t-1}$​ rather than its own predictions.
  - Stabilises training and accelerates convergence.
- **Loss Function:**
  - Uses **Cross Entropy Loss**, which minimises the negative log probability of the correct next token.
- **Backward Pass:**
  - Backpropagates loss to update parameters using an optimiser (SGD, Adam).

## W702 - Transfer Learning and BERT

### Transfer Learning Concepts

- **Definition (general)**
  - Transfer learning = **take a model trained on a large, general “base” task and adapt it to a smaller, specific “target” task**.
- **Definition (deep learning)**
  - Train a **base** network → reuse its **learned features** to initialise a **target** network → fine-tune on a new dataset.
- **When does it work?**
  - When the pre-trained features are **general enough** to be relevant across tasks (e.g., shapes + textures in vision; syntax + semantics in NLP).

### Transfer learning in Computer Vision (Origins)

- ImageNet-trained models learn universal low-level and mid-level features (edges, textures, object parts). Two main ways to resuse them:
  - **Feature Extraction:**
    - Removing final classification layer of a pre-trained model
    - Freezing all convolutional layers
    - Train a new classifier on top of the extracted features.
    - Cheap and Stable
    - **Fine-tuning:**
      - Replace the output layer
      - Continue training **all** (or most) parameters on the target task
      - Usually requires **smaller learning rates** to avoid destroying pretrained knowledge.
    - **Weight Freezing:**
      - Freeze early layers (generic features)
      - Fine-tune later layers on task-specific patterns

### First Era NLP: Static Embeddings (Word2Vec - skip-gram/CBOW, GloVe)

- **Static:** Each word type has one vector, reglardless of its sentence context.
- Vectors used as input features (Models represent **feature extraction**)
- **Usage:** Used to initialise input layers, but the rest of the model is trained from scratch (i.e - RNNs)
- Embeddings could be fine-tuned slightly, but only at the input layer
- **Limitations**
  - Cannot represent polysemy (play - sports vs. theatre vs. child)
  - Cannot adapt deeply to downstream tasks

### The NLP Shift: Contextualised Embeddings (BERT, GPT, T5):

- Contextual embeddings assign **different vectors for the same word type depending on context** (preceding and following words).
  - Expensive to pre-train from scratch
  - **Very cheap to fine-tune** for new tasks
  - A single architecture that can be adapted across a wide range of NLP tasks
- **Architectures:**
  - **Encoder-only:** BERT → for classification, tagging, QA.
  - **Decoder-only:** GPT → for generation.
  - **Encoder–decoder:** T5 → translation, summarisation, QA.

### BERT Architecture

- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.
  - A stack of **Transformer encoder** layers operating **bidirectionally** (attending left + right)
- **Model Sizes:**
  - **BERT-Base:** 12 layers, 768 hidden dim, 12 heads, 110M parameters.
  - **BERT-Large:** 24 layers, 1024 hidden dim, 16 heads, 340M parameters.
- **Why Bidirectional?**
  - Unlike GPT (causal), BERT attends to the entire context on both sides — crucial for language understanding tasks.
- **Scaling Laws:** Larger models and more data generally lead to better downstream performance.

### BERT Input Representation

- **Tokenisation:** Uses **WordPiece** vocabulary (30,000 subword tokens).
- **Special Tokens:**
  - `[CLS]`: Prepended to the start of every input. Captures aggregate sentence representation for classification tasks.
  - `[SEP]`: Separates two sentences (e.g., for QA or entailment).
- **Embedding Summation:** Each input representation is the sum of three embeddings:
  1. **Token Embeddings:** The subword vector.
  2. **Segment Embeddings:** Identifies if a token belongs to Sentence A or Sentence B.
  3. **Position Embeddings:** Learned absolute positions. $\text{InputEmbedding} = \text{TokenEmbedding} + \text{SegmentEmbedding} + \text{PositionEmbedding}$

### BERT Pre-training Tasks

BERT is trained on unlabeled text (Wikipedia + BookCorpus) using two self-supervised tasks.

#### 1. Masked Language Modelling (MLM)

- **Goal:** Predict randomly masked tokens using **full bidirectional context**.
- **Procedure:** 15% of all input tokens are selected.
  - **80%:** Replaced with the `[MASK]` token.
  - **10%:** Kept as the original word (to bias the model toward keeping correct inputs).
  - **10%:** Replaced with a random word (adds noise/regularisation).
  - **Why not always MASK?** Because `[MASK]` never appears at test time; BERT must learn to handle clean and noisy contexts.

#### 2. Next Sentence Prediction (NSP)

- **Goal:** Binary classification task: Predict if Sentence B follows Sentence A.
- **Data Setup:**
  - 50% of the time: B is the actual next sentence.
  - 50% of the time: B is a random sentence from the corpus.
- **Utility:** Teaches the model sentence-level relationships (stored in `CLS` token), crucial for tasks like Question Answering and Natural Language Inference (NLI).

### BERT Fine-Tuning

- **Process:**
  - Start with pre-trained BERT weights.
  - Add a task-specific output head
    - Classifier on top of `[CLS]`
    - token-level -> QA Spans
  - Train **all** weights with a low learning rate on the target dataset (AdamW)
- **Why fine-tuning works extremely well**
  - BERT already knows language structure.
  - Task-specific training only needs to “nudge” BERT towards the downstream objective.
- **Performance:**
  - BERT achieved SOTA on 11 NLP tasks (GLUE benchmark)
- **Fine-tuning vs. Feature Extraction:**
  - Fine-tuning (updating all weights) -> Best Performance
  - Feature Extraction (freezing weights) -> OK, but less expensive to train
  - Using deeper layers or weighted combo improves results

### Why BERT Changed NLP

- Pre-training + fine-tuning became **the dominant paradigm.**
- Pre-training is **expensive** (BERT training ≈ 100 GPU-days), so few institutions can do it.
- Fine-tuning is cheap and accessible.
- Led to explosion of specialised BERT variants across domains: biomedical, multilingual, legal, scientific
- Limitation: encoder-only -> cannot generate variable-length sequences

## W703 - Architectures for Language Models

### T5: Text-to-Text Transfer Transformer

#### A. The Core Concept: Everything is text-to-text

- **Unified Framework**:
  - T5 treats _every_ NLP task (classification, regression, translation) reframed as feeding text in and producing text out
- **Task Examples**:
  - **Translation** (Lang-Lang): Input "translate English to German: That is good" $\rightarrow$ Output "Das ist gut".
  - **Classification** (Input-Label): Input "cola sentence:..." $\rightarrow$ Output "not acceptable"
  - **Regression** (Input-Number Strings): Input "stsb sentence1:..." $\rightarrow$ Output "3.8"

#### B. Architecture Changes

- Encoder-Decoder Transformer.
- **Size**: ~220M parameters (roughly $2\times$ BERT Base) because it stacks an encoder and decoder.
- **Positional Encodings**:
  - Uses **learned _relative_ position embeddings** applied to attention scores (offsets between queries and keys) rather than absolute embeddings added to inputs.

#### C. Pre-training Objective: Span Corruption

- **Objective**: A denoising objective where the model predicts missing (corrupted) spans.
- **Process**:
  1. Randomly **mask 15%** of tokens.
  2. Replace consecutive spans of dropped tokens with a single unique **sentinel token** (e.g., `<X>`, `<Y>`).
  3. **Input**: Corrupted sequence with sentinel placeholders: _"Thank you `<X>` me to your party `<Y>` week"._
  4. **Target**: Output contains _only_ the sentinels and the missing content: "`<X>` for inviting `<Y>` last `<Z>`".
- **Why Sentinels?**:
  - Unlike BERT's generic `[MASK]`, sentinels **handle variable-length spans**.
  - Forces **the model to learn dependencies** within the masked span.
  - Prepares the model for **generation tasks where input and output lengths differ**.

#### D. Data & Cleaning (C4 Corpus)

- **Source**: _Colossal Clean Crawled Corpus_ (C4), derived from Common Crawl.
- **Size**: 365M documents, 156B tokens.
- **Heuristics for Cleaning**:
  - Discard pages with fewer than 3 sentences.
  - Discard lines not ending in punctuation.
  - Remove obscene words and lines containing "Javascript" or code.
  - Deduplicate

#### E. Fine-tuning & Scaling

- **Task Format**: Inputs are **prepended with a task** description (e.g., "summarise:", "qnli question:").
- **Benchmarks**: Evaluated on GLUE, SuperGLUE which contain a mix of NLP tasks (classification, NLU, QA, summarisation).
- **Scaling Laws**: T5 comes in sizes from Small (60M) to 11B parameters.
  - Performance improves consistently as model size (depth and width) increases.

### The GPT Family (Decoder-Only)

#### A. Architecture & Objective

- **Type**: Autoregressive / Causal Language Model (CLM).
- At each step predict $P(x_t \mid x_{<t})$
- **Mechanism**:
  - Decoder blocks only
  - **Causal masking** (tokens can only attend to the left/past context).
- **Objective**: Predict the next token given the sequence so far.

#### B. Evolution of GPT Models

- **GPT-1 (2018)**:
  - 117M parameters.
  - Trained on **BooksCorpus** (7,000 unpublished books).
  - Fine-tuned by adding **linear classifiers** for specific tasks.
- **GPT-2 (2019)**:
  - 1.5B parameters.
  - **Pre-LN**: LayerNorm moved to the input of each sub-block.
  - **Context**: Increased context size from 512 to 1,024 tokens.
  - **Data**: Trained on **WebText** (8M documents of filtered crawl), excluding Wikipedia.
- **GPT-3 (2020)**:
  - 175B parameters.
  - Introduced **in-context learning** (few-shot prompting)

#### C. Adapting Tasks for GPT

- **Classification**: Input text $\rightarrow$ Linear Head.
- **Entailment**: Concatenate "Premise + Delimiter + Hypothesis" $\rightarrow$ Transformer $\rightarrow$ Linear.
- **Multiple Choice**: Process Context + Answer N pairs separately, then softmax the results.

### Summary of Trends

- **Three Main Architectures**:
  1. **Encoder-only (BERT)**: Autoencoding (Masked Tokens) - understand & classify
  2. **Encoder-Decoder (T5)**: Sequence-to-Sequence with Denoising (Span Corruption) - Text to Text
  3. **Decoder-only (GPT)**: Autoregressive (Next-Token Prediction) - Generate
- **Performance**: General trend that larger models (parameters/data) yield better performance.
- **Unification**: Most NLP tasks are now reformatted as text generation tasks.

## W801 - In Context learning

### GPT-3 Overview

- **Architecture:**
  - Decoder-only transformer, essentially a scaled-up GPT-2.
  - **Scale:** 175 billion parameters, 96 layers, 96 heads, 12k dimensional vectors.
  - **Context Window:** Significant growth (modern models reach 1M tokens vs. BERT's 512).
- **Training:**
  - Trained on Microsoft Azure clusters **(approx. cost $10M).**
  - GPT-4/5 architecture is undisclosed but speculated to be a **"Mixture of Experts".**
- **Data Mixture:**
  - Combines Common Crawl (web text @ 60%) + Books, and Wikipedia.
  - **Sampling:** High-quality data (Books/Wiki) is up-sampled; large/low-quality data (Web) is down-sampled.
  - **Sampling Unit:** Historically sentence-level; moving toward paragraph/document level to preserve context.
  - **Contamination:** Hard to fully remove test set data from training data (GPT-3 had a known bug here).

### The Paradigm Shift: In-Context Learning (ICL)

- **The Old Way (Fine-tuning):**
  - Requires gradient updates (backpropagation) for every batch.
  - Prohibitive cost for massive models like GPT-3 (175B params).
- **The New Way (ICL):**
  - **No weight updates:** Uses the frozen pre-trained model directly.
  - **Forward pass only:** Casts tasks as language modeling problems.
  - Transfers pre-training skills to inference time by providing a prompt with task info and examples.
  - Uses **temporary context memory** rather than persistent parameter memory.
  - **Key Idea:** Pre-training gives the model general pattern-recognition abilities; during inference, it _mimics learning_ using examples provided _in the context window_, not in parameters.

#### ICL Modes:

- **Zero-shot:** Task description + Input $\rightarrow$ Output.
- **One-shot:** Task description + 1 Example + Input $\rightarrow$ Output.
- **Few-shot:** Task description + $k$ Examples + Input $\rightarrow$ Output.
  - _Performance Trend:_ Few-shot > One-shot > Zero-shot.

### Capabilities & Performance

- **Task Formatting:**
  - Classification (e.g., sentiment) is treated as predicting the next word (e.g., "positive" or "negative").
  - Can constrain output to specific labels of interest.
- **Benchmarks:**
  - **TriviaQA:** GPT-3 ICL outperformed fine-tuned SOTA models.
  - **SuperGLUE:** GPT-3 outperforms fine-tuned BERT (specialised for classification), proving general-purpose capability.
  - **NLG:** Produces fluent, coherent, and grammatically correct text.
- **Emergent Abilities:** New skills (like ICL) emerge only when models reach a certain scale.

### Why Use In-Context Learning?

- **Data Efficiency:** Useful when labelling data is expensive/scarce/require domain expertise (e.g., legal/medical).
- **Resource Efficiency:** Avoids the high cost of fine-tuning massive parameters
- **Time:** Allows immediate deployment without new training cycles.
- **Cognitive Test:** Demonstrates "intelligent behaviour" via skill transfer (similar to humans applying bike skills to scooters).

### Prompt Engineering & Robustness

- **Prompt Templates Matter:**
  - Even small rewordings of the task prompt produce large accuracy swings.
  - An "art" of formatting tasks to look like pre-training data. _Example:_ Rephrasing "Input: X, Label: Y" as "Q: What is the sentiment of X? A: Y".
- **Sensitivity/Brittleness:** Models are highly sensitive to prompt variations. Performance fluctuates significantly based on:
  - **Example Selection:** Which examples are included.
  - **Order:** The permutation of examples in the context -> large variance in accuracy.
  - **Templates:** The wording used to structure the prompt.
- **Instruction Tuning:** Note that GPT-3 was _not_ instruction tuned; it relies solely on pattern matching from web text.

### Summary & Outlook

- **Scaling Laws:** Bigger models, more data, and longer training consistently yield better results.
- **No Plateau:** We have not yet seen an end to the trend where increasing scale improves performance.

## W802 - Scaling Laws and Evals

### History of LLMs

- **1990s: Statistical LMs** N-grams, count-based probability models
- **2013: Word2Vec** introduced _task-agnostic feature learning_ and static embeddings.
- **2014-2015: RNNs, LSTMs, Attention** Enables modelling of long-distance dependencies
- **2017 Transformers** parallelisation + attention = breakthrough
- **2018: Pre-trained contextual models** (BERT, GPT-1/2) provided transferable, context-aware representations.
- **2020+:** **Large Language Models**
  - General-purpose task solvers with in-context learning

### Scaling

**Scaling is not “make the model bigger”. It is:**

1. **More compute per step** (forward/backward passes)
2. **More training iterations** (more tokens).
3. **A model large enough** to _use_ the additional compute.
   **Key Insight**

- **Small models:** Plateau early as they lack the capacity to use extra compute.
- **Large models:** Continue to reduce loss as compute increases.

### Measuring Compute: Petaflop-s-days

- **1 Petaflop = $10^{15}$** floating-point operations per second.
- Example:
  - A 100-petaflop supercomputer running for 100 days = 10,000 petaflop-s-days.
- Compute budget is the _core currency_ of large-scale training.
- **Goal:** Train models to **optimality** (best loss for a given budget) rather than to convergence.

### Scaling Laws

Empirically, **test loss follows a power law** in:

- **N** = model parameters
- **D** = dataset size (tokens)
- **C** = compute
- Mathematically
  - $L(N) \approx (\frac{N_{c}}{N})^{\alpha_{N}}$
  - $L(D) \approx (\frac{D_{c}}{D})^{\alpha_{D}}$
  - $L(C) \approx (\frac{C_{c}}{C})^{\alpha_{C}}$
- **Empirical Exponents:**
  - Compute: $\alpha_C \approx 0.05$
  - Dataset: $\alpha_D \approx 0.095$
  - Parameters: $\alpha_N \approx 0.076$
- **Meaning:**
  - Test loss decreases **smoothly & predictably** as you grow compute, data, and parameters.
  - You can **train small models → fit power law → extrapolate** how a large model will behave.

#### Parameter Count Approx ($N$)

- For non-embedding parameters in a Transformer:
  - $N \approx 12 \cdot n_{layer} \cdot d^2$ (where $d$ is the hidden dimension).
  - **Example (GPT-3):** 96 layers, $d=12288$ $\rightarrow$ $\approx$ 175 billion parameters.

#### Optimising for a Compute Budget

- If the compute budget increases by **100x**, the optimal increase is
  - **Model size:** $N_{\text{opt}} \propto C^{0.73} \quad \Rightarrow \quad 100^{0.73} \approx 29\times$
  - **Dataset size:** $D_{\text{opt}} \propto C^{0.27} \quad \Rightarrow \quad 100^{0.27} \approx 3.5\times$

**Interpretation:** Most of the gain goes into **bigger models**, not dramatically bigger datasets. This enables **planning large-scale training without trial and error**.

### LLM Evaluation

#### Intrinsic Metrics (Cheap, Automatic)

- **Validation Loss:**
  - Negative log-likelihood of held-out data.
  - Highly correlated with downstream performance.
- **Perplexity:**
  - A length-normalised variant of probability
  - Useful for comparing sequences of different lengths.
  - $\sqrt[n]{\frac{1}{\prod_{i=1}^{n}p_{\theta}(x_{i}|x_{<i})}}$

#### Extrinsic Metrics (Knowledge & Reasoning)

- **MMLU (Massive Multitask Language Understanding):**
  - Tests world knowledge across **57 subjects** (e.g., math, law, prehistory).
  - Multiple-choice questions (4 choices).
  - **Accuracy** metric (Random baseline is 25%).

#### Extrinsic Metrics (Conversational)

- **Challenge:** Generative tasks have infinite output spaces and often lack a single "correct" reference.
- **Human Pairwise Judgement (LM Arena):**
  - Users blindly vote on the better response between two anonymous models
  - Produces **win rate** or **Elo Score:** Ranks models based on win rates and the quality of opponents (similar to Chess rankings).
- **LLM-as-Judge:**
  - Strong LLM evaluates output of weaker ones
  - _Pros:_ Scalable, cheap, and fast.
  - _Cons:_
    - Biased towards verbosity (longer answers)
    - Self-preference
    - struggles to detect fluent hallucinations.

#### Task-Specific Metrics (Reference-Based)

Used for tasks like translation or summarisation where a reference exists.

- **NMT (Machine Translation):**
  - **BLEU:** Word-level n-gram overlap (precision).
  - **chrF:** Character-level overlap.
- **Summarisation:**
  - **ROUGE:** Recall-oriented (checks if key information from the reference appears in the output).

#### Beyond Accuracy

- **Efficiency:**
  - Latency, memory load, energy consumption, environmental impact.
- **Fairness:**
  - **Bias:** Models must be checked for stereotypes (e.g., gender, race) using datasets like StereoSet and BBQ.
  - **Multilinguality:** Current models skew heavily towards high-resource languages (English/Chinese), leaving many communities underrepresented.

## W803 - Memory and Compression

### Background: Hardware & Algorithms

**Three Hardware Limits**

- **Memory Capacity:**
  - Can the model + KV cache even fit in GPU RAM?
  - Example: 100GB model → requires ≥100GB GPU memory.
- **Memory Bandwidth:**
  - How fast data moves between memory and compute.
  - H100 transfers **3.35 TB/s**.
- **Compute Bandwidth:**
  - How fast operations can be executed.
  - H100 → ~**1 petaFLOP/s**

**Algorithm Trade-offs:**
Two algorithms can compute _the same output_ but differ drastically in speed because they use different resources:

- **Compute-Intensive (Algorithm A):**
  - Low memory usage but high computation (compute-bound).
  - Can appear “efficient” because GPU utilisation is high—but slow.
- **Memory-Intensive (Algorithm B):**
  - Requires extra memory but performs less computation; speed is limited by bandwidth.
  - **100× faster** because it avoids compute bottlenecks.
- **Key Insight:** GPU “utilisation” ≠ speed. An algorithm bottlenecked on memory can be much faster than one bottlenecked on compute.

### Transformer Bottlenecks

- **Two Phases of Inference:**
  - **Prefill (Processing the input prompt)**
    - Computes all pairwise Query-Key dot products.
    - Cost is quadratic $\mathcal{O}(N^2)$.
    - This phase is **compute-bound** (dense matmuls).
  - **Decoding (Generating tokens one-by-one)**
    - For each new token, the model attends to **all previous KV Cache entries**
    - Avoids re-computation -> cheap compute, but expensive **memory reads**.
    - **Memory bandwidth-bound**, not compute-bound
- **Tokenisation Matters:**
  - **Tokens define the problem size.** More tokens → larger KV cache → more reads during decoding.
  - Different languages produce _massively different_ token counts
    - English: 7 tokens
    - Burmese: 61 tokens (same semantic meaning).

### Part 1: Dynamic Memory Compression (DMC)

- **Goal:** Reduce the size of the KV cache by allowing the model to **merge** information instead of storing every token separately.
- **The Problem:** Models process all timesteps uniformly (treat all tokens as equally important), regardless of information density (e.g., silence in audio or redundant / predictable tokens).
- **DMC method:**

  - At each decoding step, the model chooses:
    - **Append** the new KV pair (normal behaviour) → $\alpha_t = 0$
    - **Accumulate / merge** into the previous KV → $\alpha_t = 1$
  - Merging is a **learned weighted average** of old + new representations
  - This produces a **compressed KV cache** without harming model accuracy.

- **Key Findings from LLaMA Experiments**
  - **1. Compression propagates from deeper layers first**
  - Early layers: capture detailed local info → cannot compress much.
  - Later layers: abstract concepts → easily compressible.
  - **2. Compression increases with token position**
    - Early tokens: high surprisal = important → stored more precisely.
    - Later tokens: predictable → highly compressible.
  - **3. Performance:**
    - Able to achieve **8x memory compression** on Llama without performance loss.
    - Huge win because decoding is memory-bandwidth-bound
  - Limitations:
    - Context windows can still reach **millions of tokens**.
    - KV cache size remains a major bottleneck.
    - Need a second technique: **Sparse Attention**.

### Part 2: Sparse Attention (Skip unimportant interactions)

- **The Long Sequence Challenge:**
  - Real workloads often require extremely large contexts:
    - 10-K financial reports → 100k tokens
    - War & Peace → 600k tokens
    - Entire codebases → millions of tokens
  - Dense attention = **O(N²)** → impossible at these scales.
- **Core Hypothesis:** Attention is naturally sparse; most token interactions have low importance. **We can skip computing them.**
  - Reduces compute costs, memory bandwidth
  - Allows much longer context lengths
- **The Challenge:** We must predict which attention interactions matter **_without_ actually computing them first.**

#### Design Choices for Sparse Attention

1. **Units of Sparsification:**

   - **Block-Sparse:**
     - Groups tokens into $B\times B$ blocks.
     - Assumes locality (important interactions cluster together).
   - **Vertical-Slash:**
     - Keep all diagonal blocks (local context), and
     - Selected global "key" columns
     - Useful for tasks with "global" tokens (e.g code structuyre)

2. **Importance Estimation:**

   - **Block Aggregation:** Uses average representations of tokens within a block to estimate relevance.
   - **Suffix Voting:** Uses the last few tokens (suffix) to "vote" on which earlier positions are important.

3. **Budget Allocation:**

   - **Fixed:** Keep top-k blocks or stripes. Simple, robust.
   - **Adaptive:** Allocate sparsity depending on input complexity. More flexible but complex / fragile.

### Key Research Findings

- **Task-dependent sparsity tolerance:**
  - Single-question QA handles extreme sparsity well
    - Only a few tokens contain the answer.
  - Multi-question QA degrades quickly
    - More uniformly important tokens
- **High average tolerance, but high variance:**
  - While models tolerate $>90\%$ average sparsity (keep only top 8%) - variance is high.
  - Some tasks break quickly even at moderate sparsity, requiring careful testing per task.
- **Efficiency Frontier:**
  - At a fixed compute budget (Pareto frontier), **large sparse models outperform small dense models**.
  - Meaning: computation saved via sparsity lets you scale width/depth instead

### Conclusion & Future Directions

- **Complementary Methods:**
  - **Memory Compression:** Reduces memory footprint/bandwidth (addresses storage inefficiencies).
  - **Sparse Attention:** Reduces computational cost (skips unimportant interactions).
- **Open Questions:**
  - How to jointly optimise compression and sparsity in a unified framework.
  - How to automatically predict safe sparsity levels (adaptive budgets) without extensive testing.

## W901 - Multilingual LLMs

### The State of World Languages

- **Imbalance in Speaker Distribution**: There are ~7,159 living languages, but the distribution is highly skewed.
  - 44% are endangered (<1,000 speakers).
  - The **top 20 languages** cover **nearly 50% of the world's population**.
- **Data Paucity:** Languages are categorised by data availability - **availability of labelled + unlabelled data** (Groups 0–5).
  - **Group 5:** High-resource (English, Spanish, German).
  - **Group 0:** Low-resource, often indigenous
    - Almost no labeled or unlabeled data.
    - Few speakers or limited digital access (socioeconomic factors)

### Motivation for Multilingual LLMs

- **Societal Motivations:** Access to healthcare, legal services, digital participation and cultural preservation often depend on language support. Language Technology can **help preserve cultural identity.**
- **Linguistic + ML Motivations:** Current models are tuned for high-resource features; **48% of typological features exist only in low-resource languages**.
  - → High-resource–centric models cannot capture the full typological diversity.
- Current solutions built for English may **fail** on languages with:
  - Rich morphology
  - Complex agreement systems
  - Different word orders
  - Non-Latin scripts
- **Approach:** Instead of training thousands of monolingual models, we **_train massive multilingual LLMs to enable knowledge transfer_**.

### Cross-Lingual Transfer Strategies

- **Zero-Shot Transfer:** Train on source languages (e.g., English, German); evaluate directly on a target language (e.g., French) without training on it (no labelled french data)
- **Translate-Test:** translate inputs from target (French) to source (English) at inference. Use a monolingual model (English);
- **Translate-Train:** Translate training data from source (English) to target (French); fine-tune the model on the translated data. (Fine-tuned on synthetic data)

### Technical Challenges & Solutions

- **Curse of Multilinguality:** Adding **_too many languages can exhaust the model's parametric capacity_**, causing performance drops.
- **Data Curation:**
  - Requires robust **Language Identification** (LID) and **Unicode normalisation** for different scripts.
  - **Datasets often mix monolingual and parallel data** (e.g., mC4, ROOTS).
- **Sampling:** Training **data is smoothed to over-sample low-resource languages** and under-sample high-resource ones to ensure coverage.
- **Architectures:**
  - **Encoder-only:** mBERT (104 languages).
  - **Encoder-decoder:** mT5 (trained on mC4).
  - **Decoder-only:** BLOOM (46 natural languages + 13 programming languages).
- **Alignment:** Shared parameters allow cross-lingual alignment to emerge naturally, even without parallel data. Contrastive objectives (like LaBSE) can further improve this.

### Tokenisation & Fertility

- **Definition:** "Fertility" is the **average number of tokens per word**.
- **The Problem:** Tokenisers are biased toward high-resource languages.
  - _Example:_ A Welsh sentence -> 64 tokens, while the English equivalent -> 33.
- **Consequences:** Higher fertility increases sequence length, making processing low-resource languages **unfairly more expensive** (quadratic attention cost).
- **Solution:** "Token-free" models (e.g., ByteT5) operate on bytes/characters but increase compute requirements due to longer sequences.

### Evaluation & Cultural Shifts

- **Benchmarks:** Datasets like XTREME (classification/QA) and FLORES-101 (translation) measure performance.
- **Dataset Biases:**
  - **Translationese:** Translated evaluation data creates artificial syntax patterns and cultural mismatches.
  - **Performance Gap:** Models perform significantly worse on low-resource languages compared to English.
    - Strong correlation between **unlabelled data size** and **performance**
  - **Machine Translation**: FLORES-101 shoes translation performance is higher within the same language family (i.e - romatic, latin, etc). **Cross-family translation is much harder.**
- **Cultural Shift:** Knowledge transfer is not just linguistic; concepts, norms, and visuals vary by region.
  - _MaRVL Dataset:_ Shows how visual concepts (e.g., specific animals) differ drastically between cultures (e.g., Swahili vs. Indonesian), complicating transfer.
  - Cross-lingual VLM tranfer is **chance-level**

## W902 - Instruction tuning

### Introduction & Goal

1. **Why Instruction Tuning?**
   - **Problem**: A pre-trained LLM is just a next-token predictor.
     - It continues text but does not inherently understand tasks or follow instructions
   - **Consequence:** Outputs can be unfocused, unsafe, or misaligned with user intent.
   - **Goal of Instruction Tuning:** Transform the model into something that understands tasks, responds helpfully, follows instructions, and aligns with human preferences.

### Language Modelling vs. Instruction Following

- **The Distinction:** Language modelling is simply completing text; instruction following is solving a specific task defined by a prompt.
- **Why pre-training ≠ instruction following**
- **Example - Serendipity:**
  - A prompt like _"Define serendipity and use it in a sentence"_ is interpreted as text continuation, not a task specification.
  - _Base Model:_ May just continue the definition (e.g., "Serendipity is the ability to see...").
  - _Instruction-Tuned Model:_ **Generates a relevant sentence** (e.g., "Running into Margaret... was a fortunate stroke of serendipity").
- **Alignment Goals:** Adaptation is required to make models safe and robust. Key objectives include:
  - Following natural language instructions.
  - Avoiding harmful behaviors.
  - Responding to human preferences.
  - Instruction following requires **task grounding**, **reasoning**, and **alignment**, which pre-training alone does not provide.

### Adaptation Strategies

- **A) Task-Specific Fine-tuning**
  - Training a pre-trained model on a specific dataset (Task A) to create a specialised model.
  - Performs well on that task, but needs a model per task.
- **B) Prompting:**
  - Using a general-purpose pre-trained model and engineering the prompt to solve tasks.
  - Success relies on the base model's quality and prompts can be brittle.
- **C) Instruction Tuning (Multitask):**
  - Fine-tuning a pre-trained model on **many different tasks** (B, C, D) specified via natural language.
  - **Benefit:** The model learns to generalise and can perform inference on **unseen tasks** (Task A) because it understands the instruction format.
  - Requires a lot of curated data.

### What is Instruction Tuning

- **Unified Text-to-Text:**
  - Based on the T5 framework ("Unified Text-to-Text Transformer").
  - **Concept:** Every task (translation, summarisation, QA) is formulated as text input and text output.
  - Provide the model with thousands of **(instruction -> response)** examples across diverse tasks
  - Trains the model to **produce the response**, not merely continue the input.
- **Effect** The model learns:
  - To detect _what task_ is being asked.
  - To follow _instruction formats_.
  - To answer _appropriately_ even on unseen tasks.

### Formatting for Decoder-Only Models:

- Inputs and outputs are concatenated using **Prompt Templates**.
- **Special Tokens:** Uses tokens like `<|user|>` for the query and `<|assistant|>` for the response to structure the interaction.
- **Loss Calculation:** The model is trained using cross-entropy loss specifically on the response tokens, not the users prompt.

### Instruction Tuning Datasets

- **Natural Instructions (NI):**
  - Dataset with 61 tasks, 193k examples.
  - Includes definitions, positive/negative examples, and explanations.
  - Tests cross-task generalisation: train on some tasks -> evaluate on unseen ones.
- **Super-Natural Instructions (SNI):**
  - Scaled to 1,600+ tasks, 3 million+ examples, 575 languages.
  - **Scaling Findings:**
    - Performance grows linearly with the number of **tasks** and **model parameters**.
    - **Diversity over Volume:** Increasing examples _per task_ has diminishing returns; task diversity is the critical factor.

### Chain-of-Thought (CoT):

- CoT encourages models to **reason step-by-step**, improving performance on multi-step tasks like math or logical reasoning.
- Works both:
  - **without examples**, if prompted (“reason step-by-step”), and
  - **with exemplars**, when the dataset includes reasoning traces.

### Modern Instruction-Tuning Pipeline

- **Phases**
  1.  **Base LLM** (pre-trained on internet-scale text)
  2.  **Supervised Fine-Tuning (SFT)** on instruction datasets
  3.  Optional: **RLHF**, **safety training**, **specialised mixtures**
- **Data Curation:**
  - **Data Mixing:** Combining datasets to single high-quality "instruction mixture" for target skills (math, code, safety).
  - **Data Generation:** Creating new synthetic / curated data where evaluations show weaknesses to address failures.
  - **Contamination:** Must ensure training data does not overlap with test data.
- **Self-Instruct (Synthetic Data):**
  - **Method:** Using a strong "Teacher" model (e.g., GPT-4) to generate data for a "Student" model.
  - **Process:** Starts with a small set of "seed tasks" (e.g., 175) and asks the teacher to generate diverse new tasks.
  - **Personas:** Teachers can adopt personas (e.g., a scientist) to increase data diversity.

### Early Open Instruction-Tuned Models

These models demonstrated the viability of small, open instruction-tuned models.

- **Alpaca (2023):** 52k synthetic tasks from text-davinci-003.
- **Vicuna:** Fine-tuned on ShareGPT conversations; introduced LLM-as-judge.
- **Koala:** Combined multiple datasets including Anthropic HH.
- **Dolly:** 15k human-written instructions.

### Evaluation

- Evaluation guides the data mixing and generation process.
- **Key Benchmarks:**
  - **Knowledge:** MMLU.
  - **Reasoning:** GSM8K (Math), BigBenchHard.
  - **Coding:** HumanEval.
  - **Safety:** Toxicity detection, jailbreak resistance.
- **Outcome:** Developers balance specialised models (e.g., code-only) vs. general-purpose chat assistants.

## W903 - RLHF

### Why Post-Training? (Context)

- **Raw pre-trained LLMs are misaligned**
  - They simply optimise **next-token prediction**, not human values.
  - Show persistent failure cases: biased completions, unsafe instructions, incorrect answers.
- **Instruction Fine-Tuning (SFT) helps but is insufficient**
  - SFT uses (prompt, response) pairs only → **no negative examples**, no notion of "better vs. worse".
  - SFT can embed helpful behaviours but cannot encode **preferences** or resolve ambiguity between safe vs unsafe options.
- **Alignment Goal — The “Triple H”**:
  - **Helpful** — follows instructions usefully.
  - **Honest** — accurate, acknowledges uncertainty.
  - **Harmless** — avoids toxic or dangerous behaviour.
  - **Key Issue:** Pre-trained LLMs do not naturally satisfy these.

### Why Reinforcement Learning?

- **SFT ≠ Preference Learning**
  - SFT only learns to imitate provided responses.
  - To choose between _multiple possible_ valid outputs, we need a way to:
    - **measure which outputs humans prefer → optimise the model accordingly**.

### Reinforcement Learning (RL) Basics

- **Core Concept:** An agent interacts with an environment to maximise a cumulative reward.
- **Components:**
  - **State ($s_t$):** The agent's observation of the world at time $t$.
  - **Policy ($\pi$):** The agent's behavior function; it determines which action ($a_t$) to take based on the state.
  - **Action** ($a_t$): The action taken by the agent.
  - **Reward ($r_t$):** A scalar feedback signal indicating if an action was good or bad.
  - **Update Loop:** The agent updates its policy iteratively to maximise future rewards.

### Adapting RL for LLMs

- **Mapping RL to NLP:**
  - **State:** The context/prompt history up to the current step (e.g., "What is the capital of France").
  - **Policy ($\pi_\theta$):** The Language Model itself, which predicts the next token.
  - **Action:** The specific tokens generated by the LLM.
  - **Reward:** A score derived from preference data indicating how good the generated sequence is.
- **Distinction:** Unlike Supervised Learning (which only copies correct answers), RL uses rewards to distinguish between correct and incorrect behaviours.

### The Reward Model (Modelling Preferences)

- **Objective:** Instead of asking humans to rate every output (which is expensive), train a model to approximate human preferences.
- **Collect Human Rankings:** For each prompt $p$, humans choose which output is better:
  - **Dataset**: $\mathcal{D} = \{(p, o^+, o^-)\}$
- **Bradley-Terry Model:** A mathematical framework used to convert preference pairs into a reward function.
  - It calculates the probability that Output $O^+$ is preferred over $O^-$.
  - The loss function minimises the difference between the predicted reward for the winner and the loser.
- **Implementation:** The Reward Model is typically a fine-tuned copy of the LLM (with a scalar head) trained as a binary classifier.
  - Score near 0: Model incorrectly preferred the rejected response.
  - Score near 1: Model correctly preferred the chosen response.
  - Reward model predicts: $P(o^+ \succ o^-|p) = \sigma(R(o^+;p) - R(o^-;p))$
    - Training objective: increase reward of preferred responses.

### RLHF Optimisation & Reward Hacking

- **The Optimisation Goal:** Train the policy ($\pi_{RL}$) to maximise the expected reward.
- **RL alone risks:**
  - **Reward hacking** (gaming the reward model without improving quality).
  - **Catastrophic drift** (forgetting language ability).
- **The Fix (KL Penalty):** A penalty term is added to the objective function.
  - It constrains the new model ($\pi_{RL}$) so it does not diverge too far from the reference model ($\pi_{SFT}$).
  - Optimisation objective: $\arg\max_{\pi_{\text{RL}}} \mathbb{E}_{p,o\sim\pi_{\text{RL}}} \Big[ R(p,o) - \beta\,\mathrm{KL}(\pi_{\text{RL}}\parallel \pi_{\text{SFT}}) \Big]$
  - KL keeps the RL-updated policy close to the SFT model.
  - $\beta$ controls exploration vs stability.
- **PPO (Proximal Policy Optimisation):** The standard algorithm used to solve this optimisation is **PPO** which clips large policy updates, preventing collapse.

### The InstructGPT Pipeline (Summary)

1. **Phase 1:** Collect human feedback (rankings of summaries/responses).
2. **Phase 2:** Train a Reward Model to predict these human preferences.
3. **Phase 3:** Optimise the LLM against the Reward Model using PPO.

   - This involves generating outputs, scoring them with the Reward Model, and updating the Policy.
   - _Cost:_ This is **computationally expensive** as it requires loading **multiple models (Policy, Reference, Reward) simultaneously**.

### RLVR - Reinforcement Learning with Verifiable Rewards

- **Motivation:** Standard RLHF is complex and costly;
  - Requires another full LLM as a Reward Model
  - PPO needs **Policy + Reference + Reward Model** in memory
  - RLVR simplifies this for tasks with objective answers.
- **Idea: Replace the reward model with a \***programmatic reward function.\*\*\*
  - **Math:** $2+2=4$. If the model outputs 4, Reward = 1; else 0.
  - **Code:** Does the code compile? Do tests pass?.
  - **Structural** **Constraints:** "Every nth word must be French".
- **Benefits:**
  - Requires fewer models in memory (only the Policy and Reference)
  - Cheap and robust when correctness is objectively verifiable
- **Failure Modes:**
  - **Reasoning Errors:** A model might get the correct final answer (high reward) via incorrect intermediate steps.
  - **Reward Hacking**: Exploit quirks in the verifier
  - **Spurious Rewards:** Models might learn from random or noisy signals if the verification isn't robust.