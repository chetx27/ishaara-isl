# ISL Grammar Rules for Synthetic Data Generation

Due to the lack of a large-scale ISL Gloss to English parallel corpus, we construct synthetic training pairs for the seq2seq translation models (Phases 5 and 6) using linguistically-informed transformations.

## Core Linguistic Rules (ISL)
Based on Indian Sign Language grammar (e.g., Vasishta, Woodward, & De Santis, 1980):

1. **Word Order (SOV)**: ISL typically follows Subject-Object-Verb, unlike English's SVO.
   - *English*: "I am eating an apple."
   - *ISL Gloss*: "I APPLE EAT"

2. **Topic-Comment Structure**: ISL often places the topic first, followed by a comment or question.
   - *English*: "What is your name?"
   - *ISL Gloss*: "YOUR NAME WHAT"

3. **Time Placement**: Time markers generally appear at the beginning of the sentence.
   - *English*: "I will go to the market tomorrow."
   - *ISL Gloss*: "TOMORROW I MARKET GO"

4. **Omission of Articles and Copulas**: ISL does not use "a", "an", "the", or "to be" verbs (is, am, are).
   - *English*: "The boy is running."
   - *ISL Gloss*: "BOY RUN"

5. **Negation**: Negation markers typically come at the end of the clause.
   - *English*: "I do not know."
   - *ISL Gloss*: "I KNOW NOT"

## Synthetic Dataset Generation Strategy
For Phase 5 & 6, we apply a rule-based tokenizer and reorderer to a standard English corpus to create proxy `(English, ISL-Gloss)` pairs to train `mBART-small`/`T5-small`.

> **Limitation Statement**: This is a linguistically-informed synthetic approach, not real-corpus training. Real ISL incorporates rich spatial grammar, classifiers, and non-manual markers that text-based glossing inherently fails to capture.
