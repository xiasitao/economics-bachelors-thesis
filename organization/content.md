# Introduction
## Socioeconomic status
- literature
- Classic predictors
- Role models as a new predictor

## Role model review
- role model literature
- socioeconomic status literature
- NLP in economics literature

## NLP
- NLP
- NLP in economics


# Data
Describing what data is about, where it comes from, how it was processed, cleaned, and filtered.
- SES data
    - origin
    - descriptives
    - filtering
        - filtering out any role models where SES is ambiguous
        - descriptives
    - balancing
        - balancing the amount of role models for low and high SES
        - descriptives
        - only necessary if articles/role models are counted per SES level
- articles
    - origin
    - descriptives
    - cleaning, processing
    - balancing (articles per role model)
        - method
        - descriptives (how many role models had too few articles, how many articles were upsampled?)
        - necessary if articles/role models are counted per SES level in order to be able to compare both SES categories
        - also necessary if distribution of topics etc. is scrutinized because topics etc. strongly correlate with role models and topics could be unnaturally overweighted/underweighted just by the number of articles per role model


# Topic Modelling
- Theory
- Data preparation for topic modelling
    - filtering out one-letter words
    - filtering out words not being a noun, proper noun, or verb
- Hyperparameter tuning
    - iteratively examining number of topics
    - number of iterations
    - convergence graph
- Evaluation
    - no proper number of topics found
        - inconsistent
        - few topics: very broad, too much variation in topic assignment
        - many topics: topics very fine grained, too much influenced by singular articles
    - accuracy not consistent (human annotation)
    - model too elastic in the input data
        - topic words contain many names and specifics
        - topics are too vague, contain more than one topic
        - topics don't confer unambiguous sentiment

# Zero-Shot Classification
- BERT and SentenceBERT theory
- BART
- skipping the training step in lack of labels
- categories and their labels
- evaluation
- fundamental advantages:
    - no overfitting
    - much more stable depending on the inputs, more robust model basis


# Semantic Similarity-based Clustering
- clustering theory