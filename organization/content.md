# Role model review
- role model literature
- socioeconomic status literature
- NLP in economics literature

# Data
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

# Semantic similarity-based clustering
- BERT and SentenceBERT theory
- clustering theory

# Zero-shot classification