# Cleaning


# Filtering
- Filter out stubs (e.g. under three sentences) and epics (e.g. more than 200 sentences)
- Filter out articles with wrong content (e.g. cookie banner)

# Descriptives
- Describe all data at every step
    - How many rows
    - Grouped by language, role models, SES: how many data rows?
- Sanity checks!

# SES
- SES data is very imbalanced
    - maybe select equal amounts of positive and negative SES samples with the role models with most articles
- most role models only mentioned once
    - filter out role models that were only mentioned once, 58 would remain
- some role models are ambiguous
    - filter out role models without SES majority
    - filter out role models only mentioned by a single SES group (problem: high SES role models might be very few then)
- When filtering role models, check that enough articles are present

# Topic model
- fix hyperparameters
- add to pipeline
- chi2 contingency test

# Zero Shot
- chi2 contingency test

# Semantic clustering
- combine with topic model