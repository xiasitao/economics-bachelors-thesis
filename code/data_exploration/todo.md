## Cleaning
- Unify sentence boundaries ('.')


## Filtering
- Filter out stubs (e.g. under three sentences) and epics (e.g. more than 200 sentences)
- Filter out articles with wrong content (e.g. cookie banner)

## SES
- SES data is very imbalanced
    - maybe select equal amounts of positive and negative SES samples with the role models with most articles
- most role models only mentioned once
    - filter out role models that were only mentioned once, 58 would remain
- some role models are ambiguous
    - filter out role models without SES majority
    - filter out role models only mentioned by a single SES group (problem: high SES role models might be very few then)