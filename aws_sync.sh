#! /bin/bash
aws s3 sync "/home/maxi/Programmieren/vwl/econ_bachelors_thesis/" "s3://econ-bachelors-thesis/" --exclude '*' \
    --include "build/articles/articles.pkl" \
    --include "build/articles/articles_balanced_50.pkl" \
    --include "build/articles/articles_human_annotated.pkl" \
    --include "build/zero_shot_classification/*" \
    --include "build/topic_modelling/*"
