#! /bin/bash
aws s3 sync "/home/maxi/Programmieren/vwl/econ_bachelors_thesis/" "s3://econ-bachelors-thesis/" --exclude '*' \
    --include "build/articles_balanced_50.pkl" \
    --include "build/zero_shot_classification.pkl" \
    --include "code/tasks/task_zero_shot.py"
