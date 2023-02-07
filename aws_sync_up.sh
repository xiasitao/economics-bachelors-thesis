#! /bin/bash

TARGETPATH_xray="/home/maxi/Programmieren/vwl/econ_bachelors_thesis/"
TARGETPATH_yankee="/home/maxi/Programming/econ_bachelors_thesis/"
TARGETPATH="."
if [ -d "$TARGETPATH_xray" ]
then
	TARGETPATH="$TARGETPATH_xray"
        echo "On xray" >&2
fi
if [ -d "$TARGETPATH_yankee" ]
then
        TARGETPATH="$TARGETPATH_yankee"
        echo "On yankee" >&2
fi

aws s3 sync "$TARGETPATH" "s3://econ-bachelors-thesis/" --exclude '*' \
    --include "build/articles/articles.pkl" \
    --include "build/articles/articles_balanced_50.pkl" \
    --include "build/articles/articles_human_annotated.pkl" \
    --include "build/articles/articles_human_annotated_distinct.pkl" \
    --include "build/role_models/*" \
    --include "build/zero_shot_classification/*" \
    --include "build/topic_modelling/*" \
    --include "build/semantic_similarity/*"
