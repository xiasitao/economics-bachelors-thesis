#! /bin/bash

TARGETPATH_xray="/home/maxi/Programmieren/vwl/econ_bachelors_thesis/"
TARGETPATH_yankee="/home/maxi/Programming/econ_bachelors_thesis/"
TARGETPATH="."
if [ -f "$TARGETPATH_xray" ]
then
	TARGETPATH="$PATH_xray"
fi
if [ -f "$TARGETPATH_yankee" ]
then
        TARGETPATH="$PATH_yankee"
fi

aws s3 sync "$TARGETPATH" "s3://econ-bachelors-thesis/" --exclude '*' \
    --include "build/articles/articles.pkl" \
    --include "build/articles/articles_balanced_50.pkl" \
    --include "build/articles/articles_human_annotated.pkl" \
    --include "build/zero_shot_classification/*" \
    --include "build/topic_modelling/*"
