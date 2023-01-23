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

aws s3 sync "s3://econ-bachelors-thesis/" "$TARGETPATH"
