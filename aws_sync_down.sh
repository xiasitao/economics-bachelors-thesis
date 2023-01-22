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

aws s3 sync "s3://econ-bachelors-thesis/" "$TARGETPATH"
