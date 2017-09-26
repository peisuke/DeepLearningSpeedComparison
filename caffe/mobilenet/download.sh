#!/bin/bash
FILE_ID=0B7ubpZO7HnlCVFFJQU5TQ0dkLUE
FILE_NAME=mobilenet.caffemodel
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
