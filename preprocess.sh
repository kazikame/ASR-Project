#!/usr/bin/env bash

mkdir wav 2> /dev/null
for f in *.m4a; do ffmpeg -loglevel panic -ss 10 -t 10 -i $f "wav/${f/%m4a/wav}"; done
cd wav
for f in *.wav; do sox --norm $f "${f/%.wav/_mono.wav}" remix 1-2 && rm ${f}; done