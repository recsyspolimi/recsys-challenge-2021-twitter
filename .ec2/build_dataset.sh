#!/bin/bash

sudo cat part-* > train.tsv && rm part-* -f

echo "Train.tsv available!"