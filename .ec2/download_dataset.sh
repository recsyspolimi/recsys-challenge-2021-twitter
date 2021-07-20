#!/bin/bash

#change the link or ask me (Alessandro Sanvito) to generate a new one for you if 72h have passed
#use this (https://unix.stackexchange.com/questions/223734/how-to-download-files-and-folders-from-onedrive-using-wget) link for reference
#download_link="https://onedrive.live.com/download?cid=FBC297C464FD01F3&resid=FBC297C464FD01F3%212602&authkey=ALrHA6LP-97fcvI"

cd /home/ubuntu/data

wget --no-check-certificate -O training_urls.txt ${download_link}

echo "List of links downloaded"

sudo apt-get update -y

#install aria2 download manager
sudo apt install aria2

#download the dataset
#-j specifies the number of jobs
sudo aria2c -itraining_urls.txt -j10

echo "Dataset downloaded"

#install lzop to decompress the dataset
sudo apt-get install -y lzop

#decompress every compressed file
lzop --verbose --delete -d *.lzo

echo "Dataset decompressed"

rm *.lzo.index

echo "The dataset is ready!"
