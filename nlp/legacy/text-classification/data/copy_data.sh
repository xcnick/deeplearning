#!/bin/bash

# copy MAXCOUNT files from each directory

MAXCOUNT=6500

for category in $( ls /mnt/data/THUCNews); do
  echo item: $category

  dir=/mnt/data/THUCNews/$category
  newdir=/home/orient/chi/data/THUCNews/$category
  if [ ! -d $newdir ]; then
    rm -rf $newdir
    mkdir -p $newdir
  fi

  COUNTER=1
  for i in $(ls $dir); do
    cp $dir/$i $newdir
    if [ $COUNTER -ge $MAXCOUNT ]
    then
      echo finished
      break
    fi
    let COUNTER=COUNTER+1
  done

done