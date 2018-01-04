#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

DATA=data/
RESIZE=/home/vislab/Colabs-2017-2018/object_detection/images/${2}_resize/
CSV=/home/vislab/Colabs-2017-2018/object_detection/${2}.csv

cd ${DATA}/
rm -f ${1}.record 
cd ../
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cd ../../
python generate_tfrecord.py --data_path=${DATA} --images_path=${RESIZE} --csv_path=${CSV} --file_name=${1}


