#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

TRAIN_DIR=train_dir/

rm -rf ${TRAIN_DIR}
mkdir ${TRAIN_DIR}
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../../
python models/research/object_detection/train.py --logtostderr --train_dir=${TRAIN_DIR} --pipeline_config_path=${1}
