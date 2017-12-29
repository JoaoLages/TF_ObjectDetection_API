TRAIN_DIR=train_dir/
MODEL_CONFIG=data/ssd_mobilenet_v1_shapes.config

rm -rf ${TRAIN_DIR}
mkdir ${TRAIN_DIR}
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../../
python models/research/object_detection/train.py --logtostderr --train_dir=${TRAIN_DIR} --pipeline_config_path=${MODEL_CONFIG}
