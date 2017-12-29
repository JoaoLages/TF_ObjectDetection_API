DATA=data/
RESIZE=/home/vislab/Colabs-2017-2018/object_detection/images/train_gg_easy_resize/
CSV=/home/vislab/Colabs-2017-2018/object_detection/train_gg_easy.csv

cd ${DATA}/
rm train.record test.record
cd ../
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ../../
python generate_tfrecord.py --data_path=${DATA} --images_path=${RESIZE} --csv_path=${CSV}


