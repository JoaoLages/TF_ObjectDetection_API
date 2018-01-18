# Run previsously:
# cd models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# cd ../../

TRAIN=real/images/easy_1
TEST=real/images/easy_2
MODEL_CONFIG=data/ssd_mobilenet_v1_shapes.config

# Create TF records
bash create_TFrecord.sh train ${TRAIN}
bash create_TFrecord.sh test ${TEST}

# Train model
bash train_model.sh ${MODEL_CONFIG}

# Test model on epoch 4000
bash test_model.sh ${TEST} ${MODEL_CONFIG} 4000

# Backup
mkdir -p ${TEST}/${TRAIN}/${MODEL_CONFIG}
cp -r data/ ${TEST}/${TRAIN}/${MODEL_CONFIG}
cp -r train_dir/ ${TEST}/${TRAIN}/${MODEL_CONFIG}
cp -r inference_results/ ${TEST}/${TRAIN}/${MODEL_CONFIG}

