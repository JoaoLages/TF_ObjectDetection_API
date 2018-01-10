# Run previsously:
# cd models/research
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# cd ../../

TRAIN=train_gg_easy
TEST=test_gg_easy
MODEL_CONFIG=data/ssd_mobilenet_v1_shapes.config

# Create TF records
bash create_TFrecord.sh train ${TRAIN}
bash create_TFrecord.sh test ${TEST}

# Train model
bash train_model.sh ${MODEL_CONFIG}

# Test model on epoch 4000
bash test_model.sh ${TEST} ${MODEL_CONFIG} 4000

# Backup
mkdir ${TRAIN}
mkdir -p ${TRAIN}/${TEST}/
cp -r data/ ${TRAIN}/${TEST}/
cp -r train_dir/ ${TRAIN}/${TEST}/

