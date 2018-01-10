EPOCH=360
TEST=test_gg_easy
MODEL_CONFIG=data/ssd_mobilenet_v1_shapes.config

# Test model on epoch $EPOCH
bash test_model.sh ${TEST} ${MODEL_CONFIG} ${EPOCH}

