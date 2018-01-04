set -o errexit
set -o nounset
set -o pipefail

PIPELINE_CONFIG_FILE=${2}
TRAIN_DIR=train_dir
IMAGES_TEST_PATH=/home/vislab/Colabs-2017-2018/object_detection/images/${1}_resize

rm -rf object_detection_graph
python models/research/object_detection/export_inference_graph.py \
	    --input_type image_tensor \
		    --pipeline_config_path ${PIPELINE_CONFIG_FILE}\
			    --trained_checkpoint_prefix ${TRAIN_DIR}/model.ckpt-${3} \
				    --output_directory object_detection_graph

python -m object_detection/inference/infer_detections \
	--input_tfrecord_paths=data/test.record \
	--output_tfrecord_path=inference_results/detections.tfrecord-00000-of-00001 \
	--inference_graph=object_detection_graph/frozen_inference_graph.pb \
	--discard_image_pixels

python -m object_detection/metrics/offline_eval_map_corloc \
	--eval_dir=inference_results \
	--eval_config_path=test_eval_config.pbtxt \
	--input_config_path=test_input_config.pbtxt

mkdir -f inference_results/tests_overlayed
python overlay.py --images_path=${IMAGES_TEST_PATH} --save_path=inference_results/tests_overlayed
