
from utils import label_map_util
from utils import visualization_utils as vis_util
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--test-images')
args = parser.parse_args(sys.argv[1:])

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'object_detection_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/label_map.pbtxt'

NUM_CLASSES = 3

PATH_TO_TEST_IMAGES_DIR = args.test_images
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(6, 12) ]
IMAGE_SIZE = (12, 12)

detection_graph = tf.Graph()
with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')

                        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
                        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
                        category_index = label_map_util.create_category_index(categories)

                        def load_image_into_numpy_array(image):
                              (im_width, im_height) = image.size
                                return np.array(image.getdata()).reshape(
                                              (im_height, im_width, 3)).astype(np.uint8)



