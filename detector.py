import os
import sys
import logging

from PIL import Image
import pandas as pd
from yolo3.yolo import YOLO

sys.path += [os.path.abspath('.'), os.path.abspath('..')]

VISUAL_EXT = '_detect'
MODEL_IMAGE_SIZE = (416, 416)

def predict_image(yolo, path_image, path_output=None):
  if not path_image:
    logging.debug('no image given')
  elif not os.path.isfile(path_image):
    logging.warning('missing image: %s', path_image)

  image = Image.open(path_image)
  image_pred, pred_items = yolo.detect_image(image)
  if path_output is None or not os.path.isdir(path_output):
    image_pred.show()
  else:
    name = os.path.splitext(os.path.basename(path_image))[0]
    path_out_img = os.path.join(path_output, name + VISUAL_EXT + '.jpg')
    path_out_csv = os.path.join(path_output, name + '.csv')
    logging.info('exporting image: "%s" and detection: "%s"',
                  path_out_img, path_out_csv)
    image_pred.save(path_out_img)
    pd.DataFrame(pred_items).to_csv(path_out_csv)


def _main(path_image = 'c1.jpeg'):
  path_weights = 'model_data/final_trained.h5'
  path_anchors = 'model_data/yolo_anchors.csv'
  path_classes = 'train_classes.txt'
  path_output = '.'
  nb_gpu=0
  yolo = YOLO(weights_path=path_weights, anchors_path=path_anchors,
              classes_path=path_classes, model_image_size=MODEL_IMAGE_SIZE,
              nb_gpu=nb_gpu)

  logging.info('Start image processing..')
  predict_image(yolo, path_image, path_output)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  _main()
  logging.info('Done')
