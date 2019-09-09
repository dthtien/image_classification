# Image Processing
- [Knowledge](https://paper.dropbox.com/doc/Image-classification-and-Object-detection--AkcEjwoOZPs8f3hYrZepKDRJAQ-RXYHSy4sAsAFb2o4CZo0k)
- Installation:
```sh
git clone git@github.com:dthtien/image_classification.git

cd image_classification

pip install flask

pip install tensorflow # tensorflow-gpu

pip install keras
```
- Classification
  ```
  python3 main.py
  # open localhost:5000
  ```

- Object detection
  Download [model](https://drive.google.com/file/d/1S92D3y4dfeS_FmpSqPtK-Qn6OvnkPcNz/view?usp=sharing) here and remember change `weight_path` and `image_path` at `detector.py`

  ```
  python3 detector.py
  ```
- Training
  Dowload [training set](https://drive.google.com/drive/folders/1r4dteg8-VV93s2vHC7hPZga-bT7E2wsZ?usp=sharing) and modify `training.py` with downloaded data, modify `train.txt`
  ```
  python3 training.py
  ```
- Current objects
```
Daisy
Dandelion
Flatrack container
General container
Roses
Sunflowers
Tulips
```

- References
  - [Tensorflow](https://www.tensorflow.org/)
  - [An Introduction to Machine Learning by Matthew Mongeau](https://www.youtube.com/watch?v=8G709hKkthY)
  - [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)
  - [Demystifying Deep Neural Nets – Rosie Campbell](https://www.youtube.com/watch?v=S4vL355capU)
  - [Faster R-CNN Object Detection with PyTorch](https://www.learnopencv.com/faster-r-cnn-object-detection-with-pytorch/)
  - [Keras-YOLO3](https://github.com/qqwweee/keras-yolo3)