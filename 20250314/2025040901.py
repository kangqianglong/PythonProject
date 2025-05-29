import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
print(sys.executable+"ol67")
import tensorflow as tf
print(tf.__version__)
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))