import Tkinter as tk
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

import tkFileDialog
from Tkinter import *
from PIL import Image, ImageTk, ImageSequence
from collections import defaultdict
from io import StringIO

sys.path.append("..")
from utils import ops as utils_ops

if tf.__version__ < '1.04.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
# This is needed to display the images.

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# download
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

# load the frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


window = tk.Tk()
window.title('object recgonition')
window.geometry('800x600')

varAddress = tk.StringVar()

l = tk.Label(window, textvariable=varAddress, bg='green', width=100, height=5)
l.pack()
l.place(x=0, y=60, width=800, height=50)

panelUpload = tk.Label(window)
panelUpload.pack()
panelUpload.place(x=10, y=120, width=380, height=380)


def uploadPicFunction():
    global varAddress
    varAddress.set(tkFileDialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"),("all files", "*.*"))))
    imgUploadRaw = Image.open(varAddress.get())
    imgUploadResized = imgUploadRaw.resize((380, 380), Image.ANTIALIAS)
    imgUpload = ImageTk.PhotoImage(imgUploadResized)
    panelUpload.configure(image=imgUpload)
    panelUpload.image = imgUpload


uploadButton = tk.Button(window, text='upload pic', height=5, command=lambda: uploadPicFunction())
uploadButton.pack()
uploadButton.place(x=100, y=10, height=40, width=100)

globalFlag = FALSE

def animate(counter, images, panel):
    global globalFlag
    counter2 = counter
    if globalFlag:
        counter2 = 0
        globalFlag = FALSE
    panel.configure(image=images[counter2])
    panel.image = images[counter2]
    window.after(66, lambda: animate((counter2 + 1) % len(images), images, panel))


def uploadGifFunction():
    global frame
    varAddress.set(tkFileDialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("gif files", "*.gif"), ("all files", "*.*"))))
    imagesRaw = [ImageTk.PhotoImage(img)
                 for img in ImageSequence.Iterator(Image.open(varAddress.get()))]
    animate(0, imagesRaw, panelUpload)


uploadGifButton = tk.Button(window, text='upload gif', height=5, command=lambda: uploadGifFunction())
uploadGifButton.pack()
uploadGifButton.place(x=200, y=10, height=40, width=100)

panelResult = tk.Label(window)
panelResult.pack()
panelResult.place(x=420, y=120, width=380, height=380)

varTime = tk.StringVar()

def objectRecgonition():
    global varTime
    t0 = time.time()
    imgUploadRaw = Image.open(varAddress.get())
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # imgUploadRaw = imgUploadRaw.resize((380, 380), Image.ANTIALIAS) //The reason result was bad
    image_np = load_image_into_numpy_array(imgUploadRaw)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)
    imageFromArray = Image.fromarray(image_np, 'RGB')
    varTime.set(time.time() - t0)
    print(varTime)
    imageResized = imageFromArray.resize((380, 380), Image.ANTIALIAS)
    imageDone = ImageTk.PhotoImage(imageResized)
    panelResult.configure(image=imageDone)
    panelResult.image = imageDone


recButton = tk.Button(window, text="JPG object recgonition", height=5, command=lambda: objectRecgonition())
recButton.pack()
recButton.place(x=420, y=10, height=40, width=160)


def objectRecGif():
    gifTemp = []
    t0 = time.time()
    for imgRaw in ImageSequence.Iterator(Image.open(varAddress.get())):
        img24 = imgRaw.convert('RGB')
        image_np = load_image_into_numpy_array(img24)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
        imageFromArray = Image.fromarray(image_np, 'RGB')
        gifTemp.append(ImageTk.PhotoImage(imageFromArray))
    varTime.set(time.time() - t0)
    globalFlag = TRUE
    animate(0, gifTemp, panelResult)


recGifButton = tk.Button(window, text="GIF object recognition", height=5, command=lambda: objectRecGif())
recGifButton.pack()
recGifButton.place(x=580, y=10, height=40, width=160)


timeCost = tk.Label(window, textvariable=varTime, bg='red', width=20, height=3)
timeCost.pack(side=BOTTOM)


window.mainloop()
