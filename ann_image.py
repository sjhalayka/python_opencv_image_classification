import cv2
import numpy as np
import math


def get_bits_for_int(src_min_bits, src_number):
    bits = bin(src_number)[2:]

    a = np.array([])

    for i in range(0, len(bits)):
        a = np.append(a, float(bits[i]))

    num_bits = len(a)
    needed_bits = 0

    if num_bits < src_min_bits:
        needed_bits = src_min_bits - num_bits

    for i in range(0, needed_bits):
        a = np.insert(a, 0, 0.0)

    return a


file = open("files.txt", "r") 

filenames = []
classifications = []

for line in file:
    filenames.append(line.split(" ")[0])
    classifications.append(int(line.split(" ")[1]))

max_class = 0

for i in range(0, len(classifications)):
    if classifications[i] > max_class:
        max_class = classifications[i]

num_classes = max_class + 1
num_bits_needed = math.floor(math.log(num_classes)/math.log(2.0))

sample_img = cv2.imread(filenames[0])

img_rows = sample_img.shape[0]
img_cols = sample_img.shape[1]
channels_per_pixel = 3

num_input_neurons = int(img_rows*img_cols*channels_per_pixel)
num_output_neurons = int(num_bits_needed)
num_hidden_neurons = int(math.floor(math.sqrt(num_input_neurons*num_output_neurons)))

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([num_input_neurons, num_hidden_neurons, num_output_neurons], dtype=np.int64))
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1, 0.000001 ))
ann.setBackpropMomentumScale(0.00001)
ann.setBackpropWeightScale(0.00001)

img_input_array = sample_img.flatten()
img_input_array = img_input_array.astype(np.float32)

for i in range(0, img_input_array.shape[0]):
    img_input_array[i] = float(img_input_array[i]) / float(255)

img_output_array = get_bits_for_int(num_output_neurons, classifications[0])
img_output_array = img_output_array.astype(np.float32)

img_input_array = img_input_array.reshape(1, img_input_array.shape[0])
img_output_array = img_output_array.reshape(1, img_output_array.shape[0])

img_td = cv2.ml.TrainData_create(img_input_array, cv2.ml.ROW_SAMPLE, img_output_array)
ann.train(img_td, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

for i in range(0, 100):

    print(i)

    for j in range(0, len(filenames)):
        img_input_array = cv2.imread(filenames[j])
        img_input_array = img_input_array.flatten()
        img_input_array = img_input_array.astype(np.float32)

        for k in range(0, img_input_array.shape[0]):
            img_input_array[k] = float(img_input_array[k]) / float(255)

        img_output_array = get_bits_for_int(num_output_neurons, classifications[j])
        img_output_array = img_output_array.astype(np.float32)

        img_input_array = img_input_array.reshape(1, img_input_array.shape[0])
        img_output_array = img_output_array.reshape(1, img_output_array.shape[0])
        
        img_td = cv2.ml.TrainData_create(img_input_array, cv2.ml.ROW_SAMPLE, img_output_array)
        ann.train(img_td, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

for i in range(0, len(filenames)):
    print(filenames[i])
    
    img_input_array = cv2.imread(filenames[i])
    img_input_array = img_input_array.flatten()
    img_input_array = img_input_array.astype(np.float32)

    for j in range(0, img_input_array.shape[0]):
        img_input_array[j] = float(img_input_array[j]) / float(255)

    img_input_array = img_input_array.reshape(1, img_input_array.shape[0])
    
    print(ann.predict(img_input_array))
