from mvnc import mvncapi as mvnc
import numpy
import cv2
import os
import sys
from typing import List

class mnist:
    def __init__(self):
        self.device, self.graph = do_initialize()

    def do_initialize() -> (self, mvnc.Device, mvnc.Graph):
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
                print('Error - No devices found')
                return (None, None)
        device = mvnc.Device(devices[0])
        device.OpenDevice()
        graph_filename = '/home/ncsdk/Desktop/workspace/ncappzoo/tensorflow/mnist/mnist_inference.graph'
        # Load graph file
        try :
            with open(graph_filename, mode='rb') as f:
                in_memory_graph = f.read()
        except :
            print ("Error reading graph file: " + graph_filename)
        graph = device.AllocateGraph(in_memory_graph)
        return device, graph


    def do_inference(self, graph: mvnc.Graph, img, number_results : int = 5) -> (List[str], List[numpy.float16]) :
        labels=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        image_for_inference = img
        image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)
        image_for_inference = cv2.resize(image_for_inference, NETWORK_IMAGE_DIMENSIONS)
        ret,image_for_inference = cv2.threshold(image_for_inference,120,255,cv2.THRESH_BINARY_INV)
        cv2.imshow("image_for_inference:", image_for_inference)
        image_for_inference = image_for_inference.astype(numpy.float32)
        image_for_inference[:] = ((image_for_inference[:] )*(1.0/255.0))

        # Start the inference by sending to the device/graph
        graph.LoadTensor(image_for_inference.astype(numpy.float16), None)

        # Get the result from the device/graph.  userobj should be the
        # same value that was passed in LoadTensor above.
        output, userobj = graph.GetResult()

        # sort indices in order of highest probabilities
        five_highest_indices = (-output).argsort()[:number_results]

        # get the labels and probabilities for the top results from the inference
        inference_labels = []
        inference_probabilities = []

        for index in range(0, number_results):
            inference_probabilities.append(str(output[five_highest_indices[index]]))
            inference_labels.append(labels[five_highest_indices[index]])

        return inference_labels, inference_probabilities


    def do_cleanup(self, device: mvnc.Device, graph: mvnc.Graph) -> None:

        graph.DeallocateGraph()
        device.CloseDevice()


    def show_inference_results(self, infer_labels: List[str],
                            infer_probabilities: List[numpy.float16]) -> None:

        num_results = len(infer_labels)
        for index in range(0, num_results):
            one_prediction = '  certainty ' + str(infer_probabilities[index]) + ' --> ' + "'" + infer_labels[index]+ "'"
            #print(one_prediction)

        #print('-----------------------------------------------------------')
        if float(infer_probabilities[0])>0.85000:
            return [infer_probabilities[0], infer_labels[0]]
        return [0, 0]

    def run_mnist(self, img):
        img = cv2.resize(img, (224,224))
        if (device == None or graph == None):
            print ("Could not initialize device.")
            quit(1)
        # lopt through all the input images and run inferences and show results
        infer_labels, infer_probabilities = do_inference(graph, img, 5)
        out = show_inference_results(infer_labels, infer_probabilities)
        output_str = str('pridict:' + str(out[1]) + '; confidence:' + str(100*out[0]))
        cv2.putText(img, output_str, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        return img

    def destroy_mnist(self):
        self.do_cleanup()
        return 0


