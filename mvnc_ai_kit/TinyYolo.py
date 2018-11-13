from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import time

class TinyYolo:
    def __init__(self):
        self.tinyyolo_status = False
        self.NETWORK_IMAGE_WIDTH = 448
        self.NETWORK_IMAGE_HEIGHT = 448
        print('-----------------------init tinyyolo-----------------------')
    
    def prepare(self):
        self.tinyyolo_status = True
        tiny_yolo_graph_file = './graph/tinyyolo_graph'
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No devices found')
            quit()
        self.device = mvnc.Device(devices[0])
        self.device.OpenDevice()
        with open(tiny_yolo_graph_file, mode='rb') as f:
            graph_from_disk = f.read()
        self.graph = self.device.AllocateGraph(graph_from_disk)

    def filter_objects(self, inference_result, input_image_width, input_image_height):

        # the raw number of floats returned from the inference (GetResult())
        num_inference_results = len(inference_result)

        # the 20 classes this network was trained on
        network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                                "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

        # only keep boxes with probabilities greater than this
        probability_threshold = 0.07

        num_classifications = len(network_classifications) # should be 20
        grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
        boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

        # grid_size is 7 (grid is 7x7)
        # num classifications is 20
        # boxes per grid cell is 2
        all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

        classification_probabilities = \
            np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
        num_of_class_probs = len(classification_probabilities)

        # The probability scale factor for each box
        box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

        # get the boxes from the results and adjust to be pixel units
        all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
        self.boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

        # adjust the probabilities with the scaling factor
        for box_index in range(boxes_per_grid_cell): # loop over boxes
            for class_index in range(num_classifications): # loop over classifications
                all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


        probability_threshold_mask = np.array(all_probabilities>=probability_threshold, dtype='bool')
        box_threshold_mask = np.nonzero(probability_threshold_mask)
        boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
        probabilities_above_threshold = all_probabilities[probability_threshold_mask]

        # sort the boxes from highest probability to lowest and then
        # sort the probabilities and classifications to match
        argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
        boxes_above_threshold = boxes_above_threshold[argsort]
        classifications_for_boxes_above = classifications_for_boxes_above[argsort]
        probabilities_above_threshold = probabilities_above_threshold[argsort]


        # get mask for boxes that seem to be the same object
        duplicate_box_mask = self.get_duplicate_box_mask(boxes_above_threshold)

        # update the boxes, probabilities and classifications removing duplicates.
        boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
        classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
        probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

        classes_boxes_and_probs = []
        for i in range(len(boxes_above_threshold)):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

        return classes_boxes_and_probs

    def get_duplicate_box_mask(self, box_list):
        max_iou = 0.35

        box_mask = np.ones(len(box_list))

        for i in range(len(box_list)):
            if box_mask[i] == 0: continue
            for j in range(i + 1, len(box_list)):
                if self.get_intersection_over_union(box_list[i], box_list[j]) > max_iou:
                    box_mask[j] = 0.0

        filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
        return filter_iou_mask

    def boxes_to_pixel_units(self, box_list, image_width, image_height, grid_size):

        # number of boxes per grid cell
        boxes_per_cell = 2

        # setup some offset values to map boxes to pixels
        # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
        box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

        # adjust the box center
        box_list[:,:,:,0] += box_offset
        box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
        box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

        # adjust the lengths and widths
        box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
        box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

        #scale the boxes to the image size in pixels
        box_list[:,:,:,0] *= image_width
        box_list[:,:,:,1] *= image_height
        box_list[:,:,:,2] *= image_width
        box_list[:,:,:,3] *= image_height

    def get_intersection_over_union(self, box_1, box_2):

        intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                            max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

        intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                            max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

        if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
            # no intersection area
            intersection_area = 0
        else :
            intersection_area =  intersection_dim_1*intersection_dim_2
        union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;
        # now we can return the intersection over union
        iou = intersection_area / union_area
        return iou

    def display_objects_in_gui(self, source_image, filtered_objects):
        # copy image so we can draw on it. Could just draw directly on source image if not concerned about that.
        display_image = source_image.copy()
        source_image_width = source_image.shape[1]
        source_image_height = source_image.shape[0]

        x_ratio = float(source_image_width) / self.NETWORK_IMAGE_WIDTH
        y_ratio = float(source_image_height) / self.NETWORK_IMAGE_HEIGHT

        # loop through each box and draw it on the image along with a classification label
        print('Found this many objects in the image: ' + str(len(filtered_objects)))
        for obj_index in range(len(filtered_objects)):
            center_x = int(filtered_objects[obj_index][1] * x_ratio) 
            center_y = int(filtered_objects[obj_index][2] * y_ratio)
            half_width = int(filtered_objects[obj_index][3] * x_ratio)//2
            half_height = int(filtered_objects[obj_index][4] * y_ratio)//2

            # calculate box (left, top) and (right, bottom) coordinates
            box_left = max(center_x - half_width, 0)
            box_top = max(center_y - half_height, 0)
            box_right = min(center_x + half_width, source_image_width)
            box_bottom = min(center_y + half_height, source_image_height)

            print('box at index ' + str(obj_index) + ' is... left: ' + str(box_left) + ', top: ' + str(box_top) + ', right: ' + str(box_right) + ', bottom: ' + str(box_bottom))  

            #draw the rectangle on the image.  This is hopefully around the object
            box_color = (0, 255, 0)  # green box
            box_thickness = 2
            cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

            # draw the classification label string just above and to the left of the rectangle
            label_background_color = (70, 120, 70) # greyish green background for text
            label_text_color = (255, 255, 255)   # white text
            cv2.rectangle(display_image,(box_left, box_top-20),(box_right,box_top), label_background_color, -1)
            cv2.putText(display_image,filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5], (box_left+5,box_top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
            return display_image

    def Run_TinyYolo(self, img):
        input_image = img
        display_image = input_image
        input_image = cv2.resize(input_image, (self.NETWORK_IMAGE_WIDTH, self.NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.divide(input_image, 255.0)
        input_image = input_image[:, :, ::-1]  # convert to RGB

        # Load tensor and get result.  This executes the inference on the NCS
        self.graph.LoadTensor(input_image.astype(np.float16), 'user object')
        output, userobj = self.graph.GetResult()

        # filter out all the objects/boxes that don't meet thresholds
        filtered_objs = self.filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0]) # fc27 instead of fc12 for yolo_small

        print('Displaying image with objects detected in GUI')
        print('Click in the GUI window and hit any key to exit')
        #display the filtered objects/boxes in a GUI window
        display_image = self.display_objects_in_gui(display_image, filtered_objs)
        return display_image

    def Distroy_TinyYolo(self):
        self.tinyyolo_status = False
        self.graph.DeallocateGraph()
        self.device.CloseDevice()
