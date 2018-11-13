from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys

class AgeNet:
    def __init__(self):
        self.agenet_status = False
        self.cascade_face = cv2.CascadeClassifier("./xml/face.xml")
        self.dim = (227,227)
        print('-----------------------init agenet-----------------------')

    def prepare(self):
        self.agenet_status = True
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No devices found')
        self.device = mvnc.Device(devices[0])
        self.device.OpenDevice()
        opt = self.device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
        blob = "./graph/age_graph"
        with open(blob, mode='rb') as f:
            blob = f.read()
        self.graph = self.device.AllocateGraph(blob)
        self.graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
        iterations = self.graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)
        self.age_list=['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
        self.ilsvrc_mean = numpy.load('./data/age_gender_mean.npy').mean(1).mean(1) #loading the mean file
        print('-----------------------prepare agenet successful-----------------------')

    def Run_AgeNet(self, img):
        print('-----------------------run agenet-----------------------')
        face = self.cascade_face.detectMultiScale(img, 1.3, 2)
        for(x, y, w, h) in face:
            frame = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, self.dim)
            img[:,:,0] = (img[:,:,0] - self.ilsvrc_mean[0])
            img[:,:,1] = (img[:,:,1] - self.ilsvrc_mean[1])
            img[:,:,2] = (img[:,:,2] - self.ilsvrc_mean[2])
            self.graph.LoadTensor(img.astype(numpy.float16), 'user object')
            output, userobj = self.graph.GetResult()
            order = output.argsort()
            last = len(order) - 1
            predicted = int(order[last])
            # print('the age range is ' + self.age_list[predicted] + ' with confidence of %3.1f%%' % (100.0*output[predicted]))
            # output_str = str('age:' + self.age_list[predicted] + '; confidence:%3.1f%%' % (100.0*output[predicted]))
            # cv2.putText(frame, output_str, (0,30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
            if 100.0*output[predicted] < 70:
                return ['Uncertain', '**', frame]
            return [self.age_list[predicted], 100.0*output[predicted], frame]
        return ['no face', '0', img]

    def Distroy_AgeNet(self):
        self.agenet_status = False
        self.graph.DeallocateGraph()
        self.device.CloseDevice()
        print('-----------------------distroy agenet-----------------------')
