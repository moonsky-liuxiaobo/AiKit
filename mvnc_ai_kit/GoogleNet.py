#! /usr/bin/env python3

# Copyright 2017 Intel Corporation. 
# The source code, information and material ("Material") contained herein is  
# owned by Intel Corporation or its suppliers or licensors, and title to such  
# Material remains with Intel Corporation or its suppliers or licensors.  
# The Material contains proprietary information of Intel or its suppliers and  
# licensors. The Material is protected by worldwide copyright laws and treaty  
# provisions.  
# No part of the Material may be used, copied, reproduced, modified, published,  
# uploaded, posted, transmitted, distributed or disclosed in any way without  
# Intel's prior express written permission. No license under any patent,  
# copyright or other intellectual property rights in the Material is granted to  
# or conferred upon you, either expressly, by implication, inducement, estoppel  
# or otherwise.  
# Any license under such intellectual property rights must be express and  
# approved by Intel in writing.

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys

class GoogleNet:
	def __init__(self):
		self.googlenet_status = False
		print('-----------------------init googlenet-----------------------')

	def prepare(self):
		self.googlenet_status = True
		self.dim=(224,224)
		labels_file='./data/synset_words.txt'
		self.labels=numpy.loadtxt(labels_file,str,delimiter='\t')
		mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
		devices = mvnc.EnumerateDevices()
		if len(devices) == 0:
			print('No devices found')
			quit()

		self.device = mvnc.Device(devices[0])
		self.device.OpenDevice()
		network_blob='./graph/GoogleNet_graph'

		with open(network_blob, mode='rb') as f:
			blob = f.read()
		self.graph = self.device.AllocateGraph(blob)

		self.ilsvrc_mean = numpy.load('./data/ilsvrc_2012_mean.npy').mean(1).mean(1)
	
	def Run_GoogleNet(self, img):
		img=cv2.resize(img, self.dim)
		img = img.astype(numpy.float32)
		img[:,:,0] = (img[:,:,0] - self.ilsvrc_mean[0])
		img[:,:,1] = (img[:,:,1] - self.ilsvrc_mean[1])
		img[:,:,2] = (img[:,:,2] - self.ilsvrc_mean[2])

		self.graph.LoadTensor(img.astype(numpy.float16), 'user object')
		output, userobj = self.graph.GetResult()
		order = output.argsort()[::-1][:6]
		res = []
		for i in range(0,4):
			if output[order[i]] > 0.20:
				res.append(self.labels[order[i]]+'('+str(int(output[order[i]]*100))+'%)')
				#print ('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self.labels[order[i]] + '  label index is: ' + str(order[i]) )
			else:
				res.append(' ')
		return res
		
	def Distroy_GoogleNet(self):
		self.googlenet_status = False
		self.graph.DeallocateGraph()
		self.device.CloseDevice()


if __name__ == '__main__':
	g = GoogleNet()
	g.prepare()
	img = cv2.imread('./1.png')
	g.Run_GoogleNet(img)