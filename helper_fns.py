'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is all supporting functions needed to run the schedulers.
'''

import numpy as np
import copy
import logging
import pickle
import os
import subprocess

import globals
from params import *
from schedule_class import *
from graph_class import *

def read_graph_data(model_name):
    graph_name  = 'models/'+model_name+'/graph.txt'
    extime_name = 'models/'+model_name+'/extime.pkl'
    tensor_name = 'models/'+model_name+'/tensors.pkl'
    weight_name = 'models/'+model_name+'/weights.pkl'
    type_name   = 'models/'+model_name+'/layer_types.pkl'
    map_name    = 'models/'+model_name+'/mapping.pkl'
    tmap_name   = 'models/'+model_name+'/tensor_mapping.pkl'

    #create task_dependency
    logging.info('[info] Reading Model Graph Information')
    modelGraph  = np.loadtxt(graph_name,dtype ='int')
    nodes       = np.unique(modelGraph.flatten())
    n_nodes     = len(nodes)
    #Define a graph structure
    Gsdcnn      = Graph(n_nodes)

    for n in nodes:
        Gsdcnn.dependency[n] = list()
    for i in range(len(modelGraph)):
        s = modelGraph[i,0]
        d = modelGraph[i,1]
        Gsdcnn.addEdge(s,d)
    
    logging.info('[info] Done')

    #read execution time
    logging.info('[info] Reading Execution Time Information')
    task_extime = pickle.load(open(extime_name, "rb"))
    for i,e in enumerate(task_extime):
        Gsdcnn.addExtime(i,e)
    logging.info('[info] Done')

    #read task tensors
    logging.info('[info] Reading Tensor Information')
    task_tensors = pickle.load(open(tensor_name, "rb"))
    for key in task_tensors.keys():
        Gsdcnn.addTensor(key,task_tensors[key])
    logging.info('[info] Done')

    #read task weights
    logging.info('[info] Reading Weight Information')
    if os.path.exists(weight_name):
        task_weights = pickle.load(open(weight_name, "rb"))
        for key in task_weights.keys():
            Gsdcnn.addWeight(key,task_weights[key])
    else:
        exit()
    logging.info('[info] Done')

    #read task types
    logging.info('[info] Reading Types Information')
    if os.path.exists(type_name):
        task_types = pickle.load(open(type_name, "rb"))
        for key in task_types.keys():
            Gsdcnn.addLtype(key,task_types[key])
    else:
        exit()
    logging.info('[info] Done')


    return Gsdcnn

def uniquefy(l):
    lset  = set(l)
    ulist = (list(lset))
    return ulist
