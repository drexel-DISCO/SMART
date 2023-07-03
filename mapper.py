'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is a mapper function. It can be extended to generate several mapping
alternatives.
'''
#Headers
import logging
import threading
from threading import Thread, Lock
import concurrent.futures
import time
import pickle
import numpy as np
import random

from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
from pulp import PULP_CBC_CMD, GUROBI
import pulp as p
import re

#Global Parameters
import globals

#Configuration Parameters
from params import *

#Helper fns
from helper_fns import *
from schedule_class import *

#Resource Definition
from nsoc_resource import *

def mapper(g):
    layer_ids    = list(g.ltype.keys())   #all layer ids
    layer_types  = list(g.ltype.values()) #all layer types
    task_tensors = list(g.tensor.values())#all tensor values
    task_map     = list()    #a task map is a list of dictionary items
    #Generate a single map
    for task_id,layer_type in enumerate(layer_types):
        layer_id    = layer_ids[task_id] #layer id
        #task mapping explorations
        tile_id = 0
        resource = 'cpu'
        #process mapping
        if layer_type not in NPU_SUPPORTED_OPERATIONS:
            resource = 'cpu'
        #hw specific name
        layer_name = layer_type
        if 'conv' in layer_name.lower() and resource == 'npu':
            layer_name = 'conv'
        elif 'pool' in layer_name.lower() and resource == 'npu':
            layer_name = 'pool'
        elif 'dense' in layer_name.lower() and resource == 'npu':
            layer_name = 'dense'
        elif resource == 'npu':
            layer_name = 'sfu'

        #tensor mapping explorations
        tensor_location = 'mem'

        m = {
            'task':layer_id,
            'layer':layer_type,
            'name':layer_name,
            'resource':resource,
            'tile':tile_id,
            'tensor':tensor_location
        }
        task_map.append(m)
    #tensor mapping here
    task_ids    = [m['task'] for m in task_map]
    tile_ids    = [m['tile'] for m in task_map]
    tensor_locs = [m['tensor'] for m in task_map]
    resources   = [m['resource'] for m in task_map]
    tensor_szs  = [g.tensor[t] for t in task_ids]


    tensor_locs = [m['tensor'] for m in task_map]
    logging.info('[info] mapped %s operations to SPM out of a total %s',tensor_locs.count('spm'),len(tensor_locs))
    return task_map
