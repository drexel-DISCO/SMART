'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is the implementation of our list scheduler.
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

#Global Parameters
import globals

#Configuration Parameters
from params import *

#Helper fns
from helper_fns import *
from schedule_class import *

#Resource Definition
from nsoc_resource import *

def list_scheduler(g,sg,task_map,model_name):
    ###############################
    #initialize local variables
    ###############################
    start_time      = time.time()       #start a timer
    n_tasks         = len(task_map)     #number of tasks
    s               = Schedule(n_tasks) #create an empty schedule
    run             = True              #run variable
    compute_threads = list()            #list of threads
    tensor_locs     = {}                #location of tensors as specified in the mapping
    ###############################
    #initialize global variables
    ###############################
    globals.task_start_times= {}
    globals.task_end_times  = {}
    globals.dma_times       = list()
    globals.mem_times       = list()
    globals.cpu_cycles      = 0
    #initialize the completion status
    for key in g.dependency.keys():
        globals.completion_status[key] = 0

    #extract the tensor locations
    for t in task_map:
        tensor_locs[t['task']] = t['tensor']
    
    resource_map    = (list(set(['cpu' if m['resource'] == 'cpu' else m['resource']+'_'+str(m['tile'])+'_'+m['name'] for m in task_map])))    #different types of resource used
    par_ste_order   = {}  #create order for parallel resources
    for resmap in resource_map:
        par_ste_order[resmap] = list()  #create empty orders for each source
    for to in sg:                       #create the order for each task from the ste order
        resmap_to  = task_map[to]['resource']   #get the resource for the task
        tilemap_to = str(task_map[to]['tile'])  #get the tile for the task
        layermap_to= task_map[to]['name']       #get the name for the task
        if resmap_to == 'cpu':                  #if the resource is cpu
            key = 'cpu'                         #the key is cpu
        else:                                   #else
            key = resmap_to + '_' + tilemap_to + '_' + layermap_to  #key is resource_tile
        par_ste_order[key].append(to)           #add the task to the order
    for key in par_ste_order.keys():            #for each pe resource
        thread = threading.Thread(target=pe, args=(g,key,par_ste_order[key],tensor_locs,))    #define the thread
        compute_threads.append(thread)          #append the thread to the list of threads
        thread.start()                          #start the pe thread
    while run:
        all_task_status = list(globals.completion_status.values())  #get status of all tasks
        if 0 not in all_task_status:
            run = False
        time.sleep(usec)
        globals.cpu_cycles += 1

    s.add_ex_start_times(globals.task_start_times)      #add ex start times
    s.add_ex_end_times(globals.task_end_times)          #add ex end times
    s.add_dma_times(globals.dma_times)                  #add dma times
    s.add_mem_times(globals.mem_times)                  #add mem times
    s.add_mapping(task_map)                             #add the resources
    #wait for 1 sec
    time.sleep(1)
    elapsed_time    = time.time() - start_time          #elapsed time
    logging.info('[info] List scheduling of %s model took %s seconds',model_name,elapsed_time)

    return s
