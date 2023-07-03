'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is a multi-threaded SMART task scheduler in Python.
'''
#Headers
import logging
import threading
from threading import Thread, Lock
import concurrent.futures
import time
import pickle
import numpy as np

#Global Parameters
import globals

#Configuration Parameters
from params import *

#Helper fns
from helper_fns import *
from scheduler import *
from schedule_class import *
from mapper import *

#Resource Definition
from nsoc_resource import *

#Import the design flow
import SMART

#Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model','--model',default='example')
parser.add_argument('-fname','--fname',default='run.log')

#Initialization
args 		                    = vars(parser.parse_args())
model_name                      = args['model']
log_fname                       = args['fname']

if __name__ == "__main__":
    format = "[%(asctime)s]: %(message)s"
    handlers = [logging.FileHandler(log_fname), logging.StreamHandler()]
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S",handlers = handlers)

    #start a timer
    start_time = time.time()
    #Reading all headers
    logging.info('[info] Reading %s Graph Information',model_name)
    Gsdcnn = read_graph_data(model_name)
    logging.info('[info] Done')
    elapsed = time.time() - start_time
    logging.info('[info] Reading %s graph took %s seconds',model_name,elapsed)
   
    #STEUni
    task_order  = SMART.STEUni(Gsdcnn,model_name)
    start_map   = mapper(Gsdcnn)
    steuni_schd = list_scheduler(Gsdcnn,task_order,start_map,model_name)
    
    #OpMap
    filtered_task_order = list()    #create a filter for those tasks that are supported on the NPU
    for task in task_order:
        if Gsdcnn.ltype[task] in NPU_SUPPORTED_OPERATIONS or len(NPU_SUPPORTED_OPERATIONS) == 0:
            filtered_task_order.append(task)
    task_tensors= Gsdcnn.tensor
    opmap       = SMART.OpMap(filtered_task_order,task_tensors,model_name)
    task_ids    = [m['task'] for m in start_map]
    opmap_map   = copy.deepcopy(start_map)
    for key in opmap.keys():
        tile     = opmap[key]['tile']
        resource = opmap[key]['resource']
        opmap_map[task_ids.index(key)]['resource']= resource 
        opmap_map[task_ids.index(key)]['tile']    = tile
    opmap_schd = list_scheduler(Gsdcnn,task_order,opmap_map,model_name)

    #ActMap
    actmap      = SMART.ActMap(Gsdcnn,opmap_map,model_name)
    actmap_map  = copy.deepcopy(opmap_map)
    all_tasks   = [task['task'] for task in actmap_map]
    for task in actmap:
        task_id                         = all_tasks.index(task)
        actmap_map[task_id]['tensor']   = 'spm'
    actmap_schd = list_scheduler(Gsdcnn,task_order,actmap_map,model_name)
    
    #STEPar
    stepar_schd = copy.deepcopy(actmap_schd)

    #IPCSchd
    ipc_schd        = SMART.IPCSchd(Gsdcnn,task_order,stepar_schd,model_name)

    #ParSchd
    par_schd        = SMART.ParSchd(Gsdcnn,ipc_schd,model_name)

    #save the results
    ofname          = 'results/'+model_name+'.pkl' 
    schedule_dict   = {'steuni':steuni_schd, 'opmap':opmap_schd, 'actmap':actmap_schd, 'stepar':stepar_schd, 'ipcschd':ipc_schd, 'parschd':par_schd}
    pickle.dump(schedule_dict,open(ofname,'wb'))
    logging.info('[info] steuni_schd = %s, opmap_schd = %s, actmap_schd = %s, stepar_schd = %s, ipc_schd = %s, par_schd = %s',steuni_schd.get_completion_time(),opmap_schd.get_completion_time(),actmap_schd.get_completion_time(),stepar_schd.get_completion_time(),ipc_schd.get_completion_time(),par_schd.get_completion_time())
