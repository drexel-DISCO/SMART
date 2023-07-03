'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is SMART core functionalities (ref. TECS 2023).
'''
#Headers
import logging
import threading
from threading import Thread, Lock
import concurrent.futures
import time
import pickle
import os
from collections import defaultdict
from sys import maxsize
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


def STEUni(SG,model_name):
    start_time = time.time()                        #start a timer
    
    task_dependency = SG.dependency                 #get the dependency
    tasks   = list(task_dependency.keys())          #list of tasks
    SG      = list()                                #list schedule

    n_tasks = len(tasks)                            #number of tasks in the original graph

    while len(tasks) > 0:                           #while there are tasks that are not scheduled
        for t in tasks:                             #for each task in tasks
            dependent_tasks = task_dependency[t]    #get the list of dependent_tasks
            if len(dependent_tasks) == 0:           #the task is independent
                SG.append(t)                        #append the task to the schedule
                tasks.remove(t)                     #remove the task from the task list
            else:
                ready = True                        #use a ready variable to identify if the task can be inserted in the schedule
                for dt in dependent_tasks:          #for each task in the dependent_task list
                    if dt not in SG:                #if the dependent task is not in SG
                        ready = False               #the task is not ready to be inserted in SG
                if ready:                           #if the task is ready
                    SG.append(t)                    #insert the task in SG
                    tasks.remove(t)                 #remove the task from the task list
    elapsed_time = time.time() - start_time         #elapsed time
    logging.info('[info] STEUni of %s model took %s seconds',model_name,elapsed_time)
    n_schd = len(SG)                                #number of tasks in the schedule
    if n_tasks != n_schd:                           #if the number of tasks in the schedule is not equal to the number of tasks in the original graph
        logging.info('[error] STEUni step of %s graph has error: original graph has %s tasks, the schedule has %s tasks',model_name,n_tasks,n_schd)
    return SG

def OpMap(SG,task_tensors,model_name):
    start_time = time.time()                        #start a timer

    #x_ij = binary variable representing mapping of task i on resource j
    #i = {0,1,..,N-1}, N = number of tasks
    #j = {0,1,..,Nt}, Nt = number of tiles, j = Nt represents CPU
    ##########################
    #define the model
    ##########################
    model   = LpProblem(name='opmap',sense=LpMinimize)  #define the model
    ##########################
    #define the variables
    ##########################
    N       = len(SG)       #number of tasks
    Nt      = n_tiles + 1   #number of resources
    nvars   = N * Nt        #number of variables of the optimization problem
    y       = {i: LpVariable(name=f"y{i}", lowBound=0, upBound=1, cat='Binary') for i in range(nvars)}    #variables of the model
    t       = LpVariable("t",0,maxsize)   #linearazation variable
    p       = LpVariable("p",0,maxsize)   #performance variable
    ##########################
    #define the constraints
    ##########################
    #Mapping constraints: 
    for i in range(N):
        model += (lpSum([y[i * Nt + j] for j in range(Nt)]) == 1, 'mapping_constraint_'+str(i))
    #Linearization constraints
    for j in range(Nt):
        model += (lpSum([y[i * Nt + j]*task_tensors[SG[i]] for i in range(N)] - t) <= 0, 'size_constraint_'+str(j))
    #optional constrain
    model += (lpSum([y[(i + 1) * Nt - 1] for i in range(N)]) == 0, 'optional_constraint')
    ##########################
    #define the objective fn
    ##########################
    model   += t
    ##########################
    #solve the problem
    ##########################
    status   = model.solve(GUROBI(msg=1))
    ##########################
    #extract results
    ##########################
    variable = {}       #all variables of the optimization problem
    tiles    = list()   #tiles to which tasks are mapped
    resources= list()   #resources to which tasks are mapped
    opmap    = {}       #dictionary to return
    for var in model.variables():
        a = re.findall(r'\d+',var.name)
        if len(a) > 0:
            variable[int(a[0])] = int(var.value())
    '''
    for i in range(N):
        b = [variable[i*Nt + j] for j in range(Nt)]
        print(b)
    '''
    for i in range(N):
        #check for errors
        if np.sum([variable[i * Nt + j] for j in range(Nt)]) != 1:
            logging.error('[error] binary ILP error: task mapped to no tile / more than one tile')
            exit()
        #assign the tile
        tiles_i = [variable[i * Nt + j] for j in range(Nt)].index(1)
        tiles.append(tiles_i)
    resources = ['npu' if tl < Nt-1 else 'cpu' for tl in tiles]

    for i, (ti, tr) in enumerate(zip(tiles,resources)):
        opmap[SG[i]] = {'resource':tr,'tile':ti}
    
    elapsed_time = time.time() - start_time         #elapsed time
    logging.info('[info] OpMap of %s model took %s seconds',model_name,elapsed_time)
    return opmap

def ActMap(g,task_map,model_name):
    start_time = time.time()                        #start a timer

    task_ids        = [m['task'] for m in task_map if m['resource'] == 'npu']
    task_tiles      = [m['tile'] for m in task_map if m['resource'] == 'npu']
    task_tensors    = [g.tensor[tid] for tid in task_ids]

    #create a tile map
    tile_map        = defaultdict(list)
    for ti,ta in zip(task_tiles,task_ids):
        tile_map[ti].append(ta)

    actmap          = list()                #list of tasks whose output are to be mapped to spm

    #for each tile
    for key in tile_map.keys():
        global_buffer   = list()            #content of this buffer may be pinned to spm
        local_buffer    = list()            #content of this buffer is always pinned to spm
        tasks           = tile_map[key]     #all tasks mapped to this tile
        reuse           = {}                #reuse factor to incorporate in tensor placement
        for ta in tasks:
            ta_out_nodes            = g.graph[ta]                                               #all output nodes of task ta
            ta_out_nodes_npu        = np.intersect1d(ta_out_nodes,task_ids)                     #how many of those nodes are on a NPU
            ta_out_nodes_resources  = [task_map[taon]['resource'] for taon in ta_out_nodes]     #resources of all output nodes
            ta_out_nodes_tiles      = [task_map[taon]['tile'] for taon in ta_out_nodes]         #tile id of all output nodes
            reuse[ta]               = len(ta_out_nodes_npu)                                     #reuse factor is equal to how many times the task's output is being used
            #consider different cases
            if 'cpu' in ta_out_nodes_resources:                     #output of ta is needed for a task on cpu
                global_buffer.append(ta)
            elif len(np.setdiff1d(ta_out_nodes_tiles,[key])) > 0:   #output of ta is needed for a task on a different npu
                global_buffer.append(ta)
            else:                                                   #output is needed in the same tile
                #find all paths from ta to a task in ta_out_nodes
                local_spm = True
                for taon in ta_out_nodes:
                    all_paths = g.getAllPaths(ta,taon)
                    for p in all_paths:
                        if len(p) > 2:
                            local_spm = False
                if local_spm:
                    local_buffer.append(ta)
                else:
                    global_buffer.append(ta)
        #solve an optimization problem here
        #x_i = binary variable representing mapping of the tensor of task i to spm or memory
        #i = tasks in the global buffer
        ##########################
        #define the model
        ##########################
        model   = LpProblem(name='actmap_'+str(key),sense=LpMinimize)  #define the model
        ##########################
        #define the variables
        ##########################
        N       = len(global_buffer)    #number of elements of the global buffer
        nvars   = N
        y       = {i: LpVariable(name=f"y{i}", lowBound=0, upBound=1, cat='Binary') for i in range(nvars)}    #variables of the model
        ##########################
        #define the constraints
        ##########################
        model += (lpSum([y[i]*g.tensor[global_buffer[i]]*bit_precision/(8*1024.0) for i in range(nvars)]) <= spm_sz_per_tile, 'mapping_constraint')
        ##########################
        #define the objective fn
        ##########################
        model += (lpSum([(1-y[i])*g.tensor[global_buffer[i]]*reuse[global_buffer[i]]*bit_precision/(8*1024.0) for i in range(nvars)]))
        ##########################
        #solve the model
        ##########################
        status   = model.solve(GUROBI(msg=1))
        ##########################
        #extract the results
        ##########################
        variable = {}       #all variables of the optimization problem
        for var in model.variables():
            a = re.findall(r'\d+',var.name)
            if len(a) > 0:
                variable[int(a[0])] = int(var.value())
        #assign the variables to local buffer
        for k in variable.keys():
            if variable[k] == 1:
                local_buffer.append(global_buffer[k])
        actmap += local_buffer

    elapsed_time = time.time() - start_time         #elapsed time
    logging.info('[info] ActMap of %s model took %s seconds',model_name,elapsed_time)
    return actmap

def IPCSchd(g,task_order,schd,model_name):
    start_time = time.time()                        #start a timer

    sg              = copy.deepcopy(task_order)
    task_map        = schd.mapping
    dma_times       = schd.dma_times
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
    gc = copy.deepcopy(g)
    #add other edges due to steuni
    for key in par_ste_order.keys():            #for all resources
        mapped_tasks = par_ste_order[key]       #mapped tasks
        if len(mapped_tasks) > 1:               #if there are more than 1 tasks mapped to the resource
            start_task = mapped_tasks.pop(0)    #get the first task
            while len(mapped_tasks) > 0:        #still tasks in the mapped tasks
                end_task = mapped_tasks.pop(0)  #get the next task
                if end_task not in gc.graph[start_task]:
                    gc.addEdge(start_task,end_task)
                start_task = end_task
    #add comm edges
    all_tasks   = sg                #all computing tasks
    all_tasks.sort()                #sort tasks in ascending order
    max_task_id = max(sg)           #highest task id
    comm_base_id= max_task_id + 1   #base id of communication task

    all_tasks_extime = {}           #execution task of all tasks
    #all computation tasks
    for task in all_tasks:          #for each computation task
        all_tasks_extime[task] = schd.ex_end_times[task] - schd.ex_start_times[task]
    #all communication tasks
    for dtx in dma_times:
        src = dtx.task['src']
        dst = dtx.task['dst']
        ext = dtx.end_time - dtx.start_time
        if dst is not None:
            #add a communication actor
            all_tasks.append(comm_base_id)
            #add execution time of this actor
            all_tasks_extime[comm_base_id] = ext
            #configure the connection
            if dst not in gc.graph[src]:
                logging.info('[error] %s is not dependent on %s and dependent list is %s',dst,src,gc.graph[src])
                exit()
            else:
                gc.addEdge(src,comm_base_id)
                gc.addEdge(comm_base_id,dst)
                gc.graph[src].remove(dst)
            #make ready for the next communication actor
            comm_base_id += 1
        else:
            all_tasks_extime[src] += ext
    #the graph is created here.
    #now we use an ILP to solve this, just to see how much improvement we can make
    ##########################
    #define the model
    ##########################
    model   = LpProblem(name='parschd',sense=LpMinimize)  #define the model
    ##########################
    #define the variables
    ##########################
    #each variable represents the start time of an actor
    nvars   = len(all_tasks)    #number of variables = number of tasks
    y       = {i: LpVariable(name=f"y{i}", lowBound=0, upBound=maxsize) for i in range(nvars)}    #variables of the model
    t       = LpVariable("t",0,maxsize)   #linearazation variable
    ##########################
    #define the constraints
    ##########################
    #linearization constraints
    for i in range(nvars):
        src   =  all_tasks[i]
        model += (y[i] + all_tasks_extime[src] - t <= 0, 'linearization_constraint_'+str(i))
    #dependency constraint
    for i in range(nvars):
        src     = all_tasks[i] 
        edges   = gc.graph[src]
        for dst in edges:
            j = all_tasks.index(dst)
            model += (y[j] - y[i] - all_tasks_extime[src] >= 0, 'extime_constraint_'+str(i)+'_'+str(j))
    ##########################
    #define the objective
    ##########################
    model   += t
    ##########################
    #solve the problem
    ##########################
    status   = model.solve(GUROBI(msg=1))
    ##########################
    #extract results
    ##########################
    variable = {}                   #all variables of the optimization problem
    for var in model.variables():
        a = re.findall(r'\d+',var.name)
        if len(a) > 0:
            variable[all_tasks[int(a[0])]] = int(var.value())
    new_start_time  = copy.deepcopy(variable)
    new_end_time    = {}
    for key in new_start_time.keys():
        new_end_time[key] = new_start_time[key] + all_tasks_extime[key]
    new_schd = copy.deepcopy(schd)                  #copy the original schedule
    new_schd.add_ex_start_times(new_start_time)     #create new start times
    new_schd.add_ex_end_times(new_end_time)         #create new end times
    
    elapsed_time = time.time() - start_time         #elapsed time
    logging.info('[info] IPCSchd of %s model took %s seconds',model_name,elapsed_time)
    return new_schd

def ParSchd(g,schd,model_name):
    start_time      = time.time()          #start a timer
    all_actors      = schd.ex_start_times.keys()
    maxCompActorID  = g.getMaxNodeId()
    startTimes      = list()
    endTimes        = list()
    maxDiff         = 0
    batchDelays     = list()
    for a in all_actors:
        if a > maxCompActorID:  #consider only the communication actors
            startTimes.append(schd.ex_start_times[a])
            endTimes.append(schd.ex_end_times[a])
    maxEndTime      = max(endTimes) + 1
    linear          = np.zeros(2 * maxEndTime)
    for st,et in zip(startTimes,endTimes):
        diff = et - st
        if diff > maxDiff:
            maxDiff = diff
        for i in range(st,et):
            linear[i] = 1
    
    dmaOccupancy= copy.deepcopy(linear)    #this is the dma occupancy
    shift       = maxDiff
    while shift < maxEndTime:
        b       = copy.deepcopy(linear)
        c       = np.roll(b,shift)
        d       = dmaOccupancy + c
        if len(np.where(d == 2)[0]) == 0:
            batchDelays.append(shift)
            dmaOccupancy = copy.deepcopy(d)
        shift   += maxDiff
    par_schd = copy.deepcopy(schd)
    par_schd.add_batch_delays(batchDelays)

    elapsed_time = time.time() - start_time         #elapsed time
    logging.info('[info] ParSchd of %s model took %s seconds',model_name,elapsed_time)
    return par_schd
