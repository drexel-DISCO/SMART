'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is a multi-threaded implementation of the nsoc resources.
'''
#Headers
import threading
from threading import Thread, Lock, Semaphore
import logging
import concurrent.futures
import time
import numpy as np

#Global Parameters
import globals
from schedule_class import *

#Configuration Parameters
from params import *

def mem(size,task,dst):
    globals.sem_mem.acquire()                   #lock the memory channel
    header = {'src':task,'dst':dst}             #transfer header
    s = MemorySchedule(header)                  #create empty memory schedule
    current_elapsed_time = globals.cpu_cycles   #current time
    s.set_start_time(current_elapsed_time)      #set the start time of the schedule
    logging.info('[info] Task %s Start Mem Transfer at %s',task,current_elapsed_time)

    mem_cycles  = size / mem_tx_granularity     #total amount of data to be transferred
    mem_time    = mem_cycles * mem_clock_period #total number of clock cycles needed for this transfer 
    expected_completion_time = current_elapsed_time + mem_time  #expected end time of the memory transfer
    while(current_elapsed_time < expected_completion_time):     #keep checking if the time has elapsed, exit the loop if so
        time.sleep(usec)                                        #sleep for a us 
        current_elapsed_time = globals.cpu_cycles               #keep pooling the current time
    current_elapsed_time = globals.cpu_cycles   #current time of the system
    s.set_end_time(current_elapsed_time)        #set the end time of the schedule
    logging.info('[info] Task %s End Mem Transfer at %s',task,current_elapsed_time)
    globals.mem_times.append(s)                 #append the schedule to the global list of memory transactions
    globals.sem_mem.release()                   #release the memory channel

def dma(size,task,dst):
    globals.sem_dma.acquire()                   #lock the external dma channel
    header = {'src':task,'dst':dst}             #transfer header
    s = MemorySchedule(header)                  #create empty dma schedule
    current_elapsed_time = globals.cpu_cycles   #current time
    s.set_start_time(current_elapsed_time)      #set the start time of the schedule
    logging.info('[info] Task %s Start DMA Transfer at %s',task,current_elapsed_time)

    dma_cycles  = size / dma_tx_granularity     #total amount of data to be transferred
    dma_time    = dma_cycles * dma_clock_period #total number of clock cycles needed for this transfer
    expected_completion_time = current_elapsed_time + dma_time  #expected end time of the dma transfer
    while(current_elapsed_time < expected_completion_time):     #keep checking if the time has elapsed, exit the loop if so
        time.sleep(usec)                                        #sleep for a us 
        current_elapsed_time = globals.cpu_cycles               #keep pooling the current time
    current_elapsed_time = globals.cpu_cycles   #current time of the system
    s.set_end_time(current_elapsed_time)        #set the end time of the schedule
    logging.info('[info] Task %s End DMA Transfer at %s',task,current_elapsed_time)
    globals.dma_times.append(s)                 #append the schedule to the global list of memory transaction
    globals.sem_dma.release()                   #release the dma channel

def pe(g,pe_type,task_list,tensor_locs):        #this is a processing element
    current_elapsed_time = globals.cpu_cycles
    logging.info('[info] Starting PE %s at time %s',pe_type,current_elapsed_time)
    for t in task_list: #for each task mapped to this PE
        #check if all its dependent tasks are ready
        #if ready, fire the task and wait for its completion
        #otherwise wait
        task_resource               = pe_type.split('_')[0]                                         #task's resource = cpu or npu
        task_extime                 = g.extime[t][task_resource]                                    #task's extime
        dependent_tasks_t           = g.dependency[t]                                               #list of tasks on which t depends on
        dependent_tasks_t_tensor_loc= [tensor_locs[task_t] for task_t in dependent_tasks_t]         #dependent task's tensors location
        dependent_tasks_t_tensor_sz = [g.tensor[task_t] for task_t in dependent_tasks_t]            #dependent task's tensors size

        ready                       = False                                                         #assume the task is not ready
        while not ready:        #while the task is not ready
            time.sleep(usec)    #sleep for a us and check again
            dependent_tasts_t_status    = [globals.completion_status[ti] for ti in dependent_tasks_t]   #status of these tasks
            if 0 not in dependent_tasts_t_status or len(dependent_tasts_t_status) == 0: #if all dependent tasks are done or no dependent tasks
                ready = True    #make the task ready
        #if the task is not ready, then lets wait
        #else fire the task on this pe
        if ready:
            ####################################
            #get input
            ####################################
            tr_threads = list()
            for tr,tr_loc,tr_sz in zip(dependent_tasks_t,dependent_tasks_t_tensor_loc,dependent_tasks_t_tensor_sz):
                if tr_loc == 'mem' and 'cpu' in pe_type:
                    thread = threading.Thread(target=mem, args=(tr_sz,tr,t,))
                    thread.start()
                    tr_threads.append(thread)
                elif tr_loc == 'mem' and 'npu' in pe_type:
                    thread = threading.Thread(target=dma, args=(tr_sz,tr,t,))
                    tr_threads.append(thread)
                    thread.start()
                elif tr_loc == 'spm' and 'cpu' in pe_type:
                    thread = threading.Thread(target=dma, args=(tr_sz,tr,t,))
                    tr_threads.append(thread)
                    thread.start()
            for thread in tr_threads:
                thread.join()
            ####################################
            #execute
            ####################################
            current_elapsed_time = globals.cpu_cycles
            logging.info('[info] PE = %s, Starting task %s (extime = %s) at time %s',pe_type,t,task_extime,current_elapsed_time)
            globals.task_start_times[t] = current_elapsed_time                  #fill the task start times
            expected_completion_time    = current_elapsed_time + task_extime    #find the expected end time
            while(current_elapsed_time < expected_completion_time):             #wait for the task completion
                time.sleep(usec)                                                #check every us to see if the task has completedc 
                current_elapsed_time = globals.cpu_cycles                       #update the current elapsed time
            globals.task_end_times[t]   = current_elapsed_time                  #fill the task end times
            logging.info('[info] PE = %s, Ending task %s at time %s',pe_type,t,current_elapsed_time)
            ####################################
            #save output
            ####################################
            rx_threads = list()
            if tensor_locs[t] == 'mem' and 'cpu' in pe_type:
                thread = threading.Thread(target=mem, args=(g.tensor[t],t,None,))
                thread.start()
                rx_threads.append(thread)
            elif tensor_locs[t] == 'mem' and 'npu' in pe_type:
                thread = threading.Thread(target=dma, args=(g.tensor[t],t,None,))
                thread.start()
                rx_threads.append(thread)
            elif tensor_locs[t] == 'spm' and 'cpu' in pe_type:
                thread = threading.Thread(target=dma, args=(g.tensor[t],t,None,))
                thread.start()
                rx_threads.append(thread)
            #join all threads
            for thread in rx_threads:
                thread.join()
            ####################################
            #update the completion status
            ####################################
            globals.completion_status[t] = 1                                #update the status of the task as completed 

