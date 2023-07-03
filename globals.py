'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is the set of global variables that are used in the scheduler.
'''

from threading import Lock,Semaphore
from params import *

sem_dma             = Semaphore(1)  #semaphore for DMA channel
sem_mem             = Semaphore(1)  #semaphore for memory channel

completion_status   = {}    #task status
task_dependency     = {}    #task dependencies
task_extime         = {}    #task execution time
task_tensors        = {}    #task tensors
task_weights        = {}    #task tensors
task_types          = {}    #task types
tensor_mapping      = {}    #tensor mapping

cpu_cycles          = 0     #cpu cycles

task_start_times    = {}    #start times of execution
task_end_times      = {}    #end times of execution
dma_times           = list()#dma start times
mem_times           = list()#mem start times
