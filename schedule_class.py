'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is the class definition of a schedule.
'''

class Schedule:
    """
    Class to hold a schedule
    """
    ex_start_times  = {}
    ex_end_times    = {}
    batch_delays    = list()
    dma_times       = list()
    mem_times       = list()

    mapping         = list()

    def __init__(self,N):
        self.ex_start_times     = {}
        self.ex_end_times       = {}
        self.dma_times          = list()
        self.mem_times          = list()
        self.mapping            = list()

    def add_ex_start_times(self,ex_start_times):
        self.ex_start_times = ex_start_times

    def add_ex_end_times(self,ex_end_times):
        self.ex_end_times = ex_end_times

    def add_batch_delays(self,delay_times):
        self.batch_delays = delay_times

    def add_dma_times(self,dma_times):
        self.dma_times = dma_times

    def add_mem_times(self,mem_times):
        self.mem_times = mem_times

    def add_mapping(self,mapping):
        self.mapping = mapping

    def get_completion_time(self):
        end_times       = list(self.ex_end_times.values())
        max_end_time    = max(end_times)
        if len(self.batch_delays) == 0:
            #return max(end_times)
            return max_end_time
        else:
            batch_end_time = max_end_time + max(self.batch_delays)
            return batch_end_time / (len(self.batch_delays) + 1)

class MemorySchedule:
    task        = None
    start_time  = None
    end_time    = None

    def __init__(self,task):
        self.task       = task
        self.start_time = None
        self.end_time   = None

    def set_start_time(self,start_time):
        self.start_time = start_time
    def set_end_time(self,end_time):
        self.end_time = end_time
    def get_task(self):
        return self.task
    def get_start_time(self):
        return self.start_time
    def get_end_time(self):
        self.end_time
