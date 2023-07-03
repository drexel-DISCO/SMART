'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is all parameters used in the scheduler.
'''

#############################
#Main Simulator Configuration
#############################
#resource
n_tiles=4           #number of tiles
spm_sz_per_tile=32  #scratchpad size per tile in KB
bit_precision=8     #bit precision of synaptic weights and tensor
usec=0.000001       #time resolution

#supportedoperations
ALL_SUPPORTED_OPERATIONS=[]
SPECK_SUPPORTED_OPERATIONS=[]
MUBRAIN_SUPPORTED_OPERATIONS=['Activation', 'Add', 'Conv2D', 'Concatenate', 'Dense', 'InputLayer', 'Normalization']
GRAI_SUPPORTED_OPERATIONS=['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'BatchNormalization', 'ZeroPadding2D', 'Dropout', 'Activation', 'AveragePooling2D', 'Concatenate', 'GlobalAveragePooling2D', 'ReLU', 'DepthwiseConv2D', 'Add']
NPU_SUPPORTED_OPERATIONS=GRAI_SUPPORTED_OPERATIONS

#clocks
dma_tx_granularity=64*8     #in bits
dma_clock_period=10.0       #in micro-seconds
mem_tx_granularity=128*8    #in bits
mem_clock_period=10.0       #in micro-seconds
