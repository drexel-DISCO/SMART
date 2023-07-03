'''
Copyright © 2023 Prof. Anup Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Author      : Anup Das
Date        : July 02, 2023
Version     : 2.0
Description : This is the graph class file. Models are converted into this graph class and then
processed using the scheduler.
'''

from collections import defaultdict
import copy

class node:
    idx         = -1
    inC         = list()
    outC        = list()
    activation  = -1
    weight      = -1

    def __init__(self,node_id):
        self.idx = node_id

    def add_input(self,in_node):
        self.inC.append(in_node)

    def add_output(self,out_node):
        self.outC.append(out_node)

    def add_activation(self,activation):
        self.activation = activation

    def add_weight(self,weight):
        self.weight = weight



# This class represents a directed graph
# using adjacency list representation
class Graph:

    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices
		
        # No. of vertices
        self.paths = list()

        # path counts
        self.path_cnt = 0
		
        # default dictionary to store graph
        self.graph = defaultdict(list)
    
        # default dictionary to store dependency
        self.dependency = defaultdict(list)
    
        # default dictionary to store execution time
        self.extime = defaultdict(list)

        # default dictionary to store tensors
        self.tensor = defaultdict(list)

        # default dictionary to store weight
        self.weight = defaultdict(list)

        # default dictionary to store type
        self.ltype  = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.dependency[v].append(u)

    def addExtime(self, u, e):
        self.extime[u] = e

    def addTensor(self, u, e):
        self.tensor[u] = e

    def addWeight(self, u, e):
        self.weight[u] = e

    def addLtype(self, u, e):
        self.ltype[u] = e

    def getMaxNodeId(self):
        nodes = list(self.graph.keys())
        for key in self.graph.keys():
            nodes += self.graph[key]
        return max(nodes)

    '''A function to make a hyperh=graph by replicating the graph n times'''
    def duplicateAndAdd(self):
        startNodeID = self.getMaxNodeId() + 1
        #add the edges
        self.dependency[startNodeID] = []
        d = copy.deepcopy(self.graph)
        for key,value in d.items():
            s = key + startNodeID
            for v in value:
                d = v + startNodeID
                self.addEdge(s,d)
        #add extime
        e = copy.deepcopy(self.extime)
        for key,value in e.items():
            u = key + startNodeID
            self.addExtime(u,value)

        #add tensor
        t = copy.deepcopy(self.tensor)
        for key,value in t.items():
            u = key + startNodeID
            self.addTensor(u,value)

        #add weight
        w = copy.deepcopy(self.weight)
        for key,value in w.items():
            u = key + startNodeID
            self.addWeight(u,value)

        #add layer type
        l = copy.deepcopy(self.ltype)
        for key,value in l.items():
            u = key + startNodeID
            self.addLtype(u,value)

        return startNodeID

    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def getAllPathUtil(self, u, d, visited, path):
        self.path_cnt += 1

        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            rpath = copy.deepcopy(path)
            self.paths.append(rpath)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                #if visited[i]== False:
                if visited[i]== False and self.path_cnt < 100:
                    self.getAllPathUtil(i, d, visited, path)
					
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False


    # Prints all paths from 's' to 'd'
    def getAllPaths(self, s, d):

        # reset the paths
        self.paths = list()

        # Mark all the vertices as not visited
        visited =[False]*(self.V)

        # Create an array to store paths
        path = []

        # Call the recursive helper function to print all paths
        self.getAllPathUtil(s, d, visited, path)
        return self.paths
