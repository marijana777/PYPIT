'''
Module provides a slit class for passing to subprocesses for reduction.
'''

from __future__ import (absolute_import, division, print_function)

class Slit(object):
    '''
    A Slit consists of a trace as well as book-keeping from either the slitmask
    design or the reduction procedure.
    '''

    def __init__(self, left_edges, right_edges, name):
        '''
        Parameters
        ----------
        left_edges : array of pixel values corresponding to the left (lower) 
                     edge of the slit, inclusive
        right_edges : array of pixel values corresponding to the right (upper)
                      edge of the slit, exclusive
        name : str, name of slit
        '''
        self.left_edges = left_edges
        self.right_edges = right_edges
        self.name = name

    def __repr__(self):
        return '<' + type(self).__name__ + ': ' + self.name + '>'

    
