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

    def read_data(self, argflag, spect, fitsdict):
        '''
        Loads up data to be used in the reduction.

        Parameters
        ----------
        argflag : dict
          Arguments and flags used for reduction
        spect : dict
          Properties of the spectrograph.
        fitsdict : dict
          Contains relevant information from fits header files
        '''
        # slice data and hold it
        pass
        
def make_slits(self, slitmask, left_traces, right_traces):
    '''
    Constructs slit objects with the measured edge traces.

    Parameters
    ----------
    slitmask : arslitmask.Slitmask instance
    left_traces : float array of left (lower) edge in pixel values
    right_traces : float array of right (upper) edge in pixel values

    Returns
    -------
    slits : a list of Slit instances
    '''
    trace_idx, slit_idx = slitmask.match_traces(left_traces, right_traces)
    # reject alignment boxes at some point
    isalign = slitmask.slits[slit_idx].isalign
    names = slitmask.slits[slit_idx].name
    lefts = left_edges[trace_idx]
    rights = right_edges[trace_idx]
    return [Slit(lefts[i], right[i], name[i]) for i in range(len(trace_idx))]

    
