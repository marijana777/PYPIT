'''
Module for multiprocessing of slit reductions.

Lifted gratuitously from the interruptable pool as described originally by jreese:
https://github.com/jreese/multiprocessing-keyboardinterrupt

and implemented in emcee by pkgw
https://github.com/dfm/emcee/blob/master/emcee/interruptible_pool.py
'''

from __future__ import (print_function, absolute_import, division, unicode_literals)

import signal
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class InterruptiblePool(Pool):
    '''
    A subclass of multiprocessing.pool.Pool with better support for keyboard 
    interruptions.
    '''
    
    wait_timeout = 3600

    def __init__(self, nworkers=None, **kwargs):
        '''
        Parameters
        ----------
        nworkers : int, number of processes to spawn, default to cpu_count
        '''
        super(InterruptiblePool, self).__init__(nworkers, init_worker, **kwargs)

    def map(self, func, iterable, chunksize=None):
        '''
        Multiprocessing map that passes on interrupts to the parent process.

        Parameters
        ----------
        func : function to be mapped
        iterable : an iterable with which the function will be applied
        chunksize : int, the number of items to send to one process as a single task
        '''
        # The key magic is that we must call r.get() with a timeout, because
        # a Condition.wait() without a timeout swallows KeyboardInterrupts.
        r = self.map_async(func, iterable, chunksize)

        while True:
            try:
                return r.get(self.wait_timeout)
            except TimeoutError:
                pass
            except KeyboardInterrupt:
                self.terminate()
                self.join()
                raise    

class function_wrapper(object):
    '''
    Lifted gratuitously from emcee.

    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    '''
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print('Exception while calling ' + self.f.__name__ + ' in ' +
                  self.f.__module__)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise
