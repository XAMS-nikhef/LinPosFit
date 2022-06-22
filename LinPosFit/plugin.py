import sys

from .LinPosFit import *
from .Clustering import *


import strax
import numpy as np
import numba
import straxen


@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsLinFit(strax.Plugin):
    """
    Computes S2 position with linearized log-likelihood fit. 
    Returns xy position and the parameters of the fit.
    """
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('xml',np.float),
            ('yml',np.float),
            ('r0',np.float), 
            ('gamma', np.float),
            ('logl',np.float), 
            ('n',np.int)]
    dtype += strax.time_fields
   
    def setup(self,):
        inch = 2.54 # cm
        pmt_to_lxe = 7.0    
        pmt_pos = straxen.pmt_positions()
        self.pmt_pos = list(zip(pmt_pos['x'].values,pmt_pos['y'].values,np.repeat(pmt_to_lxe, self.config['n_top_pmts'])))
        self.pmt_surface=(3*inch)**2*np.pi/4.

    def compute(self,peaks):
        res = np.zeros(len(peaks),self.dtype)
        
        for ix, p in enumerate(peaks):
            if p['type']!=2:
                #Only reconstruct s2 peaks. We do need to set the time of the peaks
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue
            try:
                #Some really small single electron s2s fail the minimization
                fit_result,_,_ = lpf_execute(self.pmt_pos[:self.config['n_top_pmts']],p['area_per_channel'][:self.config['n_top_pmts']],self.pmt_surface)
                
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                res[ix]['xml'] = fit_result[0]
                res[ix]['yml'] = fit_result[1]
                res[ix]['r0'] = fit_result[2]
                res[ix]['logl'] = fit_result[3]
                res[ix]['n'] = fit_result[4]
                res[ix]['gamma'] = fit_result[5]
                
            except:
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue

        return res

    

@strax.takes_config(strax.Option('n_top_pmts'))
class PeakClustering(strax.Plugin):
    """
    Work in progress. 
    Returns the number of clusters in a peak
    """
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('n_cluster',np.int)]
    dtype += strax.time_fields
    __version__="0.0.4"
   
    def compute(self,peaks):
        res = np.zeros(len(peaks),self.dtype)
        
        for ix, p in enumerate(peaks):
            
            res[ix]['time'] = p['time']
            res[ix]['endtime'] = p['endtime']
            
            if p['type']!=2:
                #Only reconstruct s2 peaks. We do need to set the time of the peaks
                continue
            try: # sometimes it fails
                # Compute optimal number of clusters
                n_cluster = new_cluster_execute(p['area_per_channel'][:self.config['n_top_pmts']])
                res[ix]['n_cluster'] = n_cluster
            except:
                continue

        return res
    
  
    

@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsInitializeLinFit(strax.Plugin):
    """
    Computes the initial xy position used in LinPosFit. 
    It's (when possible) the center of gravity of the hits in the PMTs. 
    """
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('x_00',np.float),
            ('y_00',np.float)]
    dtype += strax.time_fields
    __version__="0.1.0"

    def setup(self,):
        inch = 2.54 # cm
        pmt_to_lxe = 7.0    
        pmt_pos = straxen.pmt_positions()
        self.pmt_pos = list(zip(pmt_pos['x'].values,pmt_pos['y'].values,np.repeat(pmt_to_lxe, self.config['n_top_pmts'])))
        self.pmt_surface=(3*inch)**2*np.pi/4.

    def compute(self,peaks):
        res = np.zeros(len(peaks),self.dtype)
        
        for ix, p in enumerate(peaks):
            if p['type']!=2:
                #Only reconstruct s2 peaks. We do need to set the time of the peaks
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue
            try:
            #Some really small single electron s2s fail the minimization
            

                xhit = np.array(self.pmt_pos[:self.config['n_top_pmts']])
                nhit = np.array(p['area_per_channel'][:self.config['n_top_pmts']])

                fit_result,_ = lpf_initialize(xhit, nhit)

                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']

                res[ix]['x_00'] = fit_result[0]
                res[ix]['y_00'] = fit_result[1]

            except:
                
                res[ix]['x_00'] *= float('nan')
                res[ix]['y_00'] *= float('nan')
                
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue
            
        return res

    
class EventPositionsLinFit(strax.LoopPlugin):
    """
    Loop over events and find the S2 peak with greatest area within each of those events.
    """
    __version__="1.0.15"
    
    depends_on = 'event_info', 'peak_basics', 'peak_positions_lin_fit' 
    data_kind = 'events'
    
    rechunk_on_save = True    # Saver is not allowed to rechunk
    time_selection = 'fully_contained' # other option is 'touching'
    loop_over = 'events'
    
    def infer_dtype(self):
        
        return self.deps['peak_positions_lin_fit'].dtype  
    
    # Use the compute_loop() instead of compute()
    def compute_loop(self, events, peaks):
        
        dt = self.deps['peak_positions_lin_fit'].dtype
        dt = np.dtype(strax.utils.remove_titles_from_dtype(dt))
        
        result = {}
        s2s = peaks[peaks['type']==2]
        if len(s2s) > 0:
            arg = np.argmax(s2s[s2s['type']==2]['area'])
            for t in dt.names:
                result[t] = s2s[arg][t]
        else:
            result['time'] = peaks[0]['time']
            result['endtime'] = peaks[0]['endtime']
            
        return result 
    
   
