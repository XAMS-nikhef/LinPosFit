import strax
import numpy as np
import straxen

from .LinPosFit import *

@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsLinFit(strax.Plugin):
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
            fit_result,_,_ = lpf_execute(self.pmt_pos[:self.config['n_top_pmts']],p['area_per_channel'][:self.config['n_top_pmts']],self.pmt_surface)

            res[ix]['time'] = p['time']
            res[ix]['endtime'] = p['endtime']
            res[ix]['xml'] = fit_result[0]
            res[ix]['yml'] = fit_result[1]
            res[ix]['r0'] = fit_result[2]
            res[ix]['logl'] = fit_result[3]
            res[ix]['n'] = fit_result[4]
            res[ix]['gamma'] = fit_result[5]
        
        return res
