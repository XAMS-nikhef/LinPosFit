import sys

from .LinPosFit import *
from .Clustering import *


import strax
import numpy as np
import numba
import straxen
from warnings import warn
from straxen.plugins.position_reconstruction import DEFAULT_POSREC_ALGO_OPTION
from straxen.common import pax_file, get_resource, first_sr1_run, pre_apply_function
from straxen.get_corrections import get_correction_from_cmt, get_cmt_resource, is_cmt_option
from straxen.itp_map import InterpolatingMap
export, __all__ = strax.exporter()


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
class PeakPositionsLinFitFactorOne(strax.Plugin):
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('xml_factor1',np.float),
            ('yml_factor1',np.float),
            ('r0_factor1',np.float), 
            ('gamma_factor1', np.float),
            ('logl_factor1',np.float), 
            ('n_factor1',np.int)]
    dtype += strax.time_fields
    __version__="0.0.3"

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
                
                nhit = p['area_per_channel'][:self.config['n_top_pmts']] 
                y = []
                for x in nhit: 
                    if x > 4000:
                        x = 4000
                    y.append(1 - x * .000125)

                nhit = nhit*y
                fit_result,_,_ = lpf_execute(self.pmt_pos[:self.config['n_top_pmts']],nhit,self.pmt_surface)
                
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                res[ix]['xml_factor1'] = fit_result[0]
                res[ix]['yml_factor1'] = fit_result[1]
                res[ix]['r0_factor1'] = fit_result[2]
                res[ix]['logl_factor1'] = fit_result[3]
                res[ix]['n_factor1'] = fit_result[4]
                res[ix]['gamma_factor1'] = fit_result[5]
                
            except:
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue

        return res
    
@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsLinFitFactorTwo(strax.Plugin):
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('xml_factor2',np.float),
            ('yml_factor2',np.float),
            ('r0_factor2',np.float), 
            ('gamma_factor2', np.float),
            ('logl_factor2',np.float), 
            ('n_factor2',np.int)]
    dtype += strax.time_fields
    __version__="0.0.3"

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
                
                nhit = p['area_per_channel'][:self.config['n_top_pmts']] 
                y = []
                for x in nhit: 
                    if x > 4000:
                        x = 4000
                    y.append(1 - x * 0.25/4000)

                nhit = nhit*y
                fit_result,_,_ = lpf_execute(self.pmt_pos[:self.config['n_top_pmts']],nhit,self.pmt_surface)
                
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                res[ix]['xml_factor2'] = fit_result[0]
                res[ix]['yml_factor2'] = fit_result[1]
                res[ix]['r0_factor2'] = fit_result[2]
                res[ix]['logl_factor2'] = fit_result[3]
                res[ix]['n_factor2'] = fit_result[4]
                res[ix]['gamma_factor2'] = fit_result[5]
                
            except:
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue

        return res

    
    



    
@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsLinFitFactorThree(strax.Plugin):
    depends_on=('peaks','peak_basics')
    rechunk_on_save=False
    dtype= [('xml_factor3',np.float),
            ('yml_factor3',np.float),
            ('r0_factor3',np.float), 
            ('gamma_factor3', np.float),
            ('logl_factor3',np.float), 
            ('n_factor3',np.int)]
    dtype += strax.time_fields
    __version__="0.0.3"

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
                
                nhit = p['area_per_channel'][:self.config['n_top_pmts']] 
                y = []
                for x in nhit: 
                    if x > 3000:
                        x = 3000
                    y.append(1 - x * 0.25/1400)

                nhit = nhit*y
                fit_result,_,_ = lpf_execute(self.pmt_pos[:self.config['n_top_pmts']],nhit,self.pmt_surface)
                
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                res[ix]['xml_factor3'] = fit_result[0]
                res[ix]['yml_factor3'] = fit_result[1]
                res[ix]['r0_factor3'] = fit_result[2]
                res[ix]['logl_factor3'] = fit_result[3]
                res[ix]['n_factor3'] = fit_result[4]
                res[ix]['gamma_factor3'] = fit_result[5]
                
            except:
                res[ix]['time'] = p['time']
                res[ix]['endtime'] = p['endtime']
                continue

        return res

    

@strax.takes_config(
    strax.Option('n_top_pmts'),
    strax.Option('pmt_to_lxe',default=7,help='Distance between the PMTs and the liquid interface'))
class PeakPositionsInitializeLinFit(strax.Plugin):
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
    """Loop over events and find the peaks within each of those events."""
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

    
@export    
@strax.takes_config(
    strax.Option(
        name='electron_drift_velocity',
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
        default=("electron_drift_velocity", "ONLINE", True)
    ),
    strax.Option(
        name='electron_drift_time_gate',
        help='Electron drift time from the gate in ns',
        default=("electron_drift_time_gate", "ONLINE", True)
    ),
    strax.Option(
        name='fdc_map',
        help='3D field distortion correction map path',
        default_by_run=[
            (0, pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz')),  # noqa
            (first_sr1_run, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz')), # noqa
            (170411_0611, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part2_v1.json.gz')), # noqa
            (170704_0556, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part3_v1.json.gz')), # noqa
            (170925_0622, pax_file('XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part4_v1.json.gz'))], # noqa
    ),
    *DEFAULT_POSREC_ALGO_OPTION
)
class EventPositionsLinFitCorr(strax.Plugin):
    """
    Computes the observed and corrected position for the main S1/S2
    pairs in an event. For XENONnT data, it returns the FDC corrected
    positions of the default_reconstruction_algorithm. In case the fdc_map
    is given as a file (not through CMT), then the coordinate system
    should be given as (x, y, z), not (x, y, drift_time).
    """

    depends_on = ('event_info', 'event_positions_lin_fit')
    
    __version__ = '0.0.1'

    dtype = [
        ('xml_corr', np.float32,
         'Interaction x-position, field-distortion corrected (cm)'),
        ('yml_corr', np.float32,
         'Interaction y-position, field-distortion corrected (cm)'),
        ('rml_corr', np.float32,
         'Interaction radial position, field-distortion corrected (cm)'),
        ('rml_field_distortion_correction', np.float32,
         'Correction added to r_naive for field distortion (cm)')
            ] + strax.time_fields

    def setup(self):

        self.electron_drift_velocity = get_correction_from_cmt(self.run_id, self.config['electron_drift_velocity'])
        self.electron_drift_time_gate = get_correction_from_cmt(self.run_id, self.config['electron_drift_time_gate'])
        
        if isinstance(self.config['fdc_map'], str):
            self.map = InterpolatingMap(
                get_resource(self.config['fdc_map'], fmt='binary'))

        elif is_cmt_option(self.config['fdc_map']):
            self.map = InterpolatingMap(
                get_cmt_resource(self.run_id,
                                 tuple(['suffix',
                                        self.config['default_reconstruction_algorithm'],
                                        *self.config['fdc_map']]),
                                 fmt='binary'))
            self.map.scale_coordinates([1., 1., - self.electron_drift_velocity])

        else:
            raise NotImplementedError('FDC map format not understood.')

    def compute(self, events):

        result = {'time': events['time'],
                  'endtime': strax.endtime(events)}
        
        z_obs = - self.electron_drift_velocity * (events['drift_time'] - self.electron_drift_time_gate)
        orig_pos = np.vstack([events[f'xml'], events[f'yml'], z_obs]).T
        r_obs = np.linalg.norm(orig_pos[:, :2], axis=1)
        delta_r = self.map(orig_pos)

        # apply radial correction
        with np.errstate(invalid='ignore', divide='ignore'):
            r_cor = r_obs + delta_r
            scale = r_cor / r_obs

        # z correction due to longer drift time for distortion
        # (geometrical reasoning not valid if |delta_r| > |z_obs|,
        #  as cathetus cannot be longer than hypothenuse)
        with np.errstate(invalid='ignore'):
            z_cor = -(z_obs ** 2 - delta_r ** 2) ** 0.5
            invalid = np.abs(z_obs) < np.abs(delta_r)
            # do not apply z correction above gate
            invalid |= z_obs >= 0
        z_cor[invalid] = z_obs[invalid]
        delta_z = z_cor - z_obs

        result.update({'xml_corr': orig_pos[:, 0] * scale,
                       'yml_corr': orig_pos[:, 1] * scale,
                       'rml_corr': r_cor,
                       'rml_field_distortion_correction': delta_r,
                       })

        return result
