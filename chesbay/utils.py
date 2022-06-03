from __future__ import annotations

import contextlib
import importlib
import logging
import os
import pathlib
import sys

import numpy as np 
import pandas as pd
import xarray as xr
import geoviews as gv
import holoviews as hv
from holoviews.operation.datashader import dynspread, rasterize
from pyproj import Transformer

hv.extension("bokeh")

logger = logging.getLogger(__name__)

def read_dataset(fname,method=0,dataset_format="netCDF",prj='epsg:4326',
                 time=None,variable=None,layer=None,node=None):
    '''
    function to read information of dataset    
    Inputs:
       fname:    path of dataset file, or file handle (xr.Dataset)
       method:   different ways to read dataset
                 0: open dataset; 1: read header information; 
                 2: read dataset snapshot; 3: read time series 
       dataset_format: format of dataset 
       prj:      projection of dataset coordinate
       time:     timestamp or index of timestamp for dataset snapshot
       variable: variable to be read
       layer:    layer for 3D variables 
       node:     index of node for reading time series (for method=3)

    note: only SCHISM format is defined so far
    '''

    #open dataset
    if isinstance(fname, xr.Dataset): #already open
       ds=fname

    else:
       if fname.endswith(".nc"):
           ds = xr.open_dataset(fname, mask_and_scale=True)
       elif fname.endswith(".zarr") or fname.endswith(".zip"):
           ds = xr.open_dataset(fname, mask_and_scale=True, engine="zarr")
       else:
           raise ValueError(f"unknown format of dataset: {fname}")

       #rename variables
       if "elev" in ds.variables:
           ds = ds.rename({'elev':'elevation'})
       if "salt" in ds.variables:
           ds = ds.rename({'salt':'salinity'})
       if "temp" in ds.variables:
           ds = ds.rename({'temp':'temperature'})
       if "ICM_1" in ds.variables:
           ds = ds.rename({'ICM_1': 'Zooplankton1'})
       if "ICM_2" in ds.variables:
           ds = ds.rename({'ICM_2': 'Zooplankton2'})
       if "ICM_3" in ds.variables:
           ds = ds.rename({'ICM_3': 'Diatom'})
       if "ICM_4" in ds.variables:
           ds = ds.rename({'ICM_4': 'Green Algae'})
       if "ICM_5" in ds.variables:
           ds = ds.rename({'ICM_5': 'Cyanobacteria'})
       if "ICM_6" in ds.variables:
           ds = ds.rename({'ICM_6': 'Refractory Particulate Organic Carbon'})
       if "ICM_7" in ds.variables:
           ds = ds.rename({'ICM_7': 'Labile Particulate Organic Nitrogen'})
       if "ICM_8" in ds.variables:
           ds = ds.rename({'ICM_8': 'Dissolved Organic Carbon'})
       if "ICM_9" in ds.variables:
           ds = ds.rename({'ICM_9': 'Refractory Particulate Organic Nitrigon'})
       if "ICM_10" in ds.variables:
           ds = ds.rename({'ICM_10': 'Labile Particulate Organic Nitrigon'})
       if "ICM_11" in ds.variables:
           ds = ds.rename({'ICM_11': 'Dissolved Organic Nitrogen'})
       if "ICM_12" in ds.variables:
           ds = ds.rename({'ICM_12': 'Ammonium Nitrogen'})
       if "ICM_13" in ds.variables:
           ds = ds.rename({'ICM_13': 'Nitrate Nitrogen'}) 
       if "ICM_14" in ds.variables:
           ds = ds.rename({'ICM_14': 'Refractory Particulate Organic Phosphorus'}) 
       if "ICM_15" in ds.variables:
           ds = ds.rename({'ICM_15': 'Labile Particulate Organic Phosphorus'}) 
       if "ICM_16" in ds.variables:
           ds = ds.rename({'ICM_16': 'Dissolved Organic Phosphorus'}) 
       if "ICM_17" in ds.variables:
           ds = ds.rename({'ICM_17': 'Total Phosphate'}) 
       if "ICM_18" in ds.variables:
           ds = ds.rename({'ICM_18': 'Particulate Biogenic Silica'}) 
       if "ICM_19" in ds.variables:
           ds = ds.rename({'ICM_19': 'Available Silica'}) 
       if "ICM_20" in ds.variables:
           ds = ds.rename({'ICM_20': 'Chemical Oxygen Demand'}) 
       if "ICM_21" in ds.variables:
           ds = ds.rename({'ICM_21': 'Dissolved Oxygen'}) 
       if "ICM_22" in ds.variables:
           ds = ds.rename({'ICM_22': 'Total Inorganic Carbon'}) 
       if "ICM_23" in ds.variables:
           ds = ds.rename({'ICM_23': 'Alkalinity'}) 
       if "ICM_24" in ds.variables:
           ds = ds.rename({'ICM_24': 'Dissloved Calcium'}) 
       if "ICM_25" in ds.variables:
           ds = ds.rename({'ICM_25': 'Calcium Carbonate'}) 

       if method==0: 
          return ds

    #read dataset
    if dataset_format=="netCDF":
       if method==1: #extract header information of dataset
          #variables to be hidden from user
          hvars=[
              'time', 'SCHISM_hgrid', 'SCHISM_hgrid_face_nodes', 'SCHISM_hgrid_edge_nodes', 
              'SCHISM_hgrid_node_x', 'SCHISM_hgrid_node_y', 'node_bottom_index', 'SCHISM_hgrid_face_x',
              'SCHISM_hgrid_face_y', 'ele_bottom_index', 'SCHISM_hgrid_edge_x', 'SCHISM_hgrid_edge_y',
              'edge_bottom_index', 'depth', 'sigma', 'dry_value_flag', 'coordinate_system_flag', 
              'minimum_depth', 'sigma_h_c', 'sigma_theta_b', 'sigma_theta_f', 'sigma_maxdepth', 'Cs', 
              'wetdry_node', 'wetdry_elem', 'wetdry_side', 'zcor'] 

          times=ds['time'].to_pandas().dt.to_pydatetime()
          variables=[i for i in ds.variables if (i not in hvars)]
          x=ds.variables['SCHISM_hgrid_node_x'].values
          y=ds.variables['SCHISM_hgrid_node_y'].values
          z = ds.variables['depth'].values
          elnode=ds.variables['SCHISM_hgrid_face_nodes'].values

          #split quads          
          if elnode.shape[1]==4:
             eid=np.nonzero(~((np.isnan(elnode[:,-1]))|(elnode[:,-1]<0)))[0]
             elnode=np.r_[elnode[:,:3],np.c_[elnode[eid,0][:,None], elnode[eid,2:]]]
          if elnode.max()>=len(x):
             elnode=elnode-1
          elnode=elnode.astype('int')

          return ds, times, variables, x, y, z, elnode

       elif method==2: #extract one snapshot of dataset
          #time index
          if isinstance(time,int): 
             tid=time
          else:
             times=ds['time'].to_pandas().dt.to_pydatetime()
             tid=np.nonzero(np.array(times)==timestamp)[0][0]


          #2D and 3D variables
          if ds.variables[variable].ndim==1:
             mdata=ds.variables[variable].values
          elif ds.variables[variable].ndim==2:
             mdata=ds.variables[variable][tid].values
          elif ds.variables[variable].ndim==3:
             if layer=='surface':
                mdata=ds.variables[variable][tid,:,-1].values
             elif layer=='bottom':
                if 'node_bottom_index' in [*ds.variables]:
                   zid=ds.variables['node_bottom_index'][:].values.astype('int')
                   pid=np.arange(len(zid))
                   mdata=ds.variables[variable][tid].values[pid,zid]
                else:
                   mdata=ds.variables[variable][tid,:,0].values
             else:
                mdata=ds.variables[variable][tid,:,layer].values

          return mdata

       elif method==3: #extract time series
          if ds.variables[variable].ndim==2:
             mdata=ds.variables[variable][:,node].values
          elif ds.variables[variable].ndim==3:
             if layer=='surface':
                zid=-1
             elif layer=='bottom':
                if 'node_bottom_index' in [*ds.variables]:
                   zid=ds.variables['node_bottom_index'][node].values.astype('int')
                else:
                   zid=0
             else:
                zid=layer
             mdata=ds.variables[variable][:,node,zid].values

          return mdata 

       elif method == 4: #get 2D (time, node)
          if ds.variables[variable].ndim == 2:
             mdata = ds.variables[variable]
          elif ds.variables[variable].ndim == 3:
             if layer == 'surface':
                zid = -1
                mdata = ds.variables[variable][:, :, zid]
             elif layer == 'bottom':
                if 'node_bottom_index' in [*ds.variables]:
                   zid=ds.variables['node_bottom_index'][:].values.astype('int')
                   pid=np.arange(len(zid))
                   mdata = ds.variables[variable].values[:, pid, zid]
                else:
                   zid = 0
                   mdata = ds.variables[variable][:, :, zid]
             else:
                zid = layer
                mdata = ds.variables[variable][:, :, zid]

          return mdata, ds.time.values
       else:
        raise ValueError(f"unknown read method for SCHISM model: {method}")
    else:
        raise ValueError(f"unknown model (read method needs to be defined): {dataset_format}")

def can_be_opened_by_xarray(path):
    try:
        read_dataset(path)
    except ValueError:
        logger.debug("path cannot be opened by xarray: %s", path)
        return False
    else:
        return True

def get_tiles() -> gv.Tiles:
    #tiles = gv.WMTS("http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png")
    tiles = gv.WMTS("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.jpg")
    return tiles

def plot_grid(ds: xr.Dataset, prj=None, grid=True):

    ds,times,variables,x,y,z, elnode = read_dataset(ds, 1)
    x, y = Transformer.from_crs(prj, 'epsg:4326', always_xy=True).transform(x, y)

    df=pd.DataFrame({'longitude': x, 'latitude': y, 'depth': z})
    pdf=gv.Points(df,kdims=['longitude','latitude'], vdims='depth')
    trimesh=gv.TriMesh((elnode, pdf))
    tile = get_tiles()
    mesh = dynspread(rasterize(trimesh.edgepaths, precompute=True)).opts(height=300, active_tools=["pan", "wheel_zoom"])
    depthmap=rasterize(trimesh, precompute=True).opts(
        title=f"Bathymetry",
        colorbar=True,
        clabel='m',
        cmap="jet",
        show_legend=True,
        )
    if grid:
        return tile * depthmap * mesh
    else:
        return mesh
