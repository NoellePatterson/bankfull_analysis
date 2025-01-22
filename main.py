"""
Bankfull Intersection Identification

This script find the intersection of transect lines with the previously-identified bankfull footprint for data on the Eel river.

Noelle Patterson, USU 
September 2024
"""

import os
import geopandas as gpd
import pandas as pd
import rasterio
import raster_footprint
import ast
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
import numpy as np
from matplotlib import pyplot as plt
import dataretrieval.nwis as nwis
from datetime import datetime
import sys
import pdb
from analysis import id_benchmark_bankfull, calc_dwdh, calc_derivatives, calc_derivatives_aggregate, recurrence_interval
from visualization import plot_bankfull_increments, plot_longitudinal_bf, multi_panel_plot

reach_name = 'Scotia' # Choose 'Leggett' or 'Miranda' or 'Scotia'

# Steps for bankfull analysis:
# 1. Identify benchmark bankfull using inundation rasters (Analysis.py -> id_benchmark_bankfull)
# 2. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
# 3. Calculate first and second order derivatives of the channel widths to identify topographic bankfull (Analysis.py -> calc_derivatives)
# 4. Post-processing: plot results (Visualization.py -> plot_bankfull_increments, plot_longitudinal_bf)

# Assign run parameters based on reach name
if reach_name == 'Leggett': 
    transect_fp = 'GIS/data_inputs/Leggett/XS_Sections/Thalweg_10m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Leggett/Bankfull_raster/SFE_Leggett_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Leggett/1m_Topobathy/dem.tif'
    flow_record = nwis.get_record(sites='11475800', service='dv', start='1900-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    plot_ylim = [220, 250]

elif reach_name == 'Miranda':
    transect_fp = 'GIS/data_inputs/Miranda/XS_Sections/Thalweg_10m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Miranda/Bankfull_raster/SFE_Miranda_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Miranda/1m_Topobathy/dem.tif'
    flow_record = nwis.get_record(sites='11476500', service='dv', start='1900-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    plot_ylim = [60, 100]

elif reach_name == 'Scotia':
    transect_fp = 'GIS/data_inputs/Scotia/XS_Sections/Thalweg_15m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Scotia/Bankfull_raster/SFE_Scotia_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Scotia/1m_Topobathy/dem.tif'
    flow_record = nwis.get_record(sites='11477000', service='dv', start='1900-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    plot_ylim = [10, 42]

else:
    print('Please select one of the following Eel River reaches: Leggett, Miranda, or Scotia')
    sys.exit()

# Create output folders if needed
if not os.path.exists('data/data_outputs/{}'.format(reach_name)):
    os.makedirs('data/data_outputs/{}'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/derivative_plots'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/derivative_plots'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/transect_plots'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/transect_plots'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/first_order_roc'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/first_order_roc'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/second_order_roc'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/second_order_roc'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/all_widths'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/all_widths'.format(reach_name))

# Upload test data: transects, stations, and bankfull raster 
transects = gpd.read_file(transect_fp)
bankfull = rasterio.open(bankfull_fp)
dem = rasterio.open(dem_fp)

# Convert bankfull raster into a footprint line object
bankfull_footprint = raster_footprint.footprint_from_rasterio_reader(bankfull, destination_crs = bankfull.crs)
bankfull_footprint = shape(bankfull_footprint)
bankfull_boundary = bankfull_footprint.boundary
bankfull_boundary = gpd.GeoDataFrame({'geometry':[bankfull_boundary]}, crs=bankfull.crs)

plot_interval = 1 # set plotting interval along transect in units of meters
d_interval = 10/100 # Set intervals to step up in depth (in units meters). 10cm intervals
slope_window = 5 # Set window size for calculating slope for derivatives

# Uncomment functions to run

output = id_benchmark_bankfull(reach_name, transects, dem, d_interval, bankfull_boundary, plot_interval)
all_widths_df, bankfull_width = calc_dwdh(reach_name, transects, dem, plot_interval, d_interval)
print('Dwdh calc done!!')
topo_bankfull, topo_bankfull_detrend = calc_derivatives(reach_name, d_interval, all_widths_df, slope_window)
print('Derivatives calc done!!')
output = calc_derivatives_aggregate(reach_name, d_interval, all_widths_df, slope_window)
# output = recurrence_interval(flow_record, bankfull_results)

# Plotting functions
# output = plot_longitudinal_bf(reach_name)
output = plot_bankfull_increments(reach_name, d_interval, plot_ylim)
# output = multi_panel_plot(reach_name, transects, dem, plot_interval, d_interval, bankfull_boundary)