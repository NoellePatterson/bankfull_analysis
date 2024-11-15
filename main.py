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
import sys
import pdb
from analysis import calc_dwdh, calc_derivatives, calc_derivatives_aggregate
from visualization import plot_bankfull, plot_bankfull_increments, plot_longitudinal_bf

reach_name = 'Scotia' # Choose 'Leggett' or 'Miranda' or 'Scotia'

# Assign run parameters based on reach name
if reach_name == 'Leggett': 
    transect_fp = 'GIS/data_inputs/Leggett/XS_Sections/Thalweg_10m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Leggett/Bankfull_raster/SFE_Leggett_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Leggett/1m_Topobathy/dem.tif'
    median_bankfull = 227.474 # model-derived reach-averaged bankfull
    median_topo_bankfull = 225.7 # topography-derived reach-averaged bankfull
    modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
    topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))
    plot_ylim = [220, 270]

elif reach_name == 'Miranda':
    transect_fp = 'GIS/data_inputs/Miranda/XS_Sections/Thalweg_10m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Miranda/Bankfull_raster/SFE_Miranda_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Miranda/1m_Topobathy/dem.tif'
    median_bankfull = 71.436 # model-derived reach-averaged bankfull
    median_topo_bankfull = 68.0 # topography-derived reach-averaged bankfull
    modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
    topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))
    plot_ylim = [60, 100]

elif reach_name == 'Scotia':
    transect_fp = 'GIS/data_inputs/Scotia/XS_Sections/Thalweg_15m_adjusted.shp'
    bankfull_fp = 'GIS/data_inputs/Scotia/Bankfull_raster/SFE_Scotia_011_d_010_00.tif' # preprocessing required to convert native .flt to .tif
    dem_fp = 'GIS/data_inputs/Scotia/1m_Topobathy/dem.tif'
    median_bankfull = 21.206 # model-derived reach-averaged bankfull
    median_topo_bankfull = 14.6 # topography-derived reach-averaged bankfull
    modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
    topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))

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

# Uncomment functions to run
all_widths_df = calc_dwdh(reach_name, transects, dem, plot_interval, d_interval)
print('Dwdh calc done!!')
# output = calc_derivatives(reach_name, d_interval, all_widths_df)
# print('Derivatives calc done!!')
# output = calc_derivatives_aggregate(reach_name, d_interval, all_widths_df)

# output = plot_bankfull(reach_name, transects, dem, d_interval, bankfull_boundary, plot_interval, topo_bankfull_transects_df, plot_ylim=None)
# output = plot_longitudinal_bf(reach_name, modeled_bankfull_transects_df, topo_bankfull_transects_df, median_bankfull, median_topo_bankfull)
output = plot_bankfull_increments(reach_name, all_widths_df, d_interval, topo_bankfull_transects_df, median_bankfull, median_topo_bankfull, plot_ylim=None)
