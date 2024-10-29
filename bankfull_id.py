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
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
import numpy as np
from matplotlib import pyplot as plt
import pdb

# Run parameters 
transect_fp = 'GIS/data_inputs/Leggett/XS_Sections/Thalweg_10m_adjusted.shp'
bankfull_fp = 'GIS/data_inputs/Leggett/Bankfull_raster/SFE_Leggett_011_d_Max.tif' # preprocessing required to convert native .flt to .tif
dem_fp = 'GIS/data_inputs/Leggett/1m_Topobathy/dem.tif'
reach_name = 'Leggett' # specify reach of the Eel River

# transect_fp = 'GIS/data_inputs/Miranda/XS_Sections/Thalweg_10m_adjusted.shp'
# bankfull_fp = 'GIS/data_inputs/Miranda/Bankfull_raster/SFE_Miranda_001_d_Max.tif' # preprocessing required to convert native .flt to .tif
# dem_fp = 'GIS/data_inputs/Miranda/1m_Topobathy/dem.tif'
# reach_name = 'Miranda' # specify reach of the Eel River

# transect_fp = 'GIS/data_inputs/Scotia/XS_Sections/Thalweg_15m_adjusted.shp'
# bankfull_fp = 'GIS/data_inputs/Scotia/Bankfull_raster/SFE_Scotia_011_d_Max.tif' # preprocessing required to convert native .flt to .tif
# dem_fp = 'GIS/data_inputs/Scotia/1m_Topobathy/dem.tif'
# reach_name = 'Scotia' # specify reach of the Eel River

# Create output folders if needed
if not os.path.exists('data/data_outputs/{}'.format(reach_name)):
    os.makedirs('data/data_outputs/{}'.format(reach_name))
if not os.path.exists('data/data_outputs/{}/derivative_plots'.format(reach_name)):
    os.makedirs('data/data_outputs/{}/derivative_plots'.format(reach_name))

# Upload test data: transects, stations, and bankfull raster 
transects = gpd.read_file(transect_fp)
bankfull = rasterio.open(bankfull_fp)
dem = rasterio.open(dem_fp)

# Convert bankfull raster into a footprint line object
bankfull_footprint = raster_footprint.footprint_from_rasterio_reader(bankfull, destination_crs = bankfull.crs)
bankfull_footprint = shape(bankfull_footprint)
bankfull_boundary = bankfull_footprint.boundary
bankfull_boundary = gpd.GeoDataFrame({'geometry': [bankfull_boundary]}, crs=bankfull.crs)

interval = 1 # set plotting interval in units of meters

# transects = transects.iloc[246:,] # Memory capacity maybe reached, all Scotia plots will not generate in one push. 
def plot_bankfull():
    # For each transect, find intersection points with bankfull, and plot transects with intersections
    bankfull = []
    for index, row in transects.iterrows():
        fig_list = []
        line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=transects.crs)
        intersect_pts = line.geometry.intersection(bankfull_boundary)

        # Generate a spaced interval of stations along each transect for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], interval) 
        stations = row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        
        # Get Y and Z coordinates for bankfull intersections for plotting
        station_zero = stations['geometry'][0]
        station_zero_gpf = gpd.GeoDataFrame({'geometry':[station_zero]}, crs=transects.crs)
        station_zero_gpf = station_zero_gpf.set_geometry('geometry') # ensure geometry is set correctly
        
        if isinstance(intersect_pts.geometry[0], Point):
            print('Only one intersection point identified; skipping to next X section.')
            plt.close()
            continue
        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        # Be aware, Scotia transects size exceeds limits for figures saved in one folder (no warning issued)
        bankfull_z = list(dem.sample(coords))
        coord0 = Point(coords[0]) # separate out intersection coordinates so distance function works properly
        coord1 = Point(coords[1]) # separate out intersection coordinates so distance function works properly
        ydist0 = coord0.distance(station_zero_gpf)
        ydist1 = coord1.distance(station_zero_gpf)

        # Arrange points together for plotting
        station_z = []
        for i, value in enumerate(elevs):
            station_z.append(value)
        stations_plot_df = pd.DataFrame({'station_y':distances, 'station_z':station_z})
        bankfull_y_plot = [ydist0['geometry'][0], ydist1['geometry'][0]]
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]]
        bankfull_z_plot_avg = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies
        bankfull.append(bankfull_z_plot_avg)
        bankfull_plot_df = pd.DataFrame({'bankfull_y':bankfull_y_plot, 'bankfull_z':[bankfull_z_plot_avg, bankfull_z_plot_avg]})

        # Plot everything together
        fig, ax = plt.subplots()
        plt.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='black', linestyle='-', label='Transect')
        plt.plot(bankfull_plot_df['bankfull_y'], bankfull_plot_df['bankfull_z'],color='red', linestyle='-', label='Bankfull')
        # Create empty plot with blank marker containing bankfull label
        bankfull_label = str(round(bankfull_z_plot_avg, 2))
        plt.plot([], [], ' ', label="Bankfull elev={}m".format(bankfull_label))
        plt.xlabel('Meters')
        plt.ylabel('Elevation (Meters)')
        plt.title('Eel River at {}'.format(reach_name))
        plt.legend()
        # Can I save each plot as an object, add it to a list, and return that list from this function?
        fig_list.append(fig)
        # plt.savefig('data/data_outputs/{}/bankfull_transect_{}.jpeg'.format(reach_name, index))
        plt.close()
    print('Median bankfull is {}m'.format(np.nanmedian(bankfull)))
    return(fig_list)

def calc_dwdh(fig_list):
    # Loop through xsections and create dw/dh array for each xsection
    # df to store arrays of w and h
    all_widths_df = pd.DataFrame(columns=['widths'])
    incomplete_intersection_counter = 0
    total_measurements = 0
    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        wh_ls = []
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # # Strategy 1: normalize depths to thalweg
        # min_elevation = min(elevs)
        # normalized_elevs = elevs - min_elevation

        # Strategy 2: Base all depths on 0-elevation
        min_elevation = 0
        normalized_elevs = elevs

        d_interval = 10/100 # Set intervals to step up in depth (in units meters). 10cm intervals?
        # Determine total depth of iterations based on max rise on the lower bank
        min_z = min(normalized_elevs)
        min_y = list(normalized_elevs).index(min_z)
        max_left_bank = max(list(normalized_elevs)[0:min_y])
        max_right_bank = max(list(normalized_elevs)[min_y:])

        if max_right_bank < max_left_bank:
            max_depth = max_right_bank
        else:
            max_depth = max_left_bank
        depths = max_depth//d_interval
        # If depth is improperly assigned skip to next transect
        if depths[0] == float('inf'):
            continue
        
        # calc width at the current depth (use an additive approach for discontinuous WSE's)
        for index, depth in enumerate(range(int(depths[0]))):
            total_measurements += 1
            # find intercepts of current d with bed profile (as locs where normalized profile pts have a sign change)?
            wat_level = [x - (d_interval * index) for x in normalized_elevs]
            intercepts = []
            for i, val in enumerate(normalized_elevs[0:-1]):
                if np.sign(wat_level[i] * wat_level[i + 1]) < 0:
                    intercepts.append(distances[i])
            
            # Find distances between intercept points
            if len(intercepts) == 2: # most common case, one intercept on each side of channel
                width = intercepts[1] - intercepts[0]
            elif (len(intercepts) % 2) == 0: # other common case, bed elevation has at least one extra pair of intercepts
                partial_widths = []
                for int_index in range(0, len(intercepts), 2):
                    w = intercepts[int_index + 1] - intercepts[int_index]
                    partial_widths.append(w)
                width = sum(partial_widths)
            elif len(intercepts) == 3:
                left_bank = min(intercepts)
                right_bank = max(intercepts)
                width = right_bank - left_bank
            else:
                # print("Cannot accurately determine width with incomplete xsection intersection. Num intersections = {}.".format(len(intercepts)))
                width = np.nan
                incomplete_intersection_counter += 1 
            wh_ls.append(width)
        wh_ls = pd.DataFrame({'widths':[wh_ls], 'transect_id':transects_index})
        all_widths_df = pd.concat([all_widths_df, wh_ls], ignore_index=True)

    # calculate and plot second derivative of width (height is constant)
    ddw_ls = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        dw = []
        ddw = []
        for w_index, current_width in enumerate(xsection): # loop through all widths in current xsection
            if w_index < len(xsection) - 1: # can caluclate differences up to second to last index
                current_d = (xsection[w_index + 1] - current_width)/d_interval
                dw.append(current_d)
        for dw_index, current_dw in enumerate(dw): # loop through all first order rate changes to get second order change  (slope break)
            if dw_index < len(dw) - 1: # can calculate differences up to second to last first-order change
                current_dd = (dw[dw_index + 1] - current_dw)/d_interval
                ddw.append(current_dd)
        # Find max second derivative as bankfull
        ddw_abs = [abs(i) for i in ddw]
        max_ddw = np.nanmax(ddw_abs)
        max_ddw_index = ddw_abs.index(max_ddw)
        bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull
        # Plot x-section/first/second derivative plot and identified bankfull
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        x_section_xvals = get_x_vals(xsection)
        dw_xvals = get_x_vals(dw)
        ddw_xvals = get_x_vals(ddw)
        fig, ax = plt.subplots(3,1, figsize=(18, 9))
        plt.xlabel('Distance from 0-elevation (m)')
        ax[0].plot(x_section_xvals, xsection)
        ax[0].set_title('Cross-section')
        ax[1].plot(dw_xvals, dw)
        ax[1].set_title('First order rate of change')
        ax[2].plot(ddw_xvals, ddw)
        ax[2].set_title('Second order rate of change')
        ax[2].axvline(bankfull_id_elevation, color='black', label='Bankfull ID = {}m'.format(bankfull_id_elevation))
        ax[2].axvline(227.473, color='black', linestyle='dashed', label='2D model bankfull avg = 227.473m')
        plt.legend(loc='lower left')
        plt.savefig('data/data_outputs/{}/derivative_plots/{}'.format(reach_name, x_index))
        plt.close()
        ddw_ls.append(bankfull_id_elevation)

    breakpoint()

    print('Out of {} total measurements, {} were not accounted for based on uneven bank crossings'.format(total_measurements, incomplete_intersection_counter))
    
    plot_xss = [2,42,88,91,181,217]
    # Plot all widths spaghetti style
    for xs in plot_xss:
        fig, ax = plt.subplots()
        plt.xlabel('Distance from 0-elevation (m)')
        plt.ylabel('Channel width (m)')
        plt.title('Incremental channel top widths for {}'.format(reach_name))
        plt.axvline(22.244, label='median bankfull')
        # plt.xlim([60, 100])
        # plt.xlim([0, 8]) # use zoomed in axes for more detail at wdith break points
        # plt.ylim([0, 225]) # use zoomed in axes for more detail at wdith break points
        for index, row in all_widths_df.iterrows():
            x_len = round(len(row[0]) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            plt.plot(x_vals, row[0], alpha=0.3)
            if index == xs:
                xs_name = index
                plt.plot(x_vals, row[0], linewidth=2.5, label='x-section {}'.format(index), zorder=len(all_widths_df), color='red')
                plt.legend()
        # plt.savefig('data/data_outputs/{}/all_widths_xs-{}.jpeg'.format(reach_name, str(xs_name)), dpi=400)
        plt.close()

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    fig, ax = plt.subplots()
    plt.xlabel('Distance from channel bottom (m)')
    plt.ylabel('Channel width (m)')
    plt.title('Median incremental channel top widths for {}'.format(reach_name))
    plt.axvline(227.647, label='median bankfull')
    max_len = max(all_widths_df['widths'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded'] = all_widths_df['widths'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df = pd.DataFrame(all_widths_df['widths_padded'].tolist())
    transect_50 = padded_df.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25 = padded_df.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75 = padded_df.apply(lambda row: np.nanpercentile(row, 75), axis=0)
    x_len = round(len(transect_50) * d_interval, 4)
    x_vals = np.arange(0, x_len, d_interval)
    plt.xlim([210, 300])
    plt.plot(x_vals, transect_50, color='black')
    plt.plot(x_vals, transect_25, color='blue')
    plt.plot(x_vals, transect_75, color='blue')
    plt.savefig('data/data_outputs/{}/median_widths.jpeg'.format(reach_name), dpi=400)
    # after calcing array of w/d for each xs, calc the deltas

fig_list = plot_bankfull()
output = calc_dwdh(fig_list)
