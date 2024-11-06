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
import pdb

# Run parameters 
transect_fp = 'GIS/data_inputs/Leggett/XS_Sections/Thalweg_10m_adjusted.shp'
bankfull_fp = 'GIS/data_inputs/Leggett/Bankfull_raster/SFE_Leggett_011_d_Max.tif' # preprocessing required to convert native .flt to .tif
dem_fp = 'GIS/data_inputs/Leggett/1m_Topobathy/dem.tif'
reach_name = 'Leggett' # specify reach of the Eel River
median_bankfull = 227.647 # model-derived reach-averaged bankfull
median_topo_bankfull = 224.9 # topography-derived reach-averaged bankfull
modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))
plot_ylim = [220, 300]

# transect_fp = 'GIS/data_inputs/Miranda/XS_Sections/Thalweg_10m_adjusted.shp'
# bankfull_fp = 'GIS/data_inputs/Miranda/Bankfull_raster/SFE_Miranda_001_d_Max.tif' # preprocessing required to convert native .flt to .tif
# dem_fp = 'GIS/data_inputs/Miranda/1m_Topobathy/dem.tif'
# reach_name = 'Miranda' # specify reach of the Eel River
# median_bankfull = 72.284 # model-derived reach-averaged bankfull
# median_topo_bankfull = 68.4 # topography-derived reach-averaged bankfull
# modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
# topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))
# plot_ylim = [60, 100]

# transect_fp = 'GIS/data_inputs/Scotia/XS_Sections/Thalweg_15m_adjusted.shp'
# bankfull_fp = 'GIS/data_inputs/Scotia/Bankfull_raster/SFE_Scotia_011_d_Max.tif' # preprocessing required to convert native .flt to .tif
# dem_fp = 'GIS/data_inputs/Scotia/1m_Topobathy/dem.tif'
# reach_name = 'Scotia' # specify reach of the Eel River
# median_bankfull = 22.244 # model-derived reach-averaged bankfull
# median_topo_bankfull = 14.5 # topography-derived reach-averaged bankfull
# modeled_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))
# topo_bankfull_transects_df = pd.read_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))

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

interval = 1 # set plotting interval in units of meters

# transects = transects.iloc[246:,] # Memory capacity maybe reached, all Scotia plots will not generate in one push. 
def plot_bankfull():
    d_interval = 10/100 # units meters
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
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        station_z = []
        for i, value in enumerate(elevs):
            station_z.append(value)
        stations_plot_df = pd.DataFrame({'station_y':distances, 'station_z':station_z})
        bankfull_y_plot = [ydist0['geometry'][0], ydist1['geometry'][0]]
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]]
        bankfull_z_plot_avg = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies
        bankfull.append(bankfull_z_plot_avg)
        bankfull_plot_df = pd.DataFrame({'bankfull_y':bankfull_y_plot, 'bankfull_z':[bankfull_z_plot_avg, bankfull_z_plot_avg]})
        # Bring in channel top width data
        current_widths = pd.read_csv('data/data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, index))
        current_widths = current_widths['widths']
        width_xvals = get_x_vals(current_widths)
        # Bring in topo bankfull data
        current_topo_bankfull = topo_bankfull_transects_df['bankfull'][index]
        # Bring in rate of change data
        dw = pd.read_csv('data/data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, index))
        dw_xvals = get_x_vals(dw)
        ddw = pd.read_csv('data/data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, index))
        ddw_xvals = get_x_vals(ddw)
        # Plot everything together, 3-panel plot
        fig = plt.figure(figsize=(12,8))
        ax1 = plt.subplot2grid((2,2), (0,0), rowspan=2) # large left-side panel
        ax2 = plt.subplot2grid((2,2), (0,1)) # top-right panel
        ax3 = plt.subplot2grid((2,2), (1,1)) # bottom-right panel
        # breakpoint()
        ax1.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='black', linestyle='-', label='Transect')
        ax1.plot(bankfull_plot_df['bankfull_y'], bankfull_plot_df['bankfull_z'],color='red', linestyle='-', label='Bankfull')
        # Create empty plotline with blank marker containing bankfull label
        bankfull_label = str(round(bankfull_z_plot_avg, 2))
        ax1.plot([], [], ' ', label="Bankfull elev={}m".format(bankfull_label))
        try: 
            ax2.set_ylim(plot_ylim)
            ax3.set_ylim(plot_ylim)
        except:
            print('No xlim provided')     
        ax1.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
        ax1.set_xlabel('Meters')
        ax1.set_ylabel('Elevation (meters)')
        ax1.set_title('Eel River at {}'.format(reach_name))
        ax1.legend()
        ax2.plot(current_widths, width_xvals, label='first order rate of change')
        ax2.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='model-derived bankfull', alpha=0.5)
        ax2.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull', alpha=0.5)
        ax2.set_ylabel('Elevation (meters)')
        ax25 = ax2.twiny()
        ax25.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
        ax2.set_title('Cross-section and first-order rate of change')
        ax3.plot(ddw, ddw_xvals)
        ax3.set_ylabel('Elevation (meters)')
        ax3.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='model-derived bankfull')
        ax3.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
        ax35 = ax3.twiny()
        ax35.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
        ax3.set_title('Second order rate of change')
        plt.tight_layout()
        # Can I save each plot as an object, add it to a list, and return that list from this function?
        # fig_list.append(fig)
        plt.savefig('data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, index))
        plt.close()

        
        # x_section_xvals = get_x_vals(xsection)
        # dw_xvals = get_x_vals(dw)
        # ddw_xvals = get_x_vals(ddw)
        # fig, ax = plt.subplots(3,1, figsize=(18, 9))
        # plt.xlabel('Distance from 0-elevation (m)')
        # ax[0].plot(x_section_xvals, xsection)
        # ax[0].set_title('Cross-section')
        # ax[1].plot(dw_xvals, dw)
        # ax[1].set_title('First order rate of change')
        # ax[2].plot(ddw_xvals, ddw)
        # ax[2].set_title('Second order rate of change')
        # ax[2].axvline(bankfull_id_elevation, color='black', label='Bankfull ID = {}m'.format(bankfull_id_elevation))
        # ax[2].axvline(227.473, color='black', linestyle='dashed', label='2D model bankfull avg = 227.473m')
        # plt.legend(loc='lower left')
    # dictionary of lists
    bankfull_df = pd.DataFrame({'bankfull_ams':bankfull})
    bankfull_df.to_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))

    print('Median bankfull is {}m'.format(np.nanmedian(bankfull)))
    return()


def calc_dwdh():
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

        wh_ls_df = pd.DataFrame({'widths':wh_ls})
        # wh_ls_df.to_csv('data/data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, transects_index))
        wh_append = pd.DataFrame({'widths':[wh_ls], 'transect_id':transects_index})
        all_widths_df = pd.concat([all_widths_df, wh_append], ignore_index=True)

    # calculate and plot second derivative of width (height is constant)
    def get_x_vals(y_vals):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)

    bankfull_results = []
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
        bankfull_results.append(bankfull_id_elevation)
        # Output dw/ddw spacing, which is in 1/10 meter increments
        dw_xvals = get_x_vals(dw)
        ddw_xvals = get_x_vals(ddw)
        dw_df = pd.DataFrame({'elevation_m':dw_xvals, 'dw':dw})
        ddw_df = pd.DataFrame({'elevation_m':ddw_xvals, 'ddw':ddw})
        dw_df.to_csv('data/data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, x_index))
        ddw_df.to_csv('data/data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, x_index))

        # Plot x-section/first/second derivative plot and identified bankfull
        
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
        # plt.savefig('data/data_outputs/{}/derivative_plots/{}'.format(reach_name, x_index))
        plt.close()
    breakpoint()

    # Plot bankfull results along logitudinal profile
    modeled_bankfull_transects = modeled_bankfull_transects_df['bankfull_ams']
    modeled_bankfull_transects = [np.nan if x < 0 else x for x in modeled_bankfull_transects]
    if reach_name == 'Leggett' or reach_name == 'Miranda':
        transect_spacing = 10 # units meters
    elif reach_name == 'Scotia':
        bankfull_results = bankfull_results[:-1]
        transect_spacing = 15 # units meters
    x_len = len(bankfull_results)
    x_vals = np.arange(0, (x_len * transect_spacing), transect_spacing)
    fig, ax = plt.subplots()
    plt.xlabel('Transects from upstream to downstream (m)')
    plt.ylabel('Bankfull elevation ASL (m)')
    plt.title('Logitudinal profile of bankfull elevations, {}'.format(reach_name))
    plt.plot(x_vals, bankfull_results, label='topo-derived bankfull')
    plt.plot(x_vals, modeled_bankfull_transects, color='green', label='model-derived bankfull')
    plt.axhline(median_bankfull, linestyle='dashed', color='black', label='modeled median bankfull') 
    plt.axhline(median_topo_bankfull, linestyle='dashed', color='grey', label='topographic median bankfull')
    plt.legend(loc='upper right')
    # plt.savefig('data/data_outputs/{}/Bankfull_longitudinals'.format(reach_name))
    plt.close()
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
        # plt.xlim([0, 8]) # use zoomed in axes for more detail at width break points
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
    # save topo-derived bankfull for each transect
    bankfull_results_dt = pd.DataFrame({'bankfull':bankfull_results})
    bankfull_results_dt.to_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    fig, ax = plt.subplots()
    plt.xlabel('Distance from channel bottom (m)')
    plt.ylabel('Channel width (m)')
    plt.title('Median incremental channel top widths for {}'.format(reach_name))
    plt.axvline(median_bankfull, label='median bankfull')
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
    # plt.savefig('data/data_outputs/{}/median_widths.jpeg'.format(reach_name), dpi=400)
    # after calcing array of w/d for each xs, calc the deltas

output = plot_bankfull() 
# output = calc_dwdh()
