import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_bankfull(reach_name, transects, dem, d_interval, bankfull_boundary, plot_interval, topo_bankfull_transects_df, plot_ylim=None):
    d_interval = 10/100 # units meters
    # For each transect, find intersection points with bankfull, and plot transects with intersections
    bankfull = []
    for index, row in transects.iterrows():
        fig_list = []
        line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=transects.crs)
        intersect_pts = line.geometry.intersection(bankfull_boundary)

        # Generate a spaced interval of stations along each transect for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
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
        # ax1.plot(bankfull_plot_df['bankfull_y'], bankfull_plot_df['bankfull_z'],color='red', linestyle='-', label='Model-derived bankfull')
        ax1.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topo-derived bankfull')
        ax1.axhline(bankfull_z_plot_avg, color='red', linestyle='-', label='Benchmark bankfull')
        # Create empty plotline with blank marker containing bankfull label
        # bankfull_label = str(round(bankfull_z_plot_avg, 2))
        # ax1.plot([], [], ' ', label="Bankfull elev={}m".format(bankfull_label))
        try: 
            ax2.set_ylim(plot_ylim)
            ax3.set_ylim(plot_ylim)
        except:
            print('No xlim provided')     
        # ax1.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
        ax1.set_xlabel('Meters')
        ax1.set_ylabel('Elevation (meters)')
        ax1.set_title('Eel River at {}'.format(reach_name))
        ax1.legend()
        ax2.plot(current_widths, width_xvals, label='first order rate of change')
        ax2.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='benchmark bankfull', alpha=0.5)
        ax2.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull', alpha=0.5)
        ax2.set_ylabel('Elevation (meters)')
        ax25 = ax2.twiny()
        ax25.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
        ax2.set_title('Cross-section and first-order rate of change')
        ax3.plot(ddw, ddw_xvals)
        ax3.set_ylabel('Elevation (meters)')
        ax3.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='benchmark bankfull')
        ax3.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
        ax35 = ax3.twiny()
        ax35.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
        ax3.set_title('Second order rate of change')
        plt.tight_layout()
        plt.savefig('data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, index))
        plt.close()
    bankfull_df = pd.DataFrame({'bankfull_ams':bankfull})
    bankfull_df.to_csv('data/data_outputs/{}/transect_bankfull_modeled.csv'.format(reach_name))

    return()

def plot_longitudinal_bf(reach_name, modeled_bankfull_transects_df, topo_bankfull_transects_df, median_bankfull, median_topo_bankfull):
    # Calc bankfull ranges for plotting
    modeled_bf = modeled_bankfull_transects_df['bankfull_ams']
    topo_bf = topo_bankfull_transects_df['bankfull']
    modeled_25 = np.nanpercentile(modeled_bf, 25)
    modeled_75 = np.nanpercentile(modeled_bf, 75)
    topo_25 = np.nanpercentile(topo_bf, 25)
    topo_75 = np.nanpercentile(topo_bf, 75)
    
    # Plot bankfull results along logitudinal profile
    modeled_bankfull_transects = modeled_bankfull_transects_df['bankfull_ams']
    modeled_bankfull_transects = [np.nan if x < 0 else x for x in modeled_bankfull_transects]
    bankfull_results = topo_bankfull_transects_df['bankfull']
    if reach_name == 'Leggett' or reach_name == 'Miranda':
        transect_spacing = 10 # units meters
    elif reach_name == 'Scotia':
        bankfull_results = bankfull_results[:-1] # remove erroneous final value
        modeled_bankfull_transects = modeled_bankfull_transects[:-1]
        transect_spacing = 15 # units meters
    x_len = len(bankfull_results)
    x_vals = np.arange(0, (x_len * transect_spacing), transect_spacing)
    fig, ax = plt.subplots()
    plt.xlabel('Transects from upstream to downstream (m)')
    plt.ylabel('Bankfull elevation ASL (m)')
    plt.title('Logitudinal profile of bankfull elevations, {}'.format(reach_name))
    plt.plot(x_vals, bankfull_results, label='Topographic bankfull')
    plt.plot(x_vals, modeled_bankfull_transects, color='green', label='Benchmark bankfull')
    plt.axhline(modeled_25, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axhline(modeled_75, linestyle='dashed', color='black') 
    plt.axhline(topo_25, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.axhline(topo_75, linestyle='dashed', color='grey')
    plt.legend(loc='upper right')
    plt.savefig('data/data_outputs/{}/Bankfull_longitudinals'.format(reach_name))
    plt.close()

def plot_bankfull_increments(reach_name, all_widths_df, d_interval, topo_bankfull_transects_df, modeled_bankfull_transects_df, median_bankfull, median_topo_bankfull, bankfull_width, plot_ylim):
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style
    fig, ax = plt.subplots()
    plt.ylabel('Channel width (m)')
    plt.xlabel('Height above sea level (m)')
    plt.title('Incremental channel top widths for {}'.format(reach_name))
    # try: 
    #     plt.xlim(plot_ylim)
    # except:
    #     print('No ylim provided')
    # plt.ylim(plot_ylim)
    plt.xlim((220,270))
    for index, row in all_widths_df.iterrows():
        x_len = round(len(row[0]) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        plt.plot(x_vals, row[0], alpha=0.3, color=cmap(norm(index)), linewidth=0.75) # Try plot with axes flipped
    # plt.axvline(bankfull_width, label='Median width at modeled bankfull'.format(str(median_bankfull)), color='black', linewidth=0.75)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    plt.savefig('data/data_outputs/{}/all_widths.jpeg'.format(reach_name), dpi=400)
    plt.close()

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    modeled_bf = modeled_bankfull_transects_df['bankfull_ams']
    topo_bf = topo_bankfull_transects_df['bankfull']
    modeled_25 = np.nanpercentile(modeled_bf, 25)
    modeled_75 = np.nanpercentile(modeled_bf, 75)
    topo_25 = np.nanpercentile(topo_bf, 25)
    topo_75 = np.nanpercentile(topo_bf, 75)

    fig, ax = plt.subplots()
    plt.xlabel('Height above sea level (m)')
    plt.ylabel('Channel width (m)')
    plt.title('Median incremental channel top widths for {}'.format(reach_name))
    max_len = max(all_widths_df['widths'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded'] = all_widths_df['widths'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df = pd.DataFrame(all_widths_df['widths_padded'].tolist())
    transect_50 = padded_df.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25 = padded_df.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75 = padded_df.apply(lambda row: np.nanpercentile(row, 75), axis=0)
    x_len = round(len(transect_50) * d_interval, 4)
    x_vals = np.arange(0, x_len, d_interval)
    if reach_name == 'Scotia':
        plt.xlim(plot_ylim) # truncate unneeded values from plot
    if reach_name == 'Miranda':
        plt.xlim(plot_ylim)
    if reach_name == 'Leggett':
        plt.xlim(plot_ylim)
    plt.plot(x_vals, transect_50, color='black')
    plt.plot(x_vals, transect_25, color='blue')
    plt.plot(x_vals, transect_75, color='blue')
    plt.axvline(modeled_25, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axvline(modeled_75, linestyle='dashed', color='black') 
    plt.axvline(topo_25, linestyle='dashed', color='grey')
    plt.axvline(topo_75, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.legend()
    plt.savefig('data/data_outputs/{}/median_widths.jpeg'.format(reach_name), dpi=400)


