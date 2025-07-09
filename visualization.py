import pandas as pd
import numpy as np
from numpy import nan
import glob
import geopandas as gpd
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_longitudinal_bf(reach_name):
    bankfull_topo_detrend = pd.read_csv('data/data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    bankfull_benchmark_detrend = pd.read_csv('data/data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))

    # Calc bankfull ranges for plotting
    benchmark_25 = np.nanpercentile(bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], 25)
    benchmark_75 = np.nanpercentile(bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], 75)
    topo_25 = np.nanpercentile(bankfull_topo_detrend['bankfull'], 25)
    topo_75 = np.nanpercentile(bankfull_topo_detrend['bankfull'], 75)

    # Extract and detrend thalweg for plotting
    thalwegs = all_widths_df['thalweg_elev']
    thalwegs_detrend = []
    x_vals_thalweg = np.arange(0, len(thalwegs))
    x = np.array(x_vals_thalweg).reshape(-1, 1)
    y = np.array(thalwegs)
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from thalwegs
    for index, val in enumerate(thalwegs):
        thalwegs_detrend.append(val - fit_slope[index])

    # Plot bankfull results along logitudinal profile
    if reach_name == 'Leggett' or reach_name == 'Miranda':
        transect_spacing = 10 # units meters
    elif reach_name == 'Scotia':
        bankfull_topo_detrend = bankfull_topo_detrend[:-1] # remove erroneous final value
        bankfull_benchmark_detrend = bankfull_benchmark_detrend[:-1]
        thalwegs_detrend = thalwegs_detrend[:-1]
        transect_spacing = 15 # units meters
    x_len = len(bankfull_topo_detrend)
    x_vals = np.arange(0, (x_len * transect_spacing), transect_spacing)
    fig, ax = plt.subplots()
    plt.xlabel('Transects from upstream to downstream (m)')
    plt.ylabel('Bankfull elevation ASL (m)')
    plt.title('Logitudinal profile of bankfull elevations, {}'.format(reach_name))
    plt.plot(x_vals, bankfull_topo_detrend['bankfull'], label='Topographic bankfull')
    plt.plot(x_vals, bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], color='green', label='Benchmark bankfull')
    plt.plot(x_vals, thalwegs_detrend, color='grey', label='Thalweg (detrended)')
    plt.axhline(benchmark_25, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axhline(benchmark_75, linestyle='dashed', color='black') 
    plt.axhline(topo_25, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.axhline(topo_75, linestyle='dashed', color='grey')
    plt.legend(loc='upper right')
    plt.savefig('data/data_outputs/{}/Bankfull_longitudinals'.format(reach_name))
    plt.close()

def plot_bankfull_increments(reach_name, d_interval, plot_ylim):
    # bankfull_topo = pd.read_csv('data/data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    # bankfull_benchmark = pd.read_csv('data/data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    # aggregate_topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_aggregate_elevation.csv'.format(reach_name))
    agg_bankfull = pd.read_csv('data/data_outputs/{}/max_inflections.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])

    # Detrend widths before plotting based on thalweg elevation, and start plotting point based on detrend
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope] # unnest the array
    
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style
    fig, ax = plt.subplots()
    plt.ylabel('Channel width (m)')
    plt.xlabel('Detrended elevation (m)')
    # plt.title('Incremental channel widths for {}'.format(reach_name))
    plt.title('Incremental channel widths for Eel River upper reach')
    if reach_name == 'Leggett':
        plt.xlim((-5,30))
    if reach_name == 'Miranda':
        plt.xlim((60,80))
    elif reach_name == 'Scotia':
        plt.xlim(10,40)

    for index, row in all_widths_df.iterrows(): 
        row = row['widths']
        x_len = round(len(row) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        # apply detrend shift to xvals
        x_vals = [x_val - fit_slope[index] - intercept for x_val in x_vals]
        plt.plot(x_vals, row, alpha=0.3, color=cmap(norm(index)), linewidth=0.75) # Try plot with axes flipped
    # plt.axvline(bankfull_width, label='Median width at modeled bankfull'.format(str(median_bankfull)), color='black', linewidth=0.75)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    plt.savefig('data/data_outputs/{}/all_widths.jpeg'.format(reach_name), dpi=400)
    plt.close()

    # # Try: plot Scotia lines in distinct chunks
    # all_widths = all_widths_df['widths']
    # Scotia_1 = all_widths.iloc[0:100]
    # Scotia_2 = all_widths.iloc[100:200]
    # Scotia_3 = all_widths.iloc[200:]
    
    # fig, ax = plt.subplots()
    # plt.ylabel('Channel width (m)')
    # plt.xlabel('Height above sea level (m)')
    # plt.title('Channel width/height ratios for Scotia reaches')
    # plt.xlim(10,30)
    # plt.ylim(0,350)
    # for row in Scotia_3:
    #     x_len = round(len(row) * d_interval, 4)
    #     x_vals = np.arange(0, x_len, d_interval)
    #     if row == Scotia_3.iloc[0]:
    #         plt.plot(x_vals, row, alpha=0.3, color='blue', linewidth=0.75, label='Cross-sections 200-315')
    #     else:
    #         plt.plot(x_vals, row, alpha=0.3, color='blue', linewidth=0.75)
    # for row in Scotia_2:
    #     x_len = round(len(row) * d_interval, 4)
    #     x_vals = np.arange(0, x_len, d_interval)
    #     if row == Scotia_2.iloc[0]:
    #         plt.plot(x_vals, row, alpha=0.3, color='green', linewidth=0.75, label='Cross-sections 100-200')
    #     else:
    #         plt.plot(x_vals, row, alpha=0.3, color='green', linewidth=0.75)
    # for row in Scotia_1:
    #     x_len = round(len(row) * d_interval, 4)
    #     x_vals = np.arange(0, x_len, d_interval)
    #     if row == Scotia_1.iloc[0]:
    #         plt.plot(x_vals, row, alpha=0.3, color='yellow', linewidth=0.75, label='Cross-sections 1-100')
    #     else:
    #         plt.plot(x_vals, row, alpha=0.3, color='yellow', linewidth=0.75)
    
    
    # plt.legend()
    # plt.savefig('data/data_outputs/Scotia/widths_by_reach_2.jpeg', dpi=400)
    # breakpoint()

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    bankfull_benchmark = bankfull_benchmark['benchmark_bankfull_ams_detrend']
    bankfull_topo = bankfull_topo['bankfull']
    benchmark_25 = np.nanpercentile(bankfull_benchmark, 25)
    benchmark_75 = np.nanpercentile(bankfull_benchmark, 75)
    topo_25 = np.nanpercentile(bankfull_topo, 25)
    topo_75 = np.nanpercentile(bankfull_topo, 75)

    fig, ax = plt.subplots()
    plt.xlabel('Height above sea level (m)')
    plt.ylabel('Channel width (m)')
    plt.title('Incremental channel top widths for {}'.format(reach_name))
    # Prepare widths for plotting
    max_len = max(all_widths_df['widths'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded'] = all_widths_df['widths'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df = pd.DataFrame(all_widths_df['widths_padded'].tolist())
    transect_50 = padded_df.apply(lambda row: np.nanmedian(row), axis=0)
    transect_25 = padded_df.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75 = padded_df.apply(lambda row: np.nanpercentile(row, 75), axis=0)

    x_len = round(len(transect_50) * d_interval, 4)
    x_vals = np.arange(0, x_len, d_interval)
    if reach_name == 'Scotia':
        plt.xlim(plot_ylim) # truncate unneeded values from plot
    if reach_name == 'Miranda':
        plt.xlim(60, 80)
    if reach_name == 'Leggett':
        plt.xlim(plot_ylim)
    plt.plot(x_vals, transect_50, color='black', label='Width/height median')
    plt.plot(x_vals, transect_25, color='blue', label='Width/height 25-75%')
    plt.plot(x_vals, transect_75, color='blue')
    plt.legend(loc='center right')

    plt.axvline(benchmark_25, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axvline(benchmark_75, linestyle='dashed', color='black') 
    plt.axvline(aggregate_topo_bankfull['bankfull'][0], color='black', label='Aggregate topo-derived bankfull')
    plt.axvline(topo_25, linestyle='dashed', color='grey')
    plt.axvline(topo_75, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.legend()
    # Save 25/50/75 width lines to csv for later use
    transect_25_df = pd.DataFrame(transect_25)
    transect_50_df = pd.DataFrame(transect_50)
    transect_75_df = pd.DataFrame(transect_75)
    transect_25_df.to_csv('data/data_outputs/{}/transect_25.csv'.format(reach_name))
    transect_50_df.to_csv('data/data_outputs/{}/transect_50.csv'.format(reach_name))
    transect_75_df.to_csv('data/data_outputs/{}/transect_75.csv'.format(reach_name))
    plt.savefig('data/data_outputs/{}/median_widths.jpeg'.format(reach_name), dpi=400) # try mean instead of median

    # Detrended, aggregated cross-sections using padded-zeros approach
    all_widths_df['widths_detrend'] = [[] for _ in range(len(all_widths_df))] 
    # Loop through all_widths
    for index, row in all_widths_df.iterrows():
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if offset_int < 0: # most likely case, downstream xsections are lower elevation than furthest upstream
            # populate new column of df with width values
            all_widths_df.loc[index, 'widths_detrend'].extend([0] * abs(offset_int) + row['widths']) # add zeros to beginning of widths list. Need to unnest when using.
        elif offset_int > 0: # this probably won't come up
            all_widths_df.loc[index, 'widths_detrend'].extend(row[abs(offset_int):])
        else:
            all_widths_df.loc[index, 'widths_detrend'].extend(row['widths'])
    
    # Once all offsets applied, use zero-padding aggregation method just like with non-detrended widths.
    max_len = max(all_widths_df['widths_detrend'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded_detrend'] = all_widths_df['widths_detrend'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df_detrend = pd.DataFrame(all_widths_df['widths_padded_detrend'].tolist())
    transect_50_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 75), axis=0)
    plot_df = pd.DataFrame({'50th':transect_50_detrend, '25th':transect_25_detrend, '75th':transect_75_detrend})
    plot_df.to_csv('data/data_outputs/{}/width_elev_plotlines.csv'.format(reach_name))
    # pdb.set_trace()

def stacked_width_plots(d_interval):
    # Bring in all plot lines from upper, mid, lower
    upper_plotlines = pd.read_csv('data/data_outputs/Leggett/width_elev_plotlines.csv')
    mid_plotlines = pd.read_csv('data/data_outputs/Miranda/width_elev_plotlines.csv')
    lower_plotlines = pd.read_csv('data/data_outputs/Scotia/width_elev_plotlines.csv')
    # Determine where to begin plotting based on where median line goes above zero
    def start_plot(line_25th):
        for index, value in line_25th.items():
            if value > 0:
                return index
    upper_plot_start = start_plot(upper_plotlines['25th'])
    upper_plotlines = upper_plotlines[upper_plot_start:].reset_index()
    middle_plot_start = start_plot(mid_plotlines['25th'])
    mid_plotlines = mid_plotlines[middle_plot_start:].reset_index()
    lower_plot_start = start_plot(lower_plotlines['25th'])
    lower_plotlines = lower_plotlines[lower_plot_start:].reset_index()

    def get_x_vals(y_vals):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
    x_upper = get_x_vals(upper_plotlines['50th'][:110])
    x_mid = get_x_vals(mid_plotlines['50th'][:110])
    x_lower = get_x_vals(lower_plotlines['50th'][:110])
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(x_upper, upper_plotlines['50th'][:110], color='orange', label='upper reach')
    plt.plot(x_mid, mid_plotlines['50th'][:110], color='green',label='middle reach')
    plt.plot(x_lower, lower_plotlines['50th'][:110], color='blue',label='lower reach')
    plt.xlabel('Relative elevation (m)')
    plt.ylabel('Channel width (m)')
    plt.legend()

    plt.savefig('data/data_outputs/stacked_width_elev.jpeg')
    return

def transect_plot(transects, dem, plot_interval, d_interval, bankfull_boundary, reach_name):
    # topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))

    # Use thalweg to detrend elevation on y-axes for transect plotting. Don't remove intercept (keep at elevation) 
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        intersect_pts = line.geometry.intersection(bankfull_boundary)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        # Add detrend value to elevs
        # elevs = [i - fit_slope[transects_index] for i in elevs]

        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]] # elevation of benchmark bankfull for plotting
        bankfull_z_plot = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies

        # bring in topo bankfull for plotting
        # current_topo_bankfull = topo_bankfull['bankfull'][transects_index]

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        min_y = min(elevs)
        fig = plt.figure(figsize=(6,8))
        plt.plot(distances, elevs, color='black', linestyle='-', label='Cross section')
        # plt.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topographic-derived bankfull')
        # plt.axhline(bankfull_z_plot, color='red', linestyle='-', label='Benchmark bankfull')
        # for index, x in enumerate(inflections['pos_inflections']):
        #     if index == 0:
        #         plt.axhline(x*d_interval, color='red', label='positive inflections', alpha=0.5)
        #     else:
        #         plt.axhline(x*d_interval, color='red', alpha=0.5)
        # for index, x in enumerate(inflections['neg_inflections']):
        #     if index == 0:
        #         plt.axhline(x*d_interval, color='blue', label='negative inflections', alpha=0.5)
        #     else:
        #         plt.axhline(x*d_interval, color='blue', alpha=0.5)

        plt.axhline((15.85+fit_slope[transects_index]), color='grey', linestyle='dashed', label='Inflection Point bankfull')
        plt.axhline((19.5+fit_slope[transects_index]), color='red', label='Proposed bankfull')

        plt.xlabel('Cross section distance (meters)', fontsize=12)
        plt.ylabel('Elevation (meters)', fontsize=12)
        plt.legend(fontsize=12)
        # increase font size for axes and labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(top=45)
        # Make the bottom of the ylim fall a meter below the lowest point in cross section
        plt.tight_layout()
        plt.savefig('data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()

def plot_wd_and_xsections(reach_name, d_interval, plot_ylim, transects, dem, plot_interval):
    bankfull_topo = pd.read_csv('data/data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    bankfull_benchmark = pd.read_csv('data/data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    aggregate_topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_aggregate_elevation.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))
    median_bf_topo = np.nanmedian(bankfull_topo['bankfull'])
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style


    # Add a second panel with cross-section
    # Loop through cross-sections and plot each one
    for transects_index, transects_row in transects.iterrows():
        current_bf_topo = bankfull_topo['bankfull'][transects_index]
        # 1. Plot spaghetti lines on first plot panel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.set_ylabel('Height above sea level (m)')
        ax1.set_xlabel('Channel width (m)')
        ax1.set_title('Channel width/height ratios for {}'.format(reach_name))
        if reach_name == 'Leggett':
            ax1.set_ylim((220,270))
        elif reach_name == 'Miranda':
            ax1.set_ylim((62.5, 80))
        elif reach_name == 'Scotia':
            ax1.set_ylim(10,40)

        for index, row in all_widths_df.iterrows(): 
            row = row['widths']
            x_len = round(len(row) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            if index == transects_index:
                ax1.plot(row, x_vals, alpha=1, color='red', linewidth=1)
            else:       
                ax1.plot(row, x_vals, alpha=0.3, color=cmap(norm(index)), linewidth=0.75) # Plot w elevation on y axis
        ax1.axhline(median_bf_topo, label='Median topographic bankfull'.format(str(median_bf_topo)), color='black', linewidth=0.75)
        ax1.axhline(current_bf_topo, label='Cross-section topographic bankfull'.format(str(current_bf_topo)), color='red', linewidth=0.75)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Set array to avoid warnings
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label("Downstream distance (m)")

        # 2. Plot cross-section on second panel
        
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval)
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        normalized_elevs = elevs
        def get_x_vals(y_vals):
                x_len = round(len(y_vals) * d_interval, 4)
                x_vals = np.arange(0, x_len, d_interval)
                return(x_vals)
        min_y = min(normalized_elevs)
        ax2.plot(distances, normalized_elevs, color='black', linestyle='-', label='Cross section')
        ax2.axhline(median_bf_topo, label='Median topographic bankfull'.format(str(median_bf_topo)), color='black', linewidth=0.75)
        ax2.axhline(current_bf_topo, label='Cross-section topographic bankfull'.format(str(current_bf_topo)), color='red', linewidth=0.75)
        ax2.legend()
        ax2.set_ylim(62.5, 80)
        ax2.set_title('Cross section {}'.format(str(transects_index)))
        plt.savefig('data/data_outputs/{}/dw_xs_plots/{}.jpeg'.format(reach_name, transects_index), dpi=400)
        plt.close()

def multi_panel_plot(reach_name, transects, dem, plot_interval, d_interval, bankfull_boundary):
    topo_bankfull = pd.read_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        intersect_pts = line.geometry.intersection(bankfull_boundary)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Base all depths on 0-elevation
        normalized_elevs = elevs

        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        # Be aware, Scotia transects size exceeds limits for figures saved in one folder (no warning issued)
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]] # elevation of benchmark bankfull for plotting
        bankfull_z_plot = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies

        # bring in topo bankfull for plotting
        current_topo_bankfull = topo_bankfull['bankfull'][transects_index]

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        min_y = min(normalized_elevs)
        fig = plt.figure(figsize=(8,8))
        plt.plot(distances, normalized_elevs, color='black', linestyle='-', label='Cross section')
        plt.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topographic-derived bankfull')
        plt.axhline(bankfull_z_plot, color='red', linestyle='-', label='Benchmark bankfull')
        plt.xlabel('Cross section distance (meters)', fontsize=12)
        plt.ylabel('Elevation (meters)', fontsize=12)
        plt.legend(fontsize=12)
        # increase font size for axes and labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Make the bottom of the ylim fall a meter below the lowest point in cross section
        if reach_name == 'Scotia':
            plt.ylim((min_y-1),30)
        elif reach_name == 'Miranda':
            plt.ylim((min_y-1),80)
        elif reach_name == 'Leggett':
            plt.ylim((min_y-1),250)
        plt.tight_layout()
        plt.savefig('data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()
        # pdb.set_trace()

        
    # bankfull_z = list(dem.sample(coords))
    # coord0 = Point(coords[0]) # separate out intersection coordinates so distance function works properly
    # coord1 = Point(coords[1]) # separate out intersection coordinates so distance function works properly
    # ydist0 = coord0.distance(station_zero_gpf)
    # ydist1 = coord1.distance(station_zero_gpf)
    # # Arrange points together for plotting
    # def get_x_vals(y_vals):
    #     x_len = round(len(y_vals) * d_interval, 4)
    #     x_vals = np.arange(0, x_len, d_interval)
    #     return(x_vals)
    # station_z = []
    # for i, value in enumerate(elevs):
    #     station_z.append(value)
    # stations_plot_df = pd.DataFrame({'station_y':distances, 'station_z':station_z})
    # bankfull_y_plot = [ydist0['geometry'][0], ydist1['geometry'][0]]
    # bankfull_z_plot_avg = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies
    # bankfull.append(bankfull_z_plot_avg)
    # bankfull_plot_df = pd.DataFrame({'bankfull_y':bankfull_y_plot, 'bankfull_z':[bankfull_z_plot_avg, bankfull_z_plot_avg]})
    #  # Bring in channel top width data
    # current_widths = pd.read_csv('data/data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, index))
    # current_widths = current_widths['widths']
    # width_xvals = get_x_vals(current_widths)
    # # Bring in topo bankfull data
    # current_topo_bankfull = bankfull_results[index]
    # # Bring in rate of change data
    # dw = pd.read_csv('data/data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, index))
    # dw_xvals = get_x_vals(dw)
    # ddw = pd.read_csv('data/data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, index))
    # ddw_xvals = get_x_vals(ddw)
    # # Plot everything together, 3-panel plot
    # fig = plt.figure(figsize=(12,8))
    # ax1 = plt.subplot2grid((2,2), (0,0), rowspan=2) # large left-side panel
    # ax2 = plt.subplot2grid((2,2), (0,1)) # top-right panel
    # ax3 = plt.subplot2grid((2,2), (1,1)) # bottom-right panel
    # # breakpoint()
    # ax1.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='black', linestyle='-', label='Transect')
    # # ax1.plot(bankfull_plot_df['bankfull_y'], bankfull_plot_df['bankfull_z'],color='red', linestyle='-', label='Model-derived bankfull')
    # ax1.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topo-derived bankfull')
    # ax1.axhline(bankfull_z_plot_avg, color='red', linestyle='-', label='Benchmark bankfull')
    # # Create empty plotline with blank marker containing bankfull label
    # # bankfull_label = str(round(bankfull_z_plot_avg, 2))
    # # ax1.plot([], [], ' ', label="Bankfull elev={}m".format(bankfull_label))
    # try: 
    #     ax2.set_ylim(plot_ylim)
    #     ax3.set_ylim(plot_ylim)
    # except:
    #     print('No xlim provided')     
    # # ax1.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
    # ax1.set_xlabel('Meters')
    # ax1.set_ylabel('Elevation (meters)')
    # ax1.set_title('Eel River at {}'.format(reach_name))
    # ax1.legend()
    # ax2.plot(current_widths, width_xvals, label='first order rate of change')
    # ax2.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='benchmark bankfull', alpha=0.5)
    # ax2.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull', alpha=0.5)
    # ax2.set_ylabel('Elevation (meters)')
    # ax25 = ax2.twiny()
    # ax25.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
    # ax2.set_title('Cross-section and first-order rate of change')
    # ax3.plot(ddw, ddw_xvals)
    # ax3.set_ylabel('Elevation (meters)')
    # ax3.axhline(bankfull_plot_df['bankfull_z'][0], color='red', label='benchmark bankfull')
    # ax3.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='topo-derived bankfull')
    # ax35 = ax3.twiny()
    # ax35.plot(stations_plot_df['station_y'], stations_plot_df['station_z'], color='grey', linestyle='-', label='Transect')
    # ax3.set_title('Second order rate of change')
    # plt.tight_layout()
    # plt.savefig('data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg'.format(reach_name, index))
    # plt.close()
    # return 

def plot_inflections(all_widths_df, d_interval, reach_name):
    # bring in 2nd derivative files
    inflections_fp = glob.glob('data_outputs/{}/second_order_roc/*'.format(reach_name))
    # bring in aggregated inflections array for plotting
    inflections_array_agg = pd.read_csv('data_outputs/{}/inflections_array_agg.csv'.format(reach_name))
    # Use thalweg elevs to detrend 2nd derivatives. Don't remove intercept (keep at elevation) 
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]

    # Set up plot and create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(inflections_fp)-1)
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.ylabel('Inflection magnitude')
    plt.xlabel('Detrended elevation (m)')
    plt.title('Cross section width inflections for {}'.format(reach_name))
    if reach_name == 'Leggett':
        plt.xlim((-5,30))
    if reach_name == 'Miranda':
        plt.xlim((60,80))
    elif reach_name == 'Scotia':
        plt.xlim((75,400))
    # loop through files and plot
    for index, inflection_fp in enumerate(inflections_fp): 
        inflection = pd.read_csv(inflection_fp)
        # detrend inflections so they all plot at the same starting point
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if index >20:
            continue
            # first inflection plots all wonky, skip it
        if offset_int < 0:
            inflection = [0] * abs(offset_int) + inflection['ddw'].tolist()
        else: # Only other case is no detrend (first transect)
            inflection = inflection
        # plot all inflections spaghetti style
        plt.plot(inflection, alpha=0.5, color=cmap(norm(index)), linewidth=0.75) 
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Downstream distance (m)")
    # overlay aggregate inflections
    plt.plot(inflections_array_agg, color='black', linewidth=0.75)
    plt.savefig('data_outputs/{}/inflections_all.jpeg'.format(reach_name))
    return

def output_record(reach_name, slope_window, d_interval, lower_bound, upper_bound):
    topo_bankfull = pd.read_csv('data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    topo_bankfull_detrend = pd.read_csv('data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    benchmark_bankfull = pd.read_csv('data_outputs/{}/bankfull_benchmark.csv'.format(reach_name))
    benchmark_bankfull_detrend = pd.read_csv('data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    topo_aggregate_bankfull = pd.read_csv('data_outputs/{}/bankfull_aggregate_elevation.csv'.format(reach_name))
    topo_bf_median = np.nanmedian(topo_bankfull['bankfull'])
    topo_bf_median_detrend = np.nanmedian(topo_bankfull_detrend['bankfull'])
    benchmark_bf_median = np.nanmedian(benchmark_bankfull['benchmark_bankfull_ams'])
    benchmark_bf_median_detrend = np.nanmedian(benchmark_bankfull_detrend['benchmark_bankfull_ams_detrend'])
    topo_aggregate_df = topo_aggregate_bankfull['bankfull']
    record_df = pd.DataFrame({'topo_bf_median': [topo_bf_median], 'topo_bf_median_detrend':[topo_bf_median_detrend], 'benchmark_bf_median': [benchmark_bf_median], \
                              'benchmark_bf_median_detrend':[benchmark_bf_median_detrend],'topo_aggregate_df': [topo_aggregate_df[0]],\
                              'slope_window': [slope_window], 'd_interval': [d_interval], 'lower_search_bound': [lower_bound], 'upper_search_bound': [upper_bound]})
    record_df.to_csv('data_outputs/{}/Summary_results.csv'.format(reach_name))
