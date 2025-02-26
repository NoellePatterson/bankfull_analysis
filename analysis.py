import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
import pdb

def get_x_vals(y_vals, d_interval):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
def multipoint_slope(windowsize, timeseries, xvals):
    dw = np.zeros(len(timeseries))
    lr_window = int(windowsize/2) # indexing later requires this to be an integer
    for n in range(lr_window, len(timeseries) - lr_window):
        regress = timeseries[n - lr_window:n + lr_window]
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(xvals[n - lr_window:n + lr_window], regress)
        dw[n] = slope1  
    return dw  
    
def find_boundary(xsection, bound):
    for index, val in enumerate(xsection):
        if val > bound: # find first instance of exceeding lower width bound
            bound_index = index
            break
    return bound_index

def id_benchmark_bankfull(reach_name, transects, dem, d_interval, bankfull_boundary, plot_interval):
    # For each transect, find intersection points with bankfull, and plot transects with intersections
    bankfull_benchmark = []
    for index, row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=transects.crs)
        intersect_pts = line.geometry.intersection(bankfull_boundary)

        # Generate a spaced interval of stations along each transect for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        
        # Get Y and Z coordinates for bankfull intersections 
        station_zero = stations['geometry'][0]
        station_zero_gpf = gpd.GeoDataFrame({'geometry':[station_zero]}, crs=transects.crs)
        station_zero_gpf = station_zero_gpf.set_geometry('geometry') # ensure geometry is set correctly
        
        if isinstance(intersect_pts.geometry[0], Point):
            print('Only one intersection point identified; skipping to next cross section.')
            continue
        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        # Be aware, Scotia transects size exceeds limits for figures saved in one folder (no warning issued)
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot_avg = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies
        bankfull_benchmark.append(bankfull_z_plot_avg)

    # Detrend benchmark bankfull results
    x_vals = get_x_vals(bankfull_benchmark, d_interval)
    x = np.array(x_vals).reshape(-1, 1)
    y = np.array(bankfull_benchmark)
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from bankfull results
    bankfull_benchmark_detrend = []
    for index, val in enumerate(bankfull_benchmark):
        bankfull_benchmark_detrend.append(val - fit_slope[index])
       
    benchmark_bankfull_df = pd.DataFrame({'benchmark_bankfull_ams':bankfull_benchmark})
    benchmark_bankfull_df.to_csv('data/data_outputs/{}/bankfull_benchmark.csv'.format(reach_name))
    benchmark_bankfull_detrend_df = pd.DataFrame({'benchmark_bankfull_ams_detrend':bankfull_benchmark_detrend})
    benchmark_bankfull_detrend_df.to_csv('data/data_outputs/{}/bankfull_benchmark_detrend.csv'.format(reach_name))
    return()

def calc_dwdh(reach_name, transects, dem, plot_interval, d_interval):
    # Loop through xsections and create dw/dh array for each xsection
    # df to store arrays of w and h
    all_widths_df = pd.DataFrame(columns=['widths'])
    incomplete_intersection_counter = 0
    total_measurements = 0

    # Optionally: check width at benchmark bankfull for each transect, store in a separate list
    bankfull_width_ls = [] # using benchmark bankfull, track channel width at bankfull for each transect
    benchmark_bankfull_df = pd.read_csv('data/data_outputs/{}/bankfull_benchmark.csv'.format(reach_name))
    median_benchmark_bankfull = np.nanmedian(benchmark_bankfull_df['benchmark_bankfull_ams'])

    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        wh_ls = []
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Base all depths on 0-elevation
        normalized_elevs = elevs

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

            depth_0elev = round(d_interval * depth, 1)
            bm_bankfull = round(median_benchmark_bankfull, 1)
            if depth_0elev == bm_bankfull:
                bankfull_width_ls.append(width)
        thalweg = min(elevs) # track this for use later in detrending
        wh_ls_df = pd.DataFrame({'widths':wh_ls})
        wh_ls_df.to_csv('data/data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, transects_index))
        wh_append = pd.DataFrame({'widths':[wh_ls], 'transect_id':transects_index, 'thalweg_elev':thalweg})
        all_widths_df = pd.concat([all_widths_df, wh_append], ignore_index=True)
    bankfull_width = np.nanmedian(bankfull_width_ls)
    all_widths_df.to_csv('data/data_outputs/{}/all_widths.csv'.format(reach_name))
    return(all_widths_df, bankfull_width)

def calc_derivatives(reach_name, d_interval, all_widths_df, slope_window, lower_bound, upper_bound):
    # calculate and plot second derivative of width (height is constant)
    # Calc upper and lower bounds widths
    lower_ls = []
    upper_ls = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        for i, val in enumerate(xsection):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
        width_lower = xsection[start_width_index + lower_bound] # width at 0.5m above thalweg
        lower_ls.append(width_lower)
        try:
            width_upper = xsection[start_width_index + upper_bound] # width at 10m above thalweg
            upper_ls.append(width_upper)
        except:
            continue
    lower = np.nanmedian(lower_ls)
    upper = np.nanmedian(upper_ls)

    topo_bankfull = []
    bankfull_width = []
    test_bf_maxderiv = []
    test_bf_minderiv = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        dw = []
        ddw = []
        xs_xvals = get_x_vals(xsection, d_interval)
        dw = multipoint_slope(slope_window, xsection, xs_xvals)
        ddw = multipoint_slope(slope_window, dw, xs_xvals)

        # Find max second derivative as bankfull. 
        ddw_abs = [abs(i) for i in ddw] # This is how negative value is found. 
        # Add in upper and lower search bounds on max 2nd deriv bankfull ID
        lower_bound_index = find_boundary(xsection, lower)
        upper_bound_index = find_boundary(xsection, upper)
        max_ddw = np.nanmax(ddw_abs[lower_bound_index:upper_bound_index])
        max_neg_ddw = np.nanmin(ddw[lower_bound_index:upper_bound_index])
        max_ddw_index = ddw_abs.index(max_ddw)
        ddw = ddw.tolist()
        max_neg_ddw_index = ddw.index(max_neg_ddw)
        bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull
        topo_bankfull.append(bankfull_id_elevation)
        bankfull_width.append(xsection[max_ddw_index])
        # Output dw/ddw spacing, which is in 1/10 meter increments
        dw_xvals = get_x_vals(dw, d_interval)
        ddw_xvals = get_x_vals(ddw, d_interval)
        dw_df = pd.DataFrame({'elevation_m':dw_xvals, 'dw':dw})
        ddw_df = pd.DataFrame({'elevation_m':ddw_xvals, 'ddw':ddw})
        dw_df.to_csv('data/data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, x_index))
        ddw_df.to_csv('data/data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, x_index))
        test_bf_maxderiv.append(d_interval * max_neg_ddw_index)
        test_bf_minderiv.append(bankfull_id_elevation)

        # Plot derivative results for each cross section plus cross-section
        fig = plt.figure(figsize=(20,15))
        gs = GridSpec(3, 3)
        xs_plot = mpimg.imread("data/data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg".format(reach_name, x_index))
        if reach_name == 'Leggett':
            plot_x_lim = (2200,2300)
        elif reach_name == 'Miranda':    
            plot_x_lim = (650, 740)
        # plt.xlim((600,900)) (2200,2500) (50,400)
        plt.subplot2grid((3,3), (0,2))
        plt.plot(xsection)
        plt.axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
        # plt.axvline(max_neg_ddw_index, label='2nd derivative minima')
        plt.title('Incremental channel widths')
        plt.xlim(plot_x_lim)
        if reach_name == 'Leggett':
            plt.ylim((0, 100))
        plt.subplot2grid((3,3), (1,2))    
        plt.plot(dw)
        plt.axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
        # plt.axvline(max_neg_ddw_index, label='2nd derivative minima')
        plt.xlim(plot_x_lim)
        # plt.ylim((-200,100))
        plt.title('First derivative')
        plt.subplot2grid((3,3), (2,2))
        plt.plot(ddw)
        plt.axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
        # plt.axvline(max_neg_ddw_index, label='2nd derivative minima')
        plt.xlim(plot_x_lim)
        # plt.ylim((-1000,1000))
        plt.title('Second derivative')
        plt.legend()
        # Large plot spanning second column
        plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
        plt.imshow(xs_plot)
        
        fig.suptitle('Bankfull slope identification for {} with {}pt rolling avg'.format(reach_name, slope_window))
        plt.savefig('data/data_outputs/{}/derivative_plots/{}.jpeg'.format(reach_name, x_index))
    # breakpoint()
    # Use thalweg elevs to detrend bankfull elevation results. Don't remove intercept (keep at elevation) 
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from bankfull results
    topo_bankfull_detrend = []
    for index, val in enumerate(topo_bankfull):
        topo_bankfull_detrend.append(val - fit_slope[index])

    # save topo-derived bankfull for each transect
    topo_bankfull_detrend_dt = pd.DataFrame({'bankfull':topo_bankfull_detrend})
    topo_bankfull_detrend_dt.to_csv('data/data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    topo_bankfull_dt = pd.DataFrame({'bankfull':topo_bankfull})
    topo_bankfull_dt.to_csv('data/data_outputs/{}/bankfull_topo.csv'.format(reach_name))

    return(topo_bankfull, topo_bankfull_detrend)

def calc_derivatives_aggregate(reach_name, d_interval, all_widths_df, slope_window, lower_bound, upper_bound):
    lower_ls = []
    upper_ls = []
    # Determine upper and lower bounds for bankfull ID
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        for i, val in enumerate(xsection):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
        width_lower = xsection[start_width_index + lower_bound] # width at 0.5m above thalweg
        lower_ls.append(width_lower)
        try:
            width_upper = xsection[start_width_index + upper_bound] # width at 10m above thalweg
            upper_ls.append(width_upper)
        except:
            continue
    lower = np.nanmedian(lower_ls)
    upper = np.nanmedian(upper_ls)
    # average all xsection widths element-wise
    all_widths = all_widths_df['widths']
    max_len = max(len(ls) for ls in all_widths)
    all_widths_padded = [ls + [None] * (max_len - len(ls)) for ls in all_widths] # Pad with Nones to get all lists to same length
    # code to calculate element-wise average (from GPT)
    avg_width = [
        np.nanmedian([x for x in elements if x is not None])
        for elements in zip(*all_widths_padded)
    ]
    xs_xvals = get_x_vals(avg_width, d_interval)
    dw = multipoint_slope(slope_window, avg_width, xs_xvals)
    ddw = multipoint_slope(slope_window, dw, xs_xvals)
    lower_bound_index = find_boundary(avg_width, lower)
    upper_bound_index = find_boundary(avg_width, upper)
    # apply rolling avg derivative calcs
    # ID bankfull (one across entire reach), save in csv
    ddw_abs = [abs(i) for i in ddw]
    max_ddw = np.nanmax(ddw_abs[lower_bound_index:upper_bound_index])
    max_ddw_index = ddw_abs.index(max_ddw)
    ddw = ddw.tolist()
    max_neg_ddw = np.nanmin(ddw[lower_bound_index:upper_bound_index])
    max_neg_ddw_index = ddw.index(max_neg_ddw)
    bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull

    # Visualize steps above with a plot
    fig, axes = plt.subplots(3, 1, figsize=(15,15))
    if reach_name == 'Leggett':
        plot_x_lim = (2200,2300)
    elif reach_name == 'Miranda':    
        plot_x_lim = (650, 700)
    # plt.xlim((600,900)) (2200,2500) (50,400)
    axes[0].plot(avg_width)
    axes[0].axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
    # axes[0].axvline(max_neg_ddw_index, label='2nd derivative minima')
    axes[0].set_title('Incremental channel widths')
    axes[0].set_xlim(plot_x_lim)
    if reach_name == 'Leggett':
        axes[0].set_ylim((0, 100))
    if reach_name == 'Miranda':
        axes[0].set_ylim((-100, 100))
    axes[1].plot(dw)
    axes[1].axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
    # axes[1].axvline(max_neg_ddw_index, label='2nd derivative minima')
    axes[1].set_xlim(plot_x_lim)
    # axes[1].set_ylim((-200,100))
    axes[1].set_title('First derivative')
    axes[2].plot(ddw)
    axes[2].axvline(max_ddw_index, label='2nd derivative abs maxima', color='black')
    # axes[2].axvline(max_neg_ddw_index, label='2nd derivative minima')
    axes[2].set_xlim(plot_x_lim)
    # axes[2].set_ylim((-1000,1000))
    axes[2].set_title('Second derivative')
    plt.legend()
    
    fig.suptitle('Bankfull slope identification for {} with {}pt rolling avg'.format(reach_name, slope_window))
    plt.savefig('data/data_outputs/{}/aggregate_bankfull_slopes.jpeg'.format(reach_name))
    # output csv with the bankfull_id_elevation value in it
    bankfull_aggregate_df = pd.DataFrame({'bankfull':[bankfull_id_elevation]})
    bankfull_aggregate_df.to_csv('data/data_outputs/{}/bankfull_aggregate_elevation.csv'.format(reach_name))

    # there it is! 

def recurrence_interval():
    # Calculate recurrence interval of flow at bankfull stage along profile

    # Bring in topo bankfull results (not detrended)

    # Bring in flow-stage data (rating curve)
    # Calculate flow-recurrence intervals
    # For each bankfull result:
    # 1. Find corresponding flow
    # 2. Find correcsponding recurrence interval as years (i.e. 2-yr flow)

    # Visualize: plot recurrence intervals along profile
    pdb.set_trace()
