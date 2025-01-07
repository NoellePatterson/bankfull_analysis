import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import linregress

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

def calc_dwdh(reach_name, transects, dem, plot_interval, d_interval, median_bankfull):
    # Loop through xsections and create dw/dh array for each xsection
    # df to store arrays of w and h
    all_widths_df = pd.DataFrame(columns=['widths'])
    bankfull_width_ls = [] # using modeled bankfull, track channel width at bankfull for each transect
    thalweg_ls = []
    incomplete_intersection_counter = 0
    total_measurements = 0
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

        # # Strategy 1: normalize depths to thalweg
        # min_elevation = min(elevs)
        # normalized_elevs = elevs - min_elevation

        # Strategy 2: Base all depths on 0-elevation
        min_elevation = 0
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
            bankfull = round(median_bankfull, 1)
            if depth_0elev == bankfull:
                bankfull_width_ls.append(width)
        
        wh_ls_df = pd.DataFrame({'widths':wh_ls})
        wh_ls_df.to_csv('data/data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, transects_index))
        wh_append = pd.DataFrame({'widths':[wh_ls], 'transect_id':transects_index})
        all_widths_df = pd.concat([all_widths_df, wh_append], ignore_index=True)
    bankfull_width = np.nanmedian(bankfull_width_ls)
    return(all_widths_df, bankfull_width)

def calc_derivatives(reach_name, d_interval, all_widths_df):
    # calculate and plot second derivative of width (height is constant)
    # Calc upper and lower bounds widths
    lower_ls = []
    upper_ls = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        for i, val in enumerate(xsection):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
        width_lower = xsection[start_width_index + 5] # width at 0.5m above thalweg
        lower_ls.append(width_lower)
        try:
            width_upper = xsection[start_width_index + 100] # width at 10m above thalweg
            upper_ls.append(width_upper)
        except:
            continue
    lower = np.nanmedian(lower_ls)
    upper = np.nanmedian(upper_ls)

    bankfull_results = []
    bankfull_width = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        dw = []
        ddw = []
        xs_xvals = get_x_vals(xsection, d_interval)
        dw = multipoint_slope(5, xsection, xs_xvals)
        ddw = multipoint_slope(5, dw, xs_xvals)
        # Strategy #1: calc stepwise slopes (bench style?) 
        # for w_index, current_width in enumerate(xsection): # loop through all widths in current xsection
        #     if w_index < len(xsection) - 1: # can caluclate differences up to second to last index
        #         current_d = (xsection[w_index + 1] - current_width)/d_interval
        #         dw.append(current_d)
        # for dw_index, current_dw in enumerate(dw): # loop through all first order rate changes to get second order change  (slope break)
        #     if dw_index < len(dw) - 1: # can calculate differences up to second to last first-order change
        #         current_dd = (dw[dw_index + 1] - current_dw)/d_interval
        #         ddw.append(current_dd)

        # Find max second derivative as bankfull. turn this into a function... 
        ddw_abs = [abs(i) for i in ddw]
        # Add in upper and lower search bounds on max 2nd deriv bankfull ID
        lower_bound_index = find_boundary(xsection, lower)
        upper_bound_index = find_boundary(xsection, upper)
        max_ddw = np.nanmax(ddw_abs[lower_bound_index:upper_bound_index])
        max_ddw_index = ddw_abs.index(max_ddw)
        bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull
        bankfull_results.append(bankfull_id_elevation)
        bankfull_width.append(xsection[max_ddw_index])
        # Output dw/ddw spacing, which is in 1/10 meter increments
        dw_xvals = get_x_vals(dw, d_interval)
        ddw_xvals = get_x_vals(ddw, d_interval)
        dw_df = pd.DataFrame({'elevation_m':dw_xvals, 'dw':dw})
        ddw_df = pd.DataFrame({'elevation_m':ddw_xvals, 'ddw':ddw})
        dw_df.to_csv('data/data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, x_index))
        ddw_df.to_csv('data/data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, x_index))
    print('black creek bankfull with is {} m'.format(np.nanmean(bankfull_width)))
    # save topo-derived bankfull for each transect
    bankfull_results_dt = pd.DataFrame({'bankfull':bankfull_results})
    bankfull_results_dt.to_csv('data/data_outputs/{}/transect_bankfull_topo.csv'.format(reach_name))

def calc_derivatives_aggregate(reach_name, d_interval, all_widths_df):
    # average all xsection widths element-wise
    all_widths = all_widths_df['widths']
    max_len = max(len(ls) for ls in all_widths)
    all_widths_padded = [ls + [None] * (max_len - len(ls)) for ls in all_widths]
    # code to calculate element-wise average (from GPT)
    avg_width = [
        np.nanmedian([x for x in elements if x is not None])
        for elements in zip(*all_widths_padded)
    ]
    xs_xvals = get_x_vals(avg_width, d_interval)
    dw = multipoint_slope(10, avg_width, xs_xvals)
    ddw = multipoint_slope(10, dw, xs_xvals)
    # apply rolling avg derivative calcs
    # ID bankfull (one across entire reach), save in csv
    ddw_abs = [abs(i) for i in ddw]
    # Add in upper and lower search bounds on max 2nd deriv bankfull ID
    for i, val in enumerate(avg_width):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
    lower_bound_index = start_width_index + 5 # lower bound set at 0.5m above thalweg
    upper_bound_index = start_width_index + 100 # upper bound set at 10m above thalweg
    max_ddw = np.nanmax(ddw_abs[lower_bound_index:upper_bound_index])
    max_ddw_index = ddw_abs.index(max_ddw)
    ddw = ddw.tolist()
    max_neg_ddw = np.nanmin(ddw[lower_bound_index:upper_bound_index])
    max_neg_ddw_index = ddw.index(max_neg_ddw)
    bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull
    breakpoint()
    # figure this out with a plot...
    # avg_width = avg_width[:400] # chop off noisy end of data
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(15,15))
    # plt.xlim((600,900)) (2200,2500) (50,400)
    axes[0].plot(avg_width)
    axes[0].axvline(max_ddw_index, label='2nd derivative abs maxima')
    axes[0].axvline(max_neg_ddw_index, label='2nd derivative minima', color='black')
    axes[0].set_title('Median channel widths')
    axes[0].set_xlim((50,400))
    axes[1].plot(dw)
    axes[1].axvline(max_ddw_index, label='2nd derivative abs maxima')
    axes[1].axvline(max_neg_ddw_index, label='2nd derivative minima', color='black')
    axes[1].set_xlim((50,400))
    # axes[1].set_ylim((-200,100))
    axes[1].set_title('First derivative')
    axes[2].plot(ddw)
    axes[2].axvline(max_ddw_index, label='2nd derivative abs maxima')
    axes[2].axvline(max_neg_ddw_index, label='2nd derivative minima', color='black')
    axes[2].set_xlim((50,400))
    # axes[2].set_ylim((-1000,1000))
    axes[2].set_title('Second derivative')
    plt.legend()
    
    fig.suptitle('Brankfull slope identification for {} with 10pt rolling avg'.format(reach_name))
    plt.savefig('data/data_outputs/{}_slopes_fig.jpeg'.format(reach_name))

    # there it is! 
    breakpoint()