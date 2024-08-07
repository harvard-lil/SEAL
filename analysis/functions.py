import os
import ast
from scipy import stats
import numpy as np
from collections import Counter

def extract_list(s):
    return ast.literal_eval(s[0])


def convert_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []
    

def category_plot(df, column):
    overlap_, chosen_, rejected_ = [], [], []
    all_ = []
    for n, g in df.groupby("row_id"):
        chosen = g.loc[g.preference == "chosen"]
        rejected = g.loc[g.preference == "rejected"]
        list_v1 = extract_list(list(chosen[column]))
        list_v2 = extract_list(list(rejected[column]))
        set_v1 = set(list_v1)
        set_v2 = set(list_v2)  
        all_.append(set_v1)
        all_.append(set_v2)
        if set_v1 != set_v2:
            difference_v1 = set_v1 - set_v2
            difference_v2 = set_v2 - set_v1
            overlap = set_v1.intersection(set_v2)
            if overlap == set_v1 and overlap == set_v2:
                continue
            else:
                overlap_.append(overlap)
                chosen_.append(difference_v1)
                rejected_.append(difference_v2)
    return overlap_, chosen_, rejected_, all_


def prepare_data(df, what):
    df = df.replace({True: 1, False: 0})
    df = df.replace({"negative": -1, "neutral": 0, "positive":1})
    df = df.replace({"chosen": 1, "rejected": 0})

    #df['topics'] = df['topics'].apply(convert_to_list)
    #df['last_response_is_discriminating_against'] = df['last_response_is_discriminating_against'].apply(convert_to_list)
    #df['last_response_sentiment'] = df['last_response_sentiment'].apply(convert_to_list)
    
    if what[0]==1:
        unique_strings = set().union(*df['topics'])
        for string in unique_strings:
            df[string] = df['topics'].apply(lambda x: string in x).astype(int)
    
    if what[1]==1:
        unique_strings = set().union(*df['last_response_is_discriminating_against'])
        for string in unique_strings:
            df[string] = df['last_response_is_discriminating_against'].apply(lambda x: string in x).astype(int)
    
    if what[2]==1:
        unique_strings = set().union(*df['last_response_sentiment'])
        for string in unique_strings:
            df[string] = df['last_response_sentiment'].apply(lambda x: string in x).astype(int)
    
    #df_tree = df.drop(['topics', 'last_response_is_discriminating_against', 'text', 'last_response_sentiment'], axis=1)
    df_tree = df.copy()

    #print("Dataset Length: ", len(df_tree))
    #print("Dataset Shape: ", df_tree.shape)
    #print("Dataset: ", df_tree.head())
    #print(df_tree.columns)
    
    return df_tree


def paired_t_test_split(df, target_col, id_col):
    df["unique_id"] = df[id_col].astype(str) + df[target_col]
    positive_ids = df[df[target_col] == 'chosen']["unique_id"].unique()
    negative_ids = df[df[target_col] == 'rejected']["unique_id"].unique()
    
    feature_cols = [col for col in df.columns if col not in [target_col, id_col, "unique_id"]]
    p_values = {}
    
    for col in feature_cols:
        positive_values = df[df["unique_id"].isin(positive_ids)][col]
        negative_values = df[df["unique_id"].isin(negative_ids)][col]
        
        _, p_value = stats.ttest_rel(positive_values, negative_values)
        p_values[col] = p_value
        
    sorted_features = sorted(p_values, key=p_values.get)
    return sorted_features


def compute_angle(a, b):
    #return (a[0]*b[0] + a[1]*b[1])/(np.sqrt((a[0]**2 + a[1]**2)*(b[0]**2 + b[1]**2)))
    vec1 = a / np.linalg.norm(a)
    vec2 = b / np.linalg.norm(b)
    dot_product = np.dot(vec1, vec2)

    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    cos_angle = dot_product / (magnitude_vec1 * magnitude_vec2)
    signed_angle_rad = np.arccos(cos_angle)
    signed_angle_deg = np.degrees(signed_angle_rad)
    cross_product = np.cross(vec1, vec2)
    if cross_product < 0:
        signed_angle_deg = -signed_angle_deg
        signed_angle_rad = -signed_angle_rad
    return signed_angle_deg, signed_angle_rad


def compute_angle_360(a, b):
    #return (a[0]*b[0] + a[1]*b[1])/(np.sqrt((a[0]**2 + a[1]**2)*(b[0]**2 + b[1]**2)))
    vec1 = a / np.linalg.norm(a)
    vec2 = b / np.linalg.norm(b)
    dot_product = np.dot(vec1, vec2)

    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    cos_angle = dot_product / (magnitude_vec1 * magnitude_vec2)
    signed_angle_rad = np.arccos(cos_angle)
    signed_angle_deg = np.degrees(signed_angle_rad)
    cross_product = np.cross(vec1, vec2)
    if cross_product < 0:
        signed_angle_deg = 180 + signed_angle_deg
        signed_angle_rad = np.pi + signed_angle_rad
    return signed_angle_deg, signed_angle_rad


def compute_cos(a, b):
    return (a[0]*b[0] + a[1]*b[1])/(np.sqrt((a[0]**2 + a[1]**2)*(b[0]**2 + b[1]**2)))


def compute_I_0(lab, df):
    disagreement = {}
    dif = []
    for i, group in df[lab + ["row_id", "preference"]].groupby("row_id"):
        g_c = group.loc[group.preference == "chosen"][lab].values
        g_r = group.loc[group.preference == "rejected"][lab].values
        disagreement[i] = len(lab) - sum((g_c == g_r)[0])
        if sum((g_c == g_r)[0]) < len(lab) :
            dif.append(list(g_c - g_r)[0])
    print(Counter(disagreement.values()))
    print(Counter(disagreement.values())[0]/(len(df)/2))
    return dif, disagreement


def circular_hist(ax, x, alpha, label, color, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    print(x, bins)
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, fill=True, linewidth=1, alpha = alpha, label = label, color=color)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
    ax.set_ylabels([])

    return n, bins, patches


def plot_est_all(ax, estimates, std_errors, p_values, variables, col):
    x_pos = np.arange(len(variables))

    colors = []
    ki = 0
    ind_ki = []
    for p_value in p_values:
        if p_value < 0.05:
            colors.append('#91cd58')
        elif p_value > 2:
            colors.append("#72bbff")
        else:
            colors.append(col)
    for i, (estimate, std_error, color) in enumerate(zip(estimates, std_errors, colors)):
        ax.errorbar(x_pos[i], estimate, yerr=std_error, fmt='o', capsize=5, color=color)
    plot = ax.scatter(x_pos, estimates, c=colors)
    
    return plot

print("3a")