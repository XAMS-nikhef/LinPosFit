import time
import scipy
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib 
import strax
import straxen
import pickle

from numba import njit
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import sem
from IPython.display import clear_output

font = {'size':17}
matplotlib.rc('font',**font)
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = 10, 10

inch = 2.54 # cm
pmt_surface=(3*inch)**2*np.pi/4.
pmt_to_lxe = 7.0
npmt_top_array = 253#494

global pmt_pos

pmt_pos = straxen.pmt_positions()
pmt_pos = list(zip(pmt_pos['x'].values,pmt_pos['y'].values,np.repeat(pmt_to_lxe, npmt_top_array)))

pmt_pos = np.asarray(pmt_pos)

# 
# D.C. Konijn, Spring and Summer of 2021
#

def clustering_for_k(k, list_pmt, list_hits):
    """
    Cluster all the data in k clusters given the positions and hits on each pmt
    :param k: amount of clusters in the dataset
    :param pmt_pos: position of the photo-multiplier tubes
    :param hits: list of number of hits on each pmt
    :return kmeans: the clustered data
    """
    #clustering according to scipy kmeans, added is the weight of each hit
    kmeans = KMeans(n_clusters=k, random_state=0).fit(list_pmt, sample_weight = list_hits)
    
    return kmeans

@njit
def loop_clusters(k, list_pmt, list_hits, labels, cluster_points):
    """
    Calculate the within cluster distance metric
    :param k: amount of clusters in the dataset
    :param list_pmt: position of the photo-multiplier tubes
    :param list_hits: list of number of hits on each pmt
    :param labels: the labels on which pmt belongs to which cluster
    :cluster_points: list to loop over all points in one cluster per loop
    :return W: the within-dispersion measures for specific k
    """
        
    W = 0
    D = np.zeros(k)
    points_in_cluster = np.zeros(k)
    
    #loop through all the k clusters to determine the weighted "D"istance of each cluster 
    for i in range(0,k):
        p_counter = 0
  
        for j in cluster_points[i]:
            for m in cluster_points[i]:
                #the squared euclidian distance times the weights of each pmt
                if list_hits[j] != 0 and list_hits[m] != 0:
                    D[i] += (list_hits[j]*list_hits[m])*((list_pmt[j][0]-list_pmt[m][0])**2 +(list_pmt[j][1]-list_pmt[m][1])**2)
           
            p_counter += list_hits[j]
            
        points_in_cluster[i] = p_counter
    
    #loop through all the clusters to determine the "W"ithin cluster distance
    for i in range(len(points_in_cluster)):
        if points_in_cluster[i] != 0:
            W += (1/(2 * points_in_cluster[i]))*D[i]
            
    return W

def within_cluster_distance(k, hits):
    """
    Initialise and calculates the within-cluster distancs for specific k
    :param k: amount of clusters in the dataset
    :param pmt_pos: position of the photo-multiplier tubes
    :param hits: list of number of hits on each pmt
    :return W: the within-cluster-distance metric for specific k
    """

    #remove from the pmt- and the hit-list all events where hits = 0
    list_pmt = pmt_pos[hits != 0]
    list_hits = hits[hits != 0]
    
    #𝅘𝅥𝅮 cluster time 𝅘𝅥𝅮
    kmeans = (clustering_for_k(k, list_pmt, list_hits))
    
    labels = kmeans.labels_
    cluster_points = []
    
    for i in range(0,k):
        #for each cluster appends an array of point-indices that belong to that specific cluster
        cluster_points.append(np.asarray(np.where(np.asarray(labels) == i))[0])
    
    W = loop_clusters(k, list_pmt, list_hits, labels, cluster_points)

    return W

def mini_cluster(hits):
    """
    Calculate the optimal number of clusters for very small amount of pmt's with hits
    :param pmt_pos: position of the photo-multiplier tubes
    :param hits: number of hits on each pmt
    :return k: the optimal number of clusters for a small dataset
    """

    #remove from the pmt- and the hit-list all events where hits = 0
    list_pmt = pmt_pos[hits != 0]
    list_hits = hits[hits != 0]
    
    #sort both lists on highest hit count
    list_hits, list_pmt = zip(*sorted(zip(-np.asarray(list_hits), -np.asarray(list_pmt))))
    list_hits = -np.asarray(list_hits)
    list_pmt = -np.asarray(list_pmt)
    
    #create a list of each nonzero hit
    cluster_list = np.asarray(list(range(0,len(list_hits))))
    
    i = 0
    while i < len(cluster_list):
        j = cluster_list[i]
            
        #delete from the list each hit that lies within 27.5mm euclidian distance
        for k in (cluster_list):
            distance = ((list_pmt[j][0] - list_pmt[k][0])**2 + (list_pmt[j][1] - list_pmt[k][1])**2)**0.5

            if distance < 27.5 and k != j:
                cluster_list = np.delete(cluster_list, np.where(cluster_list == k)[0][0])            
        
        i += 1
    
    return len(cluster_list)

def new_cluster_execute(hits, switch_param = 3, linehit = 0.2680932125849651):
    """
    Calculate the most optimal amount of clusters in the data
    :param pmt_pos: position of the photo-multiplier tubes
    :param hits: number of hits on each pmt
    :param switch_param: the number of nonzero hits at which the method changes to the mini-cluster method
    :return optimum: most optimal amount of clusters
    """
    #for events with three or less nonzero hits, the distance between both points becomes a required quantity
    if len(np.where(np.asarray(hits))[0]) <= switch_param:
        
        optimum = mini_cluster(hits)
        return optimum
        
    else:
        wk1 = within_cluster_distance(1, hits)
        wk2 = within_cluster_distance(2, hits)

        ratio = (wk1-wk2)/(wk1+wk2)
        
        #0.2680932125849651 is 97.5% of 2% treshold single scatters
        if ratio > linehit:
            return 2

        return 1
    
def cluster_plot(optimum, hits):
    """
    Plot the data into calculated clusters
    :param optimum: the amount of clusters in the data
    :param pmt_pos: position of the photo-multiplier tubes
    :param hits: number of hits on each pmt
    """
    
    try:
        colors = {0: "red", 1: "green"}

        #remove from the pmt- and the hit-list all events where hits = 0
        list_pmt = pmt_pos[hits != 0]
        list_hits = hits[hits != 0]
    
        #𝅘𝅥𝅮 cluster time 𝅘𝅥𝅮
        kmeans = (clustering_for_k(optimum, list_pmt, list_hits))

        fig = plt.gcf()
        ax = fig.gca()

        #for each point in each cluster, plot a circle. the radius of the circle is related to the weight of the hit
        for i in range(optimum):
            cluster_points = np.asarray(np.where(np.asarray(kmeans.labels_) == i))[0]
            for j in cluster_points:
                circle = plt.Circle((list_pmt[j][0], list_pmt[j][1]), np.sqrt(list_hits[j])/10, color=colors[i], fill=False)
                ax.add_patch(circle)

        plt.scatter(pmt_pos[:, 0], pmt_pos[:, 1], s=1)
        plt.xlim(-100,100)
        plt.xlabel("x(mm)")
        plt.ylabel("y(mm)")
        plt.ylim(-100,100)       
    
    except TypeError:
        print("No optimum number of clusters was found, so no plot can be made") 
      
    return 

#
#as per request, my thesis figures into functions that are easy callable
#

def show_true_dist(area_per_channel, true_pos, iterations, percentage = 0.02, switch_param = 2):
    """
    Plot the distribution of the true distance 
    :param area_per_channel: array of datasets
    :param true_pos: true positions of array of datasets
    :param iterations: number of hits in distribution
    :param percentage: background threshold percentage
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    hist_list_one = []
    hist_list_two = []

    for ip in range(iterations):
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1000][:253])

        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0

        #cutoff-point is at 97.5% of single scatters with background trehshold at 0.025%
        optimum = new_cluster_execute(hits, switch_param, 0.27359306097095776)

        true_dist = ((true_pos[ip][0]-true_pos[1000+ip][0])**2 + (true_pos[ip][1]-true_pos[1000+ip][1])**2)**0.5

        if optimum == 1:
            hist_list_one.append(true_dist)
        if optimum == 2: 
            hist_list_two.append(true_dist)
            
    font = {'size':33}
    matplotlib.rc('font',**font)
    plt.rcParams['figure.figsize'] = 12, 12

    plt.hist(hist_list_two, bins=75, range=(0,150), label='Good identification', fill=True, color = 'green', histtype='step', alpha=0.4)
    plt.hist(hist_list_two, bins=75, range=(0,150), fill=False, histtype='step', color='black', linewidth = 1)
    plt.hist(hist_list_one, bins=75, range=(0,150), label='Misidentification', fill=True, color = 'red', histtype='step', alpha = 0.4, edgecolor = 'black', linewidth = 1)
    # plt.hist(hist_list_one, bins=75, range=(0,150), fill=False, histtype='step', color='black', linestyle = (0,(1,4)), linewidth = 3)
    plt.hist(hist_list_one, bins=75, range=(0,150), fill=False, histtype='step', color='black', linewidth = 1)

    plt.xlabel("$\mathcal{D}$(mm)")
    plt.ylabel("#Entries/2mm")
    plt.legend(fontsize = 25)
    plt.show()
    
def data_cluster_execute(area_per_channel, single_double, iterations, display, percentage = 0.02, switch_param = 3):
    """
    Calculate the optimal number of clusters in the data
    :param area_per_channel: array of datasets
    :param single_double: input is 'single' or 'double' hit
    :param iterations: number of hits in distribution
    :param percentage: background threshold percentage
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    for ip in range(iterations):        
        
        if single_double == 'single':
            hits = np.array(area_per_channel[ip][:253])
            hits_below_thresh = hits < (max(hits) * percentage)
            hits[hits_below_thresh] = 0
            
        elif single_double == 'double':
            hits1 = np.array(area_per_channel[ip][:253])
            hits2 = np.array(area_per_channel[ip+1][:253])
            hits = hits1 + hits2

            hits_below_thresh = hits < (max(hits) * percentage)
            hits[hits_below_thresh] = 0
        
        optimum = new_cluster_execute(hits, switch_param)
        
        if display:
            cluster_plot(optimum, hits)  
            plt.show()
            
            try:
                istat = int(input("Type: 0 to continue, 1 to quit...."))
            except ValueError:
                print('The possible inputs are 0, 1 and 2.')
                break
            
            if istat == 1:
                break

            clear_output()
    return 

def eta_hist(area_per_channel, iterations, percentage = 0.02, switch_param = 2):
    """
    Plot the distribution of eta
    :param area_per_channel: array of datasets
    :param iterations: number of hits in distribution
    :param percentage: background threshold percentage
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    hist_list = []
    hist_list2 = []

    for ip in range(iterations):
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1000][:253])

        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0

        v1=within_cluster_distance(1, hits)
        v2=within_cluster_distance(2, hits)

        ratio = (v1-v2)/(v1+v2)

        hist_list.append(ratio)

    for ip in range(iterations):
        hits1 = np.array(area_per_channel[ip][:253])
        hits = hits1
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0

        v1=within_cluster_distance(1, hits)
        v2=within_cluster_distance(2, hits)

        ratio = (v1-v2)/(v1+v2)

        hist_list2.append(ratio)

    line_hit = np.percentile(hist_list2, 97.5)
    ax = plt.gca()
    xticks = ax.xaxis.get_major_ticks() 
    xticks[0].label1.set_visible(False)
    plt.axvline(x=line_hit, color = 'k', linewidth=1.5, linestyle='dashed')
    plt.hist(hist_list2, bins=100, range=(0,1), label='Single Scatter', fill=True, color = 'yellow', histtype='step', alpha = 0.45, ec = 'black')
    # plt.hist(hist_list2, bins=100, range=(0,1), fill=False, histtype='step', color='black', linestyle = (0,(1,4)), linewidth = 3)
    plt.hist(hist_list2, bins=100, range=(0,1), fill=False, histtype='step', color='black', linewidth = 1)
    plt.hist(hist_list, bins=100, range=(0,1), label='Double Scatter', fill=True, color = 'gray', histtype='step', alpha=0.35)
    plt.hist(hist_list, bins=100, range=(0,1), fill=False, histtype='step', color='black')
    plt.xlabel("$\\eta$" , size = 35)
    plt.ylabel("#Entries/0.01", size = 35)
    plt.legend(fontsize = 25, edgecolor = 'black', frameon=False)
    
    maxy = 3000

    plt.arrow(line_hit + 0.01, maxy, 0, -260, width = 0.0001)
    plt.annotate("", xy=(line_hit+0.08, maxy - 262), xytext=(line_hit+0.0075, maxy - 262), arrowprops=dict(arrowstyle="->", linewidth=1))
    plt.annotate("Do-S", xy=(line_hit+0.08, maxy - 260), xytext=(line_hit+0.015, maxy - 220), size=25)

    plt.arrow(line_hit - 0.01, maxy, 0, -179, width = 0.0001)
    plt.annotate("", xy=(line_hit-0.007, maxy-180), xytext=(line_hit-0.08, maxy - 180), arrowprops=dict(arrowstyle="<-", linewidth=1))
    plt.annotate("Si-S", xy=(line_hit-0.02, maxy-150), xytext=(line_hit-0.10, maxy-140), size=25)
    plt.ylim(0,maxy)
    plt.xlim(0,1)
    
    plt.show()
    
def wk1_wk2_plot(area_per_channel, percentage = 0.02, switch_param = 3):
    """
    Plot wk1 and wk2 into a single plot
    :param area_per_channel: array of datasets
    :param percentage: background threshold percentage
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    ip = 133

    hits1 = np.array(area_per_channel[ip][:253])
    hits2 = np.array(area_per_channel[ip+6][:253])
    hits = hits1 + hits2

    hits_below_thresh = hits < (max(hits) * percentage)
    hits[hits_below_thresh] = 0

    l2 = []
    x2 = []

    for i in range(1,6):
        l2.append(np.log(within_cluster_distance(i, hits)))
        x2.append(i)

    hits = np.array(area_per_channel[ip][:253])
    hits_below_thresh = hits < (max(hits) * percentage)
    hits[hits_below_thresh] = 0

    l = []
    x = []

    for i in range(1,6):
        l.append(np.log(within_cluster_distance(i, hits))+2.99)
        x.append(i)

    plt.grid()
    plt.plot(x2, l2, linewidth = 4, label='Double Cluster')
    plt.scatter(x2, l2, s = 120)
    plt.plot(x, l, linewidth = 4, label='Single Cluster')
    plt.scatter(x, l, s = 120)
    plt.xlabel('Amount of clusters $k$')
    plt.ylabel("Log($W_k$)")
    plt.legend()
    plt.show()
    
def jusitification_classification(area_per_channel, iterations, percentage = 0.02,switch_param = 2):
    """
    Plot the justification of the classification line
    :param area_per_channel: array of datasets
    :param iterations: number of hits in distribution
    :param percentage: background threshold percentage
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    hist_list = []
    hist_list2 = []    
    cuteff = []
    cutmiss = []
    label_list=[]

    line_hit = [0.0, 0.06666666666666667, 0.13333333333333333, 0.2, 0.2222222, 0.2444444, 0.26666666666666666, 0.30, 0.3333333333333333, 0.4, 0.4666666666666667, 0.5333333333333333, 0.6, 0.6666666666666666, 0.7333333333333333, 0.8, 0.8666666666666667, 0.9333333333333333, 1.0]


    for ip in range(iterations):
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1000][:253])

        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0

        v1=within_cluster_distance(1, hits)
        v2=within_cluster_distance(2, hits)

        ratio = (v1-v2)/(v1+v2)

        hist_list.append(ratio)

    for ip in range(iterations):
        hits1 = np.array(area_per_channel[ip][:253])
        hits = hits1
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0

        v1=within_cluster_distance(1, hits)
        v2=within_cluster_distance(2, hits)

        ratio = (v1-v2)/(v1+v2)

        hist_list2.append(ratio)

    for i in range(len(line_hit)):
        cuteff.append(len(np.nonzero(np.asarray(hist_list2) < line_hit[i])[0]) / len(hist_list2))
        cutmiss.append(len(np.nonzero(np.asarray(hist_list) < line_hit[i])[0]) / len(hist_list))
        label_list.append(line_hit[i])

    fig, ax = plt.subplots()
    
    for i, txt in enumerate(label_list):
        if i > 8 and i < len(cuteff)-2:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.01, cutmiss[i]), fontsize = 20)
        if i > len(cuteff)-2:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.01, cutmiss[i]+0.01), fontsize = 20)
        if i < 4 and i > 2:
            ax.annotate(np.round(txt,2), (cuteff[i]-0.01, cutmiss[i]-0.037), fontsize = 20)
        if i < 1:
            ax.annotate(np.round(txt,2), (cuteff[i]-0.009, cutmiss[i]-0.035), fontsize = 20)
        if i == 8:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.014, cutmiss[i]-0.003), fontsize = 20)
        if i == 7:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.019, cutmiss[i]-0.008), fontsize = 20)
        if i == 6:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.027, cutmiss[i]-0.011), fontsize = 20)
        if i == 5:
            ax.annotate(np.round(txt,2), (cuteff[i]-0.017, cutmiss[i]-0.04), fontsize = 20)
        if i == 4:
            ax.annotate(np.round(txt,2), (cuteff[i]+0.01, cutmiss[i]-0.026), fontsize = 20)

    c_eff = list(np.asarray(cuteff))
    c_miss = list(np.asarray(cutmiss))

    del c_eff[-2]
    del c_miss[-2]

    ax.plot(single_eff, double_miss, c='k', linewidth = 2)
    ax.scatter(c_eff, c_miss, c='k', s = 50)
    plt.xlabel("Single Scatter Efficiency (%)" , size = 35)
    plt.ylabel("Double Scatter Misidentification (%)" , size = 35)
    plt.axhline(y=1, linestyle = 'dashed', c = 'k')
    plt.axvline(x=1, linestyle = 'dashed', c = 'k')
    plt.axvline(x=0.975, label='97.5%', c='b', linestyle='dashed')
    plt.xlim(0.45,1.07)
    plt.ylim(0,1.05)
    plt.show()

def func(amp, eff):
    """
    Makes sure the bin size of the single efficiency distribution is logarithmic
    """
    
    bins = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 2000, 3000])
    
    inds = np.digitize(amp, bins)

    ind= [[] for _ in range(len(bins))]
    pro= [[] for _ in range(len(bins))]

    for i in range(len(inds)):
        value = inds[i]
        ind[value].append(amp[i])
        pro[value].append(eff[i])

    for i in range(len(ind)):
        if len(ind[i]) == 0:
            ind[i] = [0]
        ind[i] = sum(ind[i])/len(ind[i])
        if len(pro[i]) == 0:
            pro[i] = [0]
        pro[i] = sum(pro[i])/len(pro[i])

    sorted_zeros = sorted(-np.argwhere(np.asarray(ind) == 0))

    for i in range(len(sorted_zeros)):
        del ind[-sorted_zeros[i][0]]
        del pro[-sorted_zeros[i][0]]
        
    return (ind, pro)
    
def single_cluster_eff(area_per_channel, mc_amplitude, loop_time, iterations, each_block, switch_param = 2):
    """
    Plot the single scatter efficiency
    Make sure that (iterations * each_block) = loop_time
    :param area_per_channel: array of datasets
    :param mc_amplitude: the monte carlo amplitude of the hits in the datasets
    :param loop_time: the number of datasets used
    :param iterations: the number of ticks on the x-axis
    :param each_block: the number of datasets used per tick
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
    
    optimum = [[],[],[]]
    maximum_amplitude = []
    
    for ip in range(loop_time):
        percentage = 0
        hits = np.array(area_per_channel[ip][:253])
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[0].append(new_cluster_execute(hits, switch_param, 0.39481508101094803))

        percentage = 0.02
        hits = np.array(area_per_channel[ip][:253])
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[1].append(new_cluster_execute(hits, switch_param, 0.2680932125849651))

        percentage = 0.05
        hits = np.array(area_per_channel[ip][:253])
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[2].append(new_cluster_execute(hits, switch_param, 0.30755463539466177))

        maximum_amplitude.append(mc_amplitude[ip])

    maximum_amplitude0, optimum[0] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[0]))))
    maximum_amplitude2, optimum[1] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[1]))))
    maximum_amplitude5, optimum[2] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[2]))))

    eff_0=[]
    eff_2=[]
    eff_5=[]

    amp_0=[]
    amp_2=[]
    amp_5=[]

    for i in range(iterations):

        cut_0 = np.asarray(optimum[0])[(i)*each_block:(i+1)*each_block]
        if len(cut_0) != 0:
            eff_0.append((len(cut_0[cut_0 == 1])) / len(cut_0))
            amp_0.append(np.average(maximum_amplitude0[(i)*each_block:(i+1)*each_block]))

        cut_2 = np.asarray(optimum[1])[(i)*each_block:(i+1)*each_block]
        if len(cut_2) != 0:
            eff_2.append((len(cut_2[cut_2 == 1])) / len(cut_2))
            amp_2.append(np.average(maximum_amplitude2[(i)*each_block:(i+1)*each_block]))

        cut_5 = np.asarray(optimum[2])[(i)*each_block:(i+1)*each_block]
        if len(cut_5) != 0:
            eff_5.append((len(cut_5[cut_5 == 1])) / len(cut_5))
            amp_5.append(np.average(maximum_amplitude5[(i)*each_block:(i+1)*each_block]))

    #uses the function to scale it logarithmically
    ind1, pro1 = func(amp_0, eff_0)
    ind2, pro2 = func(amp_2, eff_2)
    ind3, pro3 = func(amp_5, eff_5)

    font = {'size':33}
    matplotlib.rc('font',**font)
    plt.rcParams['figure.figsize'] = 12, 12
    plt.axhline(y=1, c='k', linestyle='dashed')
    plt.errorbar(ind1, pro1, yerr=((np.asarray(pro1)*1000)**0.5)/1000, fmt='o', c='blue', label = 'Threshold = 0%', markersize = 10)
    plt.errorbar(ind2, pro2, yerr=((np.asarray(pro2)*1000)**0.5)/1000, fmt='^', c='orange', label = 'Threshold = 2%', markersize = 10)
    plt.errorbar(ind3, pro3, yerr=((np.asarray(pro3)*1000)**0.5)/1000, fmt='s', c='green', label = 'Threshold = 5%', markersize = 10)
    plt.plot(ind1, pro1, linewidth = 2, c='blue')
    plt.plot(ind2, pro2, linewidth = 2, c='orange')
    plt.plot(ind3, pro3, linewidth = 2, c='green')
    plt.xlabel("Monte-Carlo Amplitude")
    # plt.xscale('log')
    plt.ylabel("Efficiency",labelpad=10)
    plt.legend(fontsize = 25, frameon=False)
    plt.xscale('log')
    plt.ylim(0,1.05)
    plt.show()
    
def double_cluster_eff(area_per_channel, mc_amplitude, loop_time, iterations, each_block, switch_param = 2):
    """
    Plot the double scatter efficiency
    Make sure that (iterations * each_block) = loop_time
    :param area_per_channel: array of datasets
    :param mc_amplitude: the monte carlo amplitude of the hits in the datasets
    :param loop_time: the number of datasets used
    :param iterations: the number of ticks on the x-axis
    :param each_block: the number of datasets used per tick
    :param switch_param: maximum number of pmts with hits at which the algorithm switches to a simple cluster finding algorithm
    """
        
    optimum = [[],[],[]]
    maximum_amplitude = []

    for ip in range(loop_time):
        percentage = 0
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1][:253])
        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[0].append(new_cluster_execute(hits, switch_param, 0.39481508101094803))

        percentage = 0.02
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1][:253])
        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[1].append(new_cluster_execute(hits, switch_param, 0.2680932125849651))

        percentage = 0.05
        hits1 = np.array(area_per_channel[ip][:253])
        hits2 = np.array(area_per_channel[ip+1][:253])
        hits = hits1 + hits2
        hits_below_thresh = hits < (max(hits) * percentage)
        hits[hits_below_thresh] = 0
        optimum[2].append(new_cluster_execute(hits, switch_param, 0.30755463539466177))

        new_ratio = np.abs((mc_amplitude[ip]- mc_amplitude[ip+1]) / (mc_amplitude[ip] + mc_amplitude[ip+1]))
        maximum_amplitude.append(new_ratio)

    maximum_amplitude0, optimum[0] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[0]))))
    maximum_amplitude2, optimum[1] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[1]))))
    maximum_amplitude5, optimum[2] = zip(*sorted(zip(np.asarray(maximum_amplitude), np.asarray(optimum[2]))))

    eff_0=[]
    eff_2=[]
    eff_5=[]

    amp_0=[]
    amp_2=[]
    amp_5=[]

    for i in range(iterations):

        cut_0 = np.asarray(optimum[0])[(i)*each_block:(i+1)*each_block]
        if len(cut_0) != 0:
            eff_0.append((len(cut_0[cut_0 == 2])) / len(cut_0))
            amp_0.append(np.average(maximum_amplitude0[(i)*each_block:(i+1)*each_block]))

        cut_2 = np.asarray(optimum[1])[(i)*each_block:(i+1)*each_block]
        if len(cut_2) != 0:
            eff_2.append((len(cut_2[cut_2 == 2])) / len(cut_2))
            amp_2.append(np.average(maximum_amplitude2[(i)*each_block:(i+1)*each_block]))

        cut_5 = np.asarray(optimum[2])[(i)*each_block:(i+1)*each_block]
        if len(cut_5) != 0:
            eff_5.append((len(cut_5[cut_5 == 2])) / len(cut_5))
            amp_5.append(np.average(maximum_amplitude5[(i)*each_block:(i+1)*each_block]))

    fig,ax = plt.subplots()
    xticks = ax.xaxis.get_major_ticks() 
    xticks[0].label1.set_visible(False)

    plt.plot(amp_0, eff_0, linewidth = 2)
    plt.plot(amp_2, eff_2, linewidth = 2)
    plt.plot(amp_5, eff_5, linewidth = 2)
    plt.errorbar(amp_0, eff_0, ((np.asarray(eff_0)*each_block)**0.5)/each_block, label = 'Threshold = 0%', fmt='o', c='blue', markersize = 10, alpha = 0.8)
    plt.errorbar(amp_2, eff_2, ((np.asarray(eff_2)*each_block)**0.5)/each_block, label = 'Threshold = 2%', fmt='^', c='orange',  markersize = 10,alpha = 0.8)
    plt.errorbar(amp_5, eff_5, ((np.asarray(eff_5)*each_block)**0.5)/each_block, label = 'Threshold = 5%', fmt='s', c='green',  markersize = 10,alpha = 0.8)
    plt.xlabel("$\\alpha$")
    plt.ylabel("Efficiency",labelpad=10)
    plt.legend(fontsize = 25, frameon=False)
    plt.xlim(0,1)
    plt.ylim(0,1.05)
    
    plt.show()