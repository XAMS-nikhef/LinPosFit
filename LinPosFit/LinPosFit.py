import numpy as np
from numba import njit

# from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from IPython.display import clear_output
import seaborn as sns

# lpf = linearized position fit
# 
# A.P. Colijn -  Feb. 2021
lpf_iter_max = 100
lpf_min_position_cor = 0.01 # convergence criterium for position minimizer
lpf_npar = 4 # number of fit parameters
#                       x      y      r0     gamma
lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix
#
debug = False


def lpf_execute(xhit, nhit, area_sensor):
    #%snakeviz
    """
    Execute the fitter
    :param xhit:
    :param nhit:
    :param area_sensor:
    :param kwargs:
    :return:
    """
    xhit = np.array(xhit)
    nhit = np.array(nhit)
    # minimize the likelihood
    fit_result, xiter, hit_is_used = lpf_minimize(xhit, nhit, area_sensor)
    return fit_result, xiter, hit_is_used

@njit
def lpf_factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

@njit
def lpf_lnlike(xhit, nhit, xf, nuv, gamma, area_sensor):
    r0 = nuv * area_sensor * xhit[0][2] / 4 / np.pi
    logl = 0

    # print('next')
    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        # print(xhit[ih], nexp, xf, r0)

        logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl

@njit
def lpf_lnlike_plot(xhit, nhit, hit_is_used, xf, r0, gamma):
    logl = 0

    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        d = lpf_dist(xhit[ih],xf)
        if hit_is_used[ih] == 1:
            logl = logl + nexp - nhit[ih] * np.log(nexp)

    return logl

@njit
def lpf_lnlike_r(xhit, nhit, xf, r0, gamma):
    logl = 0

    for ih in range(len(nhit)):
        nexp = lpf_nexp(xhit[ih], xf, r0, gamma)
        logl = logl + nexp - nhit[ih] * np.log(nexp)
 
    return logl

@njit
def lpf_initialize(xhit, nhit):
    """
    Initialize the lpf_fitter. Estimate the initial position and uv photons

    :param xhit:
    :param nhit:
    :return:
    """
    nmax = -1

    xhit_max = np.zeros(3)
    # sort hits in descending order. helps when looping over the hits
    #
    indices = np.argsort(-nhit) # descending order
    
    # find a cluster of hits
    #
    # 1. we start with the highest hit in the event
    # 2. if it is surrrounded by other hits then we found a cluster and we are done
    # 3. if not: continue at 1. with the second highest hit in the event
    # 4. repeat until we checked all possibilities
    # 5. if no cluster was found we use the highest hit as the initial fit position
    #
    
    # only consider hits within distance dist_min
    #
    dist_min = 25.
    #  - maximum fraction of the cluster signal allowed to be inside the leading hit
    #  - below this fraction we assume that we are dealing with a hit cluster
    hit_frac_cut = 0.75
    found_cluster = False
    
    # loop starting from the highest hit
    #
    xcluster = np.zeros(3)
    for i in range(len(indices)-1):
        index = indices[i]
        xhit_max = xhit[index]
        # the seed hit has signal xmax
        #
        nmax = nhit[index]
        
        # we have reached the zero-signal PMTs..... no cluster was found, so leave this loop
        #
        if nmax == 0:
            break
        
        # n_around indicates the amount of signal that is found around the seed hit
        #
        n_around = 0
        
        # check the other hits....
        #
        xcluster = nhit[index]*xhit[index]
        for j in range(i+1, len(indices)):
            jindex = indices[j]
            xtest = xhit[jindex]
            dist = lpf_dist(xhit_max, xtest)
            # consider the hit if it is near the seed hit
            #
            if dist<dist_min:
                n_around = n_around + nhit[jindex]
                xcluster = xcluster + nhit[jindex]*xhit[jindex]
        # calculate the fraction of the cluster signal in the seed hit
        #
        hit_frac = 1.*nmax/(nmax+n_around)
        # if the fraction is below hit_frac_cut we are dealing with a cluster 
        # 
        if hit_frac<hit_frac_cut:
            found_cluster = True
            break
            
    # no cluster was found... best guess is to use the hit with the maximum signal as initial position
    #
    if not found_cluster: # use maximum hit
        nmax = nhit[indices[0]]
        xhit_max = xhit[indices[0]]
    else: # we calculate the center-of-gravity as initial position
        xhit_max = xcluster / (nmax+n_around)
    
    if debug:
        print('--- lpf_initialize --- hit fraction = ',hit_frac)
        print('--- lpf_initialize --- found_cluster = ',found_cluster)
    
    # refine the determination of the initial position
    # logl_min = 1e12
    # xtemp = np.zeros(3)
    xfit = np.zeros(3)
    xfit[0] = xhit_max[0]
    xfit[1] = xhit_max[1]
    xfit[2] = 0.
    # estimate the fitted rate
    #
    r0 = nmax * xhit_max[2] ** 3

    return xfit, r0

@njit
def lpf_hit_prob(xi, ni, xf, r0, gamma):
    nexp = lpf_nexp(xi, xf, r0, gamma)
    if ni<10: # explicit calculation of ln(n!)
        nfac = lpf_factorial(ni)
        log_nfac = np.log(nfac)
    else: # use Stirlings formula to approximate ln(n!)
        log_nfac = ni*np.log(ni) - ni
    log_prob = ni*np.log(nexp) - nexp - log_nfac
    return log_prob

@njit
def lpf_minimize(xhit, nhit, area_sensor):
    """
    Linearized -log(L) fit for S2 position

    :param xhit: positions of the sensors (list with 3D arrays with length n-sensor)
    :param nhit: number of p.e. on each sensor (length n-sensor)
    :param area_sensor: area of teh sensor
    :return: fit_result: [0] = x, [1] = y, [2] = nuv [3] = lnlike [4] = n_in_fit [5] = gamma
    :return: xiter: intermediate fit results
             nuv: number of emitted photons
    """
    # initialize
    #
    xfit, r0 = lpf_initialize(xhit, nhit)
    gamma = 0.0

    # arrays to store teh fit resuults for each iteration
    xiter = np.zeros((lpf_iter_max + 1, 6))
    xstart = np.array([xfit[0],xfit[1],0])
    xiter[0][0] = xfit[0]
    xiter[0][1] = xfit[1]
    xiter[0][2] = r0
    xiter[0][3] = lpf_lnlike_r(xhit, nhit, xfit, r0, gamma)
    xiter[0][4] = 0
    xiter[0][5] = 0
    
    r_max = 10.
    # iterate & minimize
    #
    hit_is_used = np.zeros(len(nhit))
    n_in_fit = 0
    # log_pmin: minimal probability for a hit to be used in teh position fit
    #
    log_pmin = np.log(1e-10)
    nmax = np.amax(nhit)
    for lpf_iter in range(lpf_iter_max):
        # initialize error matrix and vector
        #
        g = np.zeros(lpf_npar)
        m = np.zeros((lpf_npar, lpf_npar))

        # calculate the sums
        #
        # print(lpf_iter,' xin =',xfit,' r0=',r0)
        n_in_fit = 0 # reset after each iteration
        
        if lpf_iter < 2:
            lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix
        else:
            lpf_fix_par = np.array([False, False, False, False]) # which parameters to fix
        # make a list of the active parameters
        active_parameters = np.arange(lpf_npar)[lpf_fix_par == False]

        
        for isensor in range(len(nhit)):
            if lpf_iter==0:
                hit_is_used[isensor] = 0
            # calculate the probability that a hit comes from the assumed xhit and r0
            #
            log_prob = lpf_hit_prob(xhit[isensor], nhit[isensor], xfit, r0, gamma)
            # print(isensor,' xhit =',xhit[isensor],' logP =',np.exp(log_prob),' ni =', nhit[isensor])
            #if (log_prob > log_pmin) and (nhit[isensor]>0):
            if ((lpf_iter == 0)  and (nhit[isensor] > nmax*0) and (lpf_dist(xhit[isensor],xfit) < 150.)) or (hit_is_used[isensor] == 1):
                n_in_fit = n_in_fit+1
                if lpf_iter == 0:
                    hit_is_used[isensor] = 1
                for i in active_parameters: # only fill the error matrix for parameters you want to fit 
                    g[i] = g[i] + lpf_f(i, xhit[isensor], nhit[isensor], xfit, r0, gamma)
                    for j in active_parameters:
                        m[i][j] = m[i][j] + lpf_deriv_f(i, j, xhit[isensor], nhit[isensor], xfit, r0, gamma)

        for i in np.arange(lpf_npar)[lpf_fix_par == True]:
            m[i][i] = 1

        # invert the matrix
        #
        if np.linalg.det(m) == 0.:
            # print('lpf_minimize:: singular error matrix')
            break
        
        minv = np.linalg.inv(m)
        # multiply with vector to get corrections to the current fit parameters
        #
        result = np.dot(minv, g)

        # if abs(result[1])>1000:
        #    print('WARNING:: result = ',result)
        r_new = np.sqrt((xstart[0]-(xfit[0] - result[0]))**2 + (xstart[1] - (xfit[1] - result[1]))**2)
        
        weight = 0.5*(1 + np.math.erf((r_max-r_new)/1.0))
        # weight = 1
        # update fit result
        #if weight<1e-3:
        #    print('ping ping ping.... weight =',weight)
        ###weight = 1.0
        xfit[0] = xfit[0] - result[0]*weight
        xfit[1] = xfit[1] - result[1]*weight
        
        #weight = np.exp(-lpf_dist(xstart,xfit)/25.)
        r0 = r0 - result[2]*weight
            
        gamma = gamma - result[3]*weight

        # and store the intermediate results
        #
        xiter[lpf_iter + 1][0] = xfit[0]
        xiter[lpf_iter + 1][1] = xfit[1]
        xiter[lpf_iter + 1][2] = r0
        xiter[lpf_iter + 1][3] = lpf_lnlike_r(xhit, nhit, xfit, r0, gamma)
        xiter[lpf_iter + 1][4] = n_in_fit
        xiter[lpf_iter + 1][5] = gamma


        # if (lpf_iter>=5) and (abs(result[0]) < 0.01) and (abs(result[1]) < 0.01) or (r0<0):  # if position no longer changes -> terminate loop
        if ((lpf_iter>1) and abs(result[0]) < lpf_min_position_cor) and (abs(result[1]) < lpf_min_position_cor) or (r0<0):  # if position no longer changes -> terminate loop
            break

    # calculate the number of uv photons
    #
    nuv = 4 * np.pi * r0 / area_sensor / xhit[0][2]

    # store the fit results
    #
    fit_result = np.zeros(7)
    fit_result[0] = xfit[0]
    fit_result[1] = xfit[1]
    fit_result[2] = nuv
    
    logl = 0
    for ih in range(len(nhit)):
        if nhit[ih] > 0:
            logprob = lpf_hit_prob(xhit[ih], nhit[ih], xfit, r0, gamma)
            if logprob>log_pmin:
                logl = logl + lpf_hit_prob(xhit[ih], nhit[ih], xfit, r0, gamma)

    fit_result[3] = logl # lpf_lnlike_r(xhit, nhit, xfit, r0)
    fit_result[4] = n_in_fit
    fit_result[5] = gamma
    fit_result[6] = r0


    return fit_result, xiter, hit_is_used


@njit
def lpf_dist(x0, x1):
    d2 = (x0[0] - x1[0]) ** 2 + (x0[1] - x1[1]) ** 2 + (x0[2] - x1[2]) ** 2
    return np.sqrt(d2)


@njit
def lpf_nexp(xi, xf, r0, gamma):
    """
    Calculate the expected number of p.e. for a lightsensor
    
    :param xi: sensor position
    :param xf: assumed hit position
    :param r0: assumed nuumber of UV photons (re-normalized)
    :param gamma: assumed constant background
    :return nexp: number of expected photons
    
    A.P. Colijn
    """
    delta = lpf_dist(xi, xf)
    nexp = r0 / delta ** 3 + gamma

    return nexp

@njit
def lpf_f(i, xi, ni, xf, r0, gamma):
    """
    Calculate the minimizer functions
    :param i:
    0=F0 (x)
    1=F1 (y)
    2=F2 (r0)
    3=F3 (gamma) 
    :param xi: sensor position
    :param ni: hits for the sensor
    :param xf: assumed fit position
    :param r0: assumed number of UV photons (re-normalized)
    :param gamma: assumed constant background
    :return f: function value
    """

    f = 0
    if i < 2:
        f = -3 * (xf[i] - xi[i]) * (lpf_nexp(xi, xf, r0, gamma) - ni) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        f = (lpf_nexp(xi, xf, r0, gamma) - ni) / r0
    elif i == 3:
        f = 1. - ni / lpf_nexp(xi, xf, r0, gamma)

    return f

@njit
def lpf_deriv_f(i, j, xi, ni, xf, r0, gamma):
    """
    Derivatives of the minimizer functions

    :param i:
    0=F0
    1=F1
    2=F2
    :param j:
    0=x
    1=y
    z=r0
    :param xi: hit position
    :param ni: number of hits
    :param xf: fit position
    :param r0: number of photons
    :return:
    """

    d = lpf_dist(xi, xf)
    n0 = lpf_nexp(xi, xf, r0, gamma)

    deriv = 0
    if i == 0:
        dx = xf[0] - xi[0]
        if j == 0:  # dF0/dx
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(0, xi, xf, r0, gamma) / d ** 2
        elif j == 1:  # dF0/dy
            deriv = - 3 * dx * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dx * lpf_deriv_n(1, xi, xf, r0, gamma) / d ** 2
        elif j == 2:  # dF0/dr0
            deriv = -3 * dx * lpf_deriv_n(2, xi, xf, r0, gamma) / d ** 2
        elif j == 3: #dF0/dgamma
            deriv = -3 * dx * lpf_deriv_n(3, xi, xf, r0, gamma) / d ** 2        
    elif i == 1:
        dy = xf[1] - xi[1]
        if j == 0:  # dF1/dx
            deriv = - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(0, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(0, xi, xf, r0, gamma) / d ** 2
        elif j == 1:  # dF1/dy
            deriv = -3 * (n0 - ni) / d ** 2
            deriv = deriv - 3 * dy * (n0 - ni) * lpf_deriv_dist_min2(1, xi, xf)
            deriv = deriv - 3 * dy * lpf_deriv_n(1, xi, xf, r0, gamma) / d ** 2
        elif j == 2:  # dF1/dr0
            deriv = -3 * dy * lpf_deriv_n(2, xi, xf, r0, gamma) / d ** 2
        elif j == 3: #dF0/dgamma
            deriv = -3 * dy * lpf_deriv_n(3, xi, xf, r0, gamma) / d ** 2 
    elif i == 2:
        #if j == 0:
        #    deriv = lpf_deriv_n(0, xi, xf, r0)
        #elif j == 1:
        #    deriv = lpf_deriv_n(1, xi, xf, r0)
        #elif j == 2:
        #    deriv = lpf_deriv_n(2, xi, xf, r0)
        if j == 0:
            deriv = lpf_deriv_n(0, xi, xf, r0, gamma) / r0
        elif j == 1:
            deriv = lpf_deriv_n(1, xi, xf, r0, gamma) / r0
        elif j == 2:
            deriv = lpf_deriv_n(2, xi, xf, r0, gamma) / r0 - (n0 - ni) / r0 ** 2
        elif j == 3:
            deriv = lpf_deriv_n(3, xi, xf, r0, gamma) / r0
    elif i == 3:
        deriv = ni * lpf_deriv_n(j, xi, xf, r0, gamma) / n0 ** 2
    else:
        deriv = 0.
        

    return deriv


@njit
def lpf_deriv_n(i, xi, xf, r0, gamma):
    """
    Derivative of n wrt to fit parameters

    :param i:
    0=x
    1=y
    2=r0
    3=gamma
    :param xi: hit position
    :param xf: fit position
    :param r0: number of photons
    :param gamma:  constant background term
    :return: dn/di
    """

    if i < 2:
        deriv = -3 * lpf_nexp(xi, xf, r0, gamma) * (xf[i] - xi[i]) / lpf_dist(xi, xf) ** 2
    elif i == 2:
        deriv = lpf_nexp(xi, xf, r0, gamma) / r0
    elif i == 3:
        deriv = 1.
    else:
        deriv = 0.

    return deriv


@njit
def lpf_deriv_dist(i, xi, xf):
    """
    Derivative of distance wrt fit parameters

    :param i: 0=x
              1=y
              2=r0
    :param xi: hit position
    :param xf: fit position
    :return: dDist/di
    """

    if i < 2:
        deriv = (xf[i] - xi[i]) / lpf_dist(xi, xf)
    else:
        deriv = 0.0

    return deriv


@njit
def lpf_deriv_dist_min2(i, xi, xf):
    """
    Derivative of 1/dist**2

     :param i: 0=x
               1=y
               2=r0
    :param xi: hit position
    :param xf: fit position
    :return: d(1/Dist**2)/di
    """

    deriv = 0.0
    if i < 2:
        d = lpf_dist(xi, xf)
        deriv = -(2 / d ** 3) * lpf_deriv_dist(i, xi, xf)

    return deriv


# --------------------------------------------------------------------------------------- #
def lpf_event_display(xhit, nhit, fit_result, hit_is_used, xiter, **kwargs):
    """
    Event display

    :param xhit:
    :param nhit:
    :param fit_result:
    :param hit-is_used: 
    :param xiter:
    :param kwargs:
    :return:
    """
    
    xhit = np.array(xhit)
    nhit = np.array(nhit)

    plot_range = kwargs.pop('range', None)
    zoom = kwargs.pop('zoom',-1)
    nbins = kwargs.pop('nbins', 15)

    if plot_range == 'None':
        plot_range = ((0, 100), (0, 100))

    if zoom > 0:
        plot_range = ((fit_result[0]-zoom/2,fit_result[0]+zoom/2),(fit_result[1]-zoom/2,fit_result[1]+zoom/2))
        
    print("Reconstruction::lpf_event_display() ")
    fig = plt.figure(figsize=(16, 6))

    gs = GridSpec(2, 2, figure=fig)
    ax0 = plt.subplot(gs.new_subplotspec((0, 0), rowspan=2))
    ax1 = plt.subplot(gs.new_subplotspec((0, 1), rowspan=1))
    ax2 = plt.subplot(gs.new_subplotspec((1, 1), rowspan=1))

    # ax1 = plt.subplot(222)
    # ax2 = plt.subplot(224)

    #    fig, ax0 = plt.subplots(nrows=1)

    # make a list of the intermediate steps
    xp = []
    yp = []
    nn = []
    iiter = []
    for i in range(len(xiter)):
        if xiter[i][3] != 0:
            xp.append(xiter[i][0])
            yp.append(xiter[i][1])
            nn.append(xiter[i][2])
            iiter.append(i)
    xp = np.array(xp)
    yp = np.array(yp)
    nn = np.array(nn)
    iiter = np.array(iiter)

    niter = len(xp)
    ax1.plot(iiter, xp-fit_result[0])
    ax1.plot(iiter, yp-fit_result[1])
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('difference between position and fit result')
    ax2.plot(iiter, nn)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('number of photons')

    # make these smaller to increase the resolution
    dx, dy = (plot_range[0][1]-plot_range[0][0])/400,(plot_range[1][1]-plot_range[1][0])/400

    # generate 2 2d grids for the x & y bounds
    x = np.arange(plot_range[0][0], plot_range[0][1], dx)
    y = np.arange(plot_range[1][0], plot_range[1][1], dy)
    z = np.zeros((len(y), len(x)))

    #print(x,y)
    for i in range(len(x)):
        for j in range(len(y)):
            xx = x[i]
            yy = y[j]
            xff = np.array([xx,yy,0])
            #print('xfit =',xff,' r0 =',xiter[niter-1][2])
            z[j][i] = lpf_lnlike_plot(xhit, nhit, hit_is_used, xff, xiter[niter-1][2], xiter[niter-1][5])

    # z = z[:-1, :-1]
    levels = MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())

    # cmap = plt.get_cmap('afmhot')
    cmap = plt.get_cmap('PiYG')

    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # ax0 = fig.gca()

    cf = ax0.contourf(x + dx / 2., y + dy / 2., z, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    title_string = 'x = {:8.2f} y = {:8.2f} r0= {:8.2f}'.format(fit_result[0], fit_result[1], fit_result[2])
    ax0.set_title(title_string)

    # add the light detectors
    mx_eff = -1
    for ih in range(len(nhit)):
        if nhit[ih] > mx_eff:
            mx_eff = nhit[ih]

    for ih in range(len(nhit)):
        # draw location of SiPM
        xs = xhit[ih]

        # plot sensor only if in range
        if (xs[0] > plot_range[0][0]) & (xs[0] < plot_range[0][1]) & \
                (xs[1] > plot_range[1][0]) & (xs[1] < plot_range[1][1]):
            #dx = nhit[ih] / mx_eff *10 
            rr = (nhit[ih]+0.25) / mx_eff *10
            #sq = plt.Rectangle(xy=(xs[0] - dx / 2, xs[1] - dx / 2),
            #                   height=dx,
            #                   width=dx,
            #                   fill=False, color='red')
            if nhit[ih]>0:
                if hit_is_used[ih]:
                    color = 'red'
                else:
                    color = 'black'
                sq = plt.Circle(xy=(xs[0], xs[1]),
                                radius=rr,
                                fill=False, color=color)
            else:
                sq = plt.Circle(xy=(xs[0], xs[1]),
                                radius=rr,
                                fill=False, color='black')

            ax0.add_artist(sq)
            # write number of detected photons
            ### txs = str(nhit[ih])
            ### ax0.text(xs[0] + dx / 2 + 2.5, xs[1], txs, color='red')

    # initial position
    ax0.plot(xiter[0][0], xiter[0][1], 'o', markersize=10, color='cyan')
    ax0.plot(xp, yp, 'w-o', markersize=5)

    # true position
    # plt.plot(self.sim.get_x0()[0], self.sim.get_x0()[1], 'x', markersize=14, color='cyan')
    # reconstructed position
    # if abs(self.fdata['xr']) < 100:
    ax0.plot(fit_result[0], fit_result[1], 'wo', markersize=10)
    ax0.set_xlabel('x (cm)', fontsize=18)
    ax0.set_ylabel('y (cm)', fontsize=18)
    ax0.set_xlim([plot_range[0][0],plot_range[0][1]])
    ax0.set_ylim([plot_range[1][0],plot_range[1][1]])
    plt.show()

    istat = int(input("Type: 0 to continue, 1 to make pdf, 2 to quit...."))

    if istat == 1:
        fname = 'event.pdf'
        fig.savefig(fname)
        fname = 'event.png'
        fig.savefig(fname)

    clear_output()
    return istat


#------------------------------
def difference(pmt_pos,nhit,xml,yml,r0,gamma):
    """
    Give a plot of the expected number of hits and the detected number of hits
    Also shows the impact of the gamma factor on the expected number of hits
    """
    
    #different lists used
    difference_n = []
    x_lijst = []
    y_lijst = []
    n_exp_lijst = []
    n_exp_gamma_lijst = []
    x_nexp_lijst = []
    
    #loops through all the PMTs
    for iii in range(len(pmt_pos)):
        x_nexp_lijst.append(iii)
        
        # calculate expected number of hits
        n_exp = lpf_nexp(pmt_pos[iii],(xml,yml,0),r0,gamma)
        difference = (((n_exp-nhit[iii])))/(n_exp**0.5)
        difference_n.append(difference)
        n_exp_lijst.append(n_exp)
        n_exp_gamma_lijst.append(n_exp - gamma)
        x_lijst.append(pmt_pos[iii][0])
        y_lijst.append(pmt_pos[iii][1])

    # bins the detected hits into clear datapoints with error values
    nbins5 = 100
    n15, _____ = np.histogram(x_nexp_lijst, bins=nbins5)
    sy15, _____ = np.histogram(x_nexp_lijst, bins=nbins5, weights=(nhit))
    sy215, _____ = np.histogram(x_nexp_lijst, bins=nbins5, weights=(nhit)*(nhit))
    mean15 = sy15 / n15
    std15 = np.sqrt(sy215/n15 - mean15*mean15)
    
    # plots the expected number of photons with and without gamma
    plt.errorbar((_____[1:] + _____[:-1])/2, mean15, yerr=std15, fmt='ko',zorder=10)
    plt.plot(n_exp_lijst,label= "Fitted hits with gamma",linewidth = 2,color = 'red',zorder=5)
    plt.plot(n_exp_gamma_lijst, '--',label= "Fitted hits without gamma",linewidth = 2,color = 'red',zorder=0) 
    plt.xlabel("PMT number",fontsize=14)
    plt.ylabel("number of hits",fontsize=14)
    plt.legend(loc="upper right")
    plt.yscale('log')
    plt.xlim(0,250)
    plt.ylim(0,10000)
    plt.show()

