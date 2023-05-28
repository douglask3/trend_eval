import os
import numpy  as np
import pandas as pd
import csv
import math

import pymc  as pm

from pdb import set_trace as browser

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter

n_itertations = 10000
#n_itertations = 250000
############################################################
## Fuctions. I'll do these as class objects at some point ##
############################################################
def run_regression(xs, ys):
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        epsilon = pm.LogNormal("epsilon", 0, 10)
        y0 = pm.Normal("y0", np.mean(ys), sigma=np.std(ys))
        beta = pm.Normal("beta", 0, (np.max(ys) - np.min(ys))/(np.max(xs) - np.min(xs)))
        
        prediction = y0 + beta * xs
        
        # Define likelihood
        likelihood = pm.Normal("mod", mu=prediction, sigma=epsilon, observed=ys)
    
        # Inference!
        # draw 1000 posterior samples using NUTS sampling
        trace = pm.sample(n_itertations, return_inferencedata=True)

    return(trace)


def conpare_gradients(beta1, beta2, ls = 'solid', color = 'red'):
    min_beta = np.min(np.append(beta1, beta2))
    max_beta = np.max(np.append(beta1, beta2))
    nbins = int(np.ceil(np.sqrt(beta2.size))) 

    bins = np.linspace(min_beta, max_beta, nbins)

    def normHist(beta) :
        out = np.histogram(beta, bins)[0]
        return out / np.max(out)
    
    dist1 = normHist(beta1)
    dist2 = normHist(beta2)
    
    pval = 0.1
    pltRnge =  np.array([0, 0])
    while pltRnge[1]-pltRnge[0] < 20:
        pval = pval * 0.5
        pltRnge = np.where(np.max(np.array([dist1, dist2]), axis = 0) > pval)[0][[0, -1]]
        if pval < 0.00001: pltRnge = np.array([0, len(bins)])
    
    def pltHist(dist, color):
        plt.gca().plot(bins[1:][pltRnge[0]:pltRnge[1]], dist[pltRnge[0]:pltRnge[1]], color = color)
    if ls == 'solid' and color == 'red': pltHist(dist1, 'blue')
    pltHist(dist2, color)  

    def compareFUN(d1, d2): return np.sqrt(np.sum(d1*d2))/np.sqrt(np.sum(d2))

    def firectionFUN(b1, b2):
        return np.mean(b1>0.0) * (np.mean(b2>0.0))  + np.mean(b1<0.0) * np.mean(b2 <0.0)  
 
    dist3 = np.min(np.array([dist1, dist2]), axis = 0)
    
    prob =  np.sqrt(np.sum(dist2*dist3)/np.sum(dist2*dist2))

    direction_sim = firectionFUN(beta1, beta2)
    direction_obs = firectionFUN(beta1, beta1)
     
    return prob#, direction_sim, direction_obs


##############################################
## The actual example you'll need to apdate ##
##############################################

def for_timeSeries(cname, years, dat, color, lineS, regionID, variable):
    
    burnt_area = dat[cname].values
    x = years[~np.isnan(burnt_area)]
    y = burnt_area[~np.isnan(burnt_area)] #+ (x - 1998)*10
    x = x
    #linestyle = 'dotted'
    plt.gca().plot(x, y/1000.0, 'x', color = color, linestyle = lineS, label=cname, markersize=0.67, linewidth = 0.25)
    
    tfile = 'temp/compare_trends-betas-' + cname + '-' + str(len(y)) + \
            '-' + regionID + '-' + '-' + variable + '-' + str(n_itertations) + '.csv'
    
    if os.path.isfile(tfile):
        dat = pd.read_csv(tfile).values
        beta = dat[0,]
        y0 = dat[1,]
    else:
        print(tfile)
        if len(y) == 1: return None
        try:
            traces = run_regression(x - np.min(x), np.log(y + 0.000000001)).posterior#["beta"].values
        except:
            browser()
        beta = traces['beta'].values.flatten()
        y0 = traces['y0'].values.flatten() 
        
        pd.DataFrame([beta, y0]).to_csv(tfile, index = False)
    
    x_sample = np.linspace(start=np.min(x), stop=np.max(x), num=100)

    prediction = np.array([np.quantile(np.exp(y0 + beta * x)/1000.0, [0.05, 0.5, 0.95]) for x in x_sample  - np.min(x)])
    
    polyX = np.append(x_sample, np.flip(x_sample))
    polyY = np.append(prediction[:,0], np.flip(prediction[:,2]))
    xy = np.transpose(np.array([polyX, polyY]))
    polygon = Polygon(xy, closed=True)
    collection = PatchCollection([polygon], alpha=0.1, facecolor = color)
    plt.gca().add_collection(collection)

    return(beta, len(y))



def forRegion(time_series_dir, file, variable, varname, units, fig):   
    pltN = int(float(file[5:7]))
    if pltN == 0: pltN = 15
    print(pltN)
    ax = fig.add_subplot(8, 4, 1+(pltN-1)*2)
    plt.title(file[0:4], loc = 'left')
    if file[0:4] == 'MIDE' or file[0:4] == 'NHAF': plt.ylabel(varname + " (" + units + ")")
    
    dat = pd.read_csv(time_series_dir + file)
    years = dat['years'].values
    
    nObs = np.sum([name[0:3] == "Obs" for name in dat.columns])
    nSim = np.sum([name[0:3] == "Sim" for name in dat.columns])


    colors = ['blue'] * nObs + ['red'] * nSim
    lineSs = ['solid'] * max(0, nObs -4) + ['solid', 'dotted', 'dashed', 'dashdot'][0:min(nObs, 4)] + \
             ['solid', 'dotted', 'dashed', 'dashdot', 'solid'][0:min(nSim, 5)]
   
    betas = [for_timeSeries(col, years, dat, color, lineS, file[0:4], variable)
              for col, color, lineS in zip(dat.columns[2:], colors, lineSs)]

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    #ax.yaxis.set_minor_formatter(ScalarFormatter())

    Obs = [beta for beta, name in zip(betas, dat.columns[2:]) if name.startswith('Obs')]
    Obs = [x for x in Obs if x is not None] 
    try:
        n_samples = np.array([i[1] for i in Obs])
        n_samples = np.round(4*n_itertations* n_samples/ np.sum(n_samples))
    except:
        browser()
    def selectN(samples, nsamples):
        sampl = samples[0].flatten()
        sampl = sampl[np.random.choice(len(sampl), size=int(nsamples), replace=True)]
        return(sampl)

    
    ObsS = np.array([])
    for O, ns in zip(Obs, n_samples): ObsS = np.append(ObsS, selectN(O, ns))

    Sims = [beta[0] for beta, name in zip(betas, dat.columns[2:]) if name.startswith('Sim')]
    
    ax = fig.add_subplot(8, 4, pltN*2, xlabel="log(" + varname + " (" + units + " $year^-1$)")
    if file[0:4] == 'MIDE' or file[0:4] == 'NHAF': plt.ylabel("Probablity")
                         
    scores = np.array(conpare_gradients(ObsS, np.zeros(n_itertations), color = 'black'))
    grad_range = np.percentile(ObsS, [5, 95])
    for Sim, ls in zip(Sims, ['solid', 'dotted', 'dashed', 'dashdot', 'solid']):
        scores = np.append(scores, conpare_gradients(ObsS, Sim, ls)) 
        grad_range = np.append(grad_range, np.percentile(Sim, [5, 95]))
    
    return scores, grad_range


def run_for_variable(variable, time_series_dir, varname, units): 

    files = os.listdir(time_series_dir)
    plotN = np.array([float(file[5:7]) for file in files])
    plotN[plotN == 0.0] += np.max(plotN)+1.0
    sort_index = np.argsort(plotN)
    files = np.array(files)[sort_index]

    fig = plt.figure(figsize=(12, 15))
    fig.tight_layout(pad=5.0)
    plt.rcParams['axes.titley'] = 0.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = 10
    outs = [forRegion(time_series_dir, file, variable, varname, units, fig) for file in files]
    
    probs = pd.DataFrame([i[0] for i in outs])
    grads = pd.DataFrame([i[1] for i in outs])

    pd.DataFrame(probs).to_csv('outputs/' + variable + 'probs_isimip2b_regional_trend_eval.csv')
    pd.DataFrame(grads).to_csv('outputs/' + variable + 'grads_isimip2b_regional_trend_eval.csv')

    plt.savefig("figs/compare_" + variable + "_trends.png", dpi = 300)

run_for_variable("burnt_area", 'outputs/burnt_area_obs_isimip2b/', "Burnt Area", "$1000 km^2$")

run_for_variable("tree_cover_fireOff", 'outputs/trees_obs_isimip2b_fireOff/', "Tree Cover", "$1000 km^2$")
run_for_variable("tree_cover_fireOn", 'outputs/trees_obs_isimip2b_fireOn/', "Tree Cover", "$1000 km^2$")

run_for_variable("tallTree_cover_fireOff", 'outputs/tallTrees_obs_isimip2b_fireOff/', "Tall Tree Cover", "$1000 km^2$")
run_for_variable("tallTree_cover_fireOn", 'outputs/tallTrees_obs_isimip2b_fireOn/', "Tall Tree Cover", "$1000 km^2$")

run_for_variable("fire_emissions", "outputs/fire_emissions_veg_obs_isimip2b/", "Fire Emission", "some units")

browser()


