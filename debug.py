# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:38:38 2023

Copy paste code to debug

@author: George
"""

def mode(dataset):
    
    """ of a normal distribution """
    
    x = np.array(dataset,dtype = int)
    x = x.flatten()
    vals,counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    return vals[index]

# plt.close('all')

# sns.set_style("white")
# sns.set_style("ticks")

# plot_params = {
#     'title':'Mean data',
#     'axvline':80,
#     'axvline_label':None,
#     'legend':False,
#     'color':'red',
#     'xlabel':'Data Count',
#     'ylabel':'Erpobdella movement'}

fig,axe = plt.subplots(1,3,figsize = (20,7))
fig.suptitle('Mean Activity Distribution',fontsize = 18)
fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
for i,s in enumerate(specie):
    
    histdata = np.array(data_mean[s])
    histdata = histdata.flatten()
    
    sns.histplot(histdata,ax=axe[i],color = data.species_colors[s],kde = True)
    axe[i].set_title(data.species[s],fontsize = 16)
    if s == 'E': axe[i].set_xlim((-5,150))
    
    q75 = np.abs(np.quantile(histdata,0.75))
    axe[i].axvline(q75,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Upper quartile')
    
    if i == 2: axe[i].legend(fontsize = 17)
    
fig,axe = plt.subplots(1,3,figsize = (20,7))
fig.suptitle('High Quantile Activity Distribution',fontsize = 18)
fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
for i,s in enumerate(specie):
    
    histdata = np.array(data_qlow[s])
    histdata = histdata.flatten()
    
    sns.histplot(histdata,ax=axe[i],color = data.species_colors[s])
    axe[i].set_title(data.species[s],fontsize = 16)
    
    q90 = np.quantile(histdata,0.1)
    axe[i].axvline(q90,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Quantile -0.9')
    
    if i == 1: axe[i].legend(fontsize = 17)
    
fig,axe = plt.subplots(1,3,figsize = (20,7))
fig.suptitle('Low Quantile Activity Distribution',fontsize = 18)
fig.text(0.5, 0.04, 'Distance', ha='center',fontsize = 16)
for i,s in enumerate(specie):
    
    histdata = np.array(data_qhigh[s])
    histdata = histdata.flatten()
    
    sns.histplot(histdata,ax=axe[i],color = data.species_colors[s])
    axe[i].set_title(data.species[s],fontsize = 16)
    
    axe[i].set_ylim((0,3000))
    
    q95 = np.quantile(histdata,0.95)
    axe[i].axvline(q95,color = 'black',linestyle = '--',linewidth = 1.5,label = 'Quantile 0.95')
    
    if i == 2: axe[i].legend(fontsize = 17)
    
fig = plt.figure(figsize = (13,7))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('Scaled Low Quantile Activity Distributions',fontsize = 18)

for i,s in enumerate(specie):
    
    histdata = data_qlow[s].copy()
    histdata = histdata.flatten()/np.abs(np.quantile(data_qlow[s],0.1))
    sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])
    
axe.legend(fontsize = 18)
fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)


fig = plt.figure(figsize = (13,7))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('Scaled High Quantile Activity Distributions',fontsize = 18)

for i,s in enumerate(specie):
    
    histdata = data_qhigh[s].copy()
    histdata = histdata.flatten()/np.abs(np.quantile(data_qhigh[s],0.95))
    sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])

axe.legend(fontsize = 18)
axe.set_xlim(0,1.5)
axe.set_ylim(0,8000)
fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)
    
fig = plt.figure(figsize = (13,7))
axe = fig.add_axes([0.1,0.1,0.8,0.8])
axe.set_title('Scaled Mean Activity Distributions',fontsize = 18)

for i,s in enumerate(specie):
    
    histdata = data_mean[s].copy()
    histdata = histdata.flatten()/np.abs(np.mean(data_mean[s]))
    sns.histplot(histdata,ax=axe,color = data.species_colors[s],alpha = 1-0.3*i,label = data.species[s])
 
axe.legend(fontsize = 18)
axe.set_xlim((-0.25,4))
fig.text(0.5, 0.04, 'Scaled Distance', ha='center',fontsize = 16)