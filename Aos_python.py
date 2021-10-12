# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:53:04 2021

@author: Abednego Wamuhindo
"""
import pandas as pd
from datetime import datetime
from datetime import timedelta 
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy import fftpack

#load the dataset file
dataset_apps = pd.read_csv (r'appsv.csv', header=None, names=["Time", "Application"])
dataset_apps.rename(columns={"0": "Time", "1": "Application"})

#define the minutes of one time bucket
min_to_add = 60

#print (dataset_appss.loc[dataset_appss["Date"]=="22:46:43"])

#print (dataset_apps.loc[dataset_apps["Date"]dataset_apps["Date"][0][11:16]=="22:46"])
dataset_apps2 = dataset_apps.copy()
dataset_apps2['Day'] = dataset_apps["Time"]
dataset_apps2['Datetime'] = [datetime.strptime(date[0:19], '%Y-%m-%d %H:%M:%S') for date in dataset_apps2["Time"] ]
dataset_apps2['Exec_time'] = [0]*len(dataset_apps.index) 
row_value = ""
mday = 0


#arrange the time displayed and add a day column
for index, row in dataset_apps.iterrows():
    mdate = datetime.strptime(row["Time"][0:19], '%Y-%m-%d %H:%M:%S')
    dataset_apps2.at[index, 'Time']=  mdate.time()
    if index < len(dataset_apps.index)-1 and dataset_apps2.at[index+1, 'Datetime'].day == dataset_apps2.at[index, 'Datetime'].day: 
        secs = (dataset_apps2.at[index+1, 'Datetime']-dataset_apps2.at[index, 'Datetime']).total_seconds()
 
        if secs<=3600:
            dataset_apps2.at[index, 'Exec_time'] = secs
        else:
            #print(dataset_apps2.at[index, 'Application'])
            dt_tmp = dataset_apps2[dataset_apps2["Application"]==dataset_apps2.at[index, 'Application']] 
            dt_tmp = dt_tmp[dt_tmp["Exec_time"]!=0]      
            if len(dt_tmp.index)!= 0:               
                dataset_apps2.at[index, 'Exec_time'] = sum(dt_tmp["Exec_time"])//len(dt_tmp.index)
            else:
                dataset_apps2.at[index, 'Exec_time'] =dataset_apps2.at[index-1, 'Exec_time']           
    else:
        dataset_apps2.at[index, 'Exec_time'] = dataset_apps2.at[index-1, 'Exec_time']
    if mdate.day != row_value:
        mday = mday + 1
        dataset_apps2.at[index,'Day']= mday
        row_value = mdate.day
    else:
         dataset_apps2.at[index,'Day']= mday
         
#print( dataset_apps)
#print( dataset_apps2[dataset_apps2["Application"]=="com.android.chrome"])
#print(sum(dataset_apps2[dataset_apps2["Application"]=="com.android.chrome"]["Exec_time"])/len(dataset_apps2[dataset_apps2["Application"]=="com.android.chrome"].index))


initial_time = datetime(year=1,day=1,month=1,hour=0, minute=0, second=0)
initial_time2  = initial_time + timedelta(minutes=min_to_add)


#Bucket for query time hours 
hour_bucket_collection = {} 
# numer of bucket 
number_bucket = int((24*60)/min_to_add)
for h in range(number_bucket):
    hour_bucket_collection[h] = dataset_apps2[dataset_apps2["Time"].between(initial_time.time(),initial_time2.time())]
    initial_time = initial_time2
    initial_time2 = initial_time2 + timedelta(minutes=15)
    
#list of all the apps
applications = dataset_apps.Application.unique()   

#count the global usage of each app and compute the probability 
global_count = dataset_apps2["Application"].value_counts(normalize=True).tolist()
list_apps = dataset_apps2["Application"].value_counts().index.tolist()

#global usage trace
global_usage = pd.DataFrame(list_apps,columns=['Application'])
global_usage["prob"] = global_count

print(global_usage)


usage_counts ={}
for h in range(number_bucket):
    n = hour_bucket_collection[h].Day.nunique() 
    count = (hour_bucket_collection[h]["Application"].value_counts()/n).tolist()
    list_app = hour_bucket_collection[h]["Application"].value_counts().index.tolist()
    
    usage_count = pd.DataFrame(list_app,columns=['Application'])
    usage_count["count"] = count
    usage_counts[h] = usage_count
#hour_bucket_collection[0]["usage_count"]=usage_count

#the time of the request the choice should be between 0 and number bucket.
bucket_query=22
#print(usage_counts[41]["count"].div(len(usage_counts[41])))

#temporal usage of the apps
temporal = pd.DataFrame(usage_counts[bucket_query]["Application"],columns=["Application"])
temporal["prob"] = usage_counts[bucket_query]["count"].div(usage_counts[bucket_query]["count"].sum())

print(temporal)


initial_datetime = dataset_apps2.iloc[0]["Datetime"]
t = initial_datetime.time()
#print(t)
tot_sec = int(timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
#print(tot_sec)
initial_datetime  = initial_datetime - timedelta(seconds=tot_sec)
#print(initial_datetime)
initial_datetime2  = initial_datetime + timedelta(minutes=60)
#print(initial_datetime2)


#compute the frequencial usage

freq_bucket_collection = {}
number_row_freq = 24*dataset_apps2.iloc[len(dataset_apps2.index)-1]["Day"]

#print("Heure "+str(number_row_freq))
for hr in range(number_row_freq):
    freq_bucket_collection[hr] = dataset_apps2[dataset_apps2["Datetime"].between(initial_datetime,initial_datetime2)]
    initial_datetime = initial_datetime2
    initial_datetime2 = initial_datetime2 + timedelta(minutes=60)
    #print(initial_datetime)

#compute the usage count for the period
usage_counts_freq = {}
for hrr in range(number_row_freq):

    count2 = (freq_bucket_collection[hrr]["Application"].value_counts()).tolist()
    list_app = freq_bucket_collection[hrr]["Application"].value_counts().index.tolist()
    
    usage_count_freq = pd.DataFrame(list_app,columns=['Application'])
    usage_count_freq["count"] = count2
    #print(count2)
    usage_counts_freq[hrr] = usage_count_freq


#print(("0" if len(usage_counts_freq[23][usage_counts_freq[23]["Application"]==applications[7]]["count"])==0 else usage_counts_freq[23][usage_counts_freq[23]["Application"]==applications[7]]["count"].item()))
data_freq = {}

for bck in range(len(applications)):
    arr = []
    for hb in range(number_row_freq):
        mdataset_apps = usage_counts_freq[hb]
        if len(mdataset_apps[mdataset_apps["Application"]==applications[bck]]["count"]) == 0:
            arr.append(0)
        else:
            count_app_dataset_apps = mdataset_apps[mdataset_apps["Application"]==applications[bck]]["count"].item()
            arr.append(count_app_dataset_apps)
        #print(count_app_dataset_apps["count"])
        #print("\n")
        # if len(count_app_dataset_apps)==0:
        #     data = 0
        # else:
        #     data = count_app_dataset_apps["count"]
        
        
        #print(arr)
       
    data_freq[applications[bck]] = arr
    
    
#print(applications)

#Here you can print the apps with their frequential usage he number in the bracket is the hours
print(usage_counts_freq[35])
#print(data_freq["com.whatsapp"])
#dataset_apps_f = dataset_apps2.copy()[dataset_apps2["Application"]=="com.whatsapp"]

#dataset_apps_f["Hour"]=np.random.uniform(0,1, len(dataset_apps_f.index) )
'''
for index, row in usage_counts_freq[22].iterrows():
     print(str(index)+". ") 
     print(row)
'''
#print(dataset_apps2)

# Make plots appear inline, set custom plotting style
%matplotlib inline

#plt.style.use('style/elegant.mplstyle')
random.seed(1234)

y = data_freq["com.whatsapp"]
x = list(range(number_row_freq))

fig, ax = plt.subplots()
ax.set_title('Whatsapp usage')
ax.plot(x, y)
ax.set_xlabel('Time [hour]')
ax.set_ylabel('Usage Count');
plt.savefig('whatsapp_usage.pdf') 


#temporal usage 
print(usage_counts[bucket_query]["Application"])
yt = usage_counts[bucket_query]["count"]
#xt = ["Whatsapp","Chrome","hermione","Files","Twitter","Gmail","Settings","Messenger"]
xt = usage_counts[bucket_query]["Application"]
fig, ax = plt.subplots()
ax.set_title('Temporal usage')
ax.stem(xt, yt)
ax.set_xlabel('Applications')
ax.set_ylabel('Temporal Usage Count');
plt.xticks(rotation='vertical')
ax.set_ylim(0, 2.5)
plt.savefig('temporal_usage2.pdf') 

#plot the DFT after shuffling


y2 = y.copy()
random.shuffle(y2)
X = fftpack.fft(y2)
freqs = fftpack.fftfreq(len(y2))
#print(X)
fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency (times/hour)')
ax.set_ylabel('Power')
ax.set_xlim(0, 0.3)
ax.set_ylim(0, 600)


#plot the DFT without shuffling and the line for the max from the FFT with shuffling
#usage of the fourrier transform 

X2= fftpack.fft(y)
freqs = fftpack.fftfreq(len(y))


#Compute the candidate frequency
#ax.stem(freqs, [max(np.abs(X2))]*len(freqs))
max_shuffle = max([item for item in np.abs(X) if item not in [max(np.abs(X))]])
cand_freq = np.array_split(np.abs(X2), 2)[1]
cand_freq1 = cand_freq[cand_freq>=max_shuffle]
cand_freq1 = [item for item in cand_freq1 if item not in [max(np.abs(X))]]
freq_ind = []
for cand in cand_freq1:
    freq_ind.append(freqs[np.where(np.abs(X2) == cand)[0][0]])

#print(freq_ind) 
#print(cand_freq1) 
#print(cand_freq)
fig, ax = plt.subplots()
ax.stem(freqs, np.abs(X2))
ax.plot(freqs,[max_shuffle]*len(freqs),'--r',label=''+str(max_shuffle))
colors=['r','g','y','m','c']*10
for ind in range(len(cand_freq1)):
    ax.plot(freq_ind[ind],[cand_freq1[ind]],'o'+str(colors[ind]))
    ax.text(freq_ind[ind],cand_freq1[ind],'P'+str(ind),horizontalalignment='left',verticalalignment='bottom')
#ax.scatter(freq_ind[0][0],cand_freq1[0],'r')
ax.set_xlabel('Frequency (times/hour)')
ax.set_ylabel('Power')
ax.set_xlim(0, 0.3)

#the 1800 can be change to adjust the height of te figure depending on the amount of data in the dataset
ax.set_ylim(0, 600)
plt.savefig('whatsapp_periodogram.pdf') 

#candidate periods
period_cand = [1/item for item in freq_ind if item > 0.]
#print(period_cand)

def autocorrelation(x):
    xp = fftpack.ifftshift((x - np.average(x))/np.std(x))
    #xp = fftpack.ifftshift((x - np.average([0,0])))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fftpack.fft(xp)
    p = np.absolute(f)**2
    pi = fftpack.ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)


fig, ax = plt.subplots()
ax.plot(x[:number_row_freq//2], autocorrelation(y))
ax.set_xlim(0, 205)


#Prediction ok k tops apps;
k = 3
Hf_gu = [0]*k
Hf_tmp = [0]*k
Hf_per = [0]*k



def computeEntropy(usage_probs):
    entropy = 0
    for indice, prob in usage_probs.iterrows():
        entropy = entropy + prob["prob"]*math.log10(prob["prob"])
    return -entropy

global_usage_clone = global_usage.copy()
temporal_clone = temporal.copy()

#print(global_usage_clone[global_usage_clone["Application"]=="com.whatsapp"]["prob"].item())
#print(temporal_clone[temporal_clone["Application"]=="com.android.chrome"]["prob"].item())
predict_app = ['']*k
prob_app =[0]*k
feature = 3
for ki in range(k):
    if ki ==0 :
        Hf_gu[ki] = computeEntropy(global_usage_clone)
        Hf_tmp[ki] = computeEntropy(temporal_clone)
    else:
        if not global_usage_clone[global_usage_clone["Application"]==predict_app[ki-1]].empty:
            if int(prob_app[ki-1])==1:
                Hf_gu = 1000
            else:
                Hf_gu[ki] = (Hf_gu[ki-1]+prob_app[ki-1]*math.log10(prob_app[ki-1]))/(1-prob_app[ki-1])+math.log10(1-prob_app[ki-1])
        if not temporal_clone[temporal_clone["Application"]==predict_app[ki-1]].empty:
            if int(prob_app[ki-1])==1:
                Hf_tmp = 1000
            else:
                Hf_tmp[ki] = (Hf_tmp[ki-1]+prob_app[ki-1]*math.log10(prob_app[ki-1]))/(1-prob_app[ki-1])+math.log10(1-prob_app[ki-1])

    arr_feature_ent = np.array([Hf_gu[ki],Hf_tmp[ki]] )
    feature = np.argmin(arr_feature_ent)
    print(feature)
    if feature==0:
        predict_app[ki] = global_usage_clone[global_usage_clone["prob"]==max(global_usage_clone["prob"])]["Application"].item()
        prob_app[ki] = global_usage_clone[global_usage_clone["Application"]==predict_app[ki]]["prob"].item()
        
        
    elif feature == 1:
        if not temporal_clone[temporal_clone["prob"]==max(temporal_clone["prob"])].empty:
            predict_app[ki] = temporal_clone[temporal_clone["prob"]==max(temporal_clone["prob"])]["Application"].item()
            prob_app[ki] = temporal_clone[temporal_clone["Application"]==predict_app[ki]]["prob"].item()
        else:
            predict_app[ki] = ''
            prob_app[ki] = 0
        
    else:
        predict_app[ki] = temporal_clone[temporal_clone["prob"]==max(temporal_clone["prob"])]["Application"].item()
        
    global_usage_clone = global_usage_clone[global_usage_clone["Application"]!=predict_app[ki] ]
    temporal_clone = temporal_clone[temporal_clone["Application"]!=predict_app[ki]] 

print(Hf_gu)
print(Hf_tmp)
print(predict_app)

#print(temporal)
#print(hour_bucket_collection[bucket_query]["Day"])

#computation of the total execution time
def getExecutionTime(bucket_query_time,app):
    bucket = hour_bucket_collection[bucket_query_time]
    exec_instances = bucket[bucket["Application"]==app];
    days = exec_instances.Day.nunique()
    indexes = exec_instances.index
    tot_exec_time = 0
    for index in indexes:
        tot_exec_time = tot_exec_time + exec_instances.at[index,"Exec_time"]
    if days == 0:
        tot_exec_time = sum(dataset_apps2[dataset_apps2["Application"]==app]["Exec_time"])//len(dataset_apps2[dataset_apps2["Application"]==app]["Exec_time"].index)
    else:
        tot_exec_time = tot_exec_time //days
    print(tot_exec_time)
    return tot_exec_time

getExecutionTime(bucket_query, "com.whatsapp")    
#print(dataset_apps2.at[3163,"Datetime"])
#print(dataset_apps2.at[3164,"Datetime"])

'''
bucket = hour_bucket_collection[bucket_query]
exec_instances = bucket[bucket["Application"]=="com.android.chrome"];
print(exec_instances)
print(exec_instances.Day.nunique())
print(bucket.at[1910,"Exec_time"])
'''
#print(predict_app)

#Here we want to export the json file
'''
#As we did the experience for 3 apps, "Whatsapp", 
"youtube" and "chrome" we will set that mannually
 as predicted app but in the real app that should
 come from the above predicted app
 We set also the execution time manually to see 
 resonnable effects for the simulation
'''
import json
predict_app =['com.whatsapp','com.google.android.youtube','com.android.chrome']
data = {}
data['device'] = {}
data['apps'] = []
energy ={}
energy1 ={}
energy2 ={}

# these frequencies are the ones we computed by extrapolation
data['device']["frequency"] = {}
data['device']["frequency"]["f1"]=960
data['device']["frequency"]["f2"]=1094
data['device']["frequency"]["f3"]=1228
data['device']["frequency"]["f4"]=1362
data['device']["frequency"]["f5"]=1400
data['device']["battery_level"]=75
data['device']["capacity_level"]=2920
data['device']["number_cores"]= 8

'''
# these values of energies come also by extrapolation
these are for whatsapp application
'''
energy1["f1"]= 739.4
energy1["f2"]= 776.3
energy1["f3"]= 782.4
energy1["f4"]= 815.9
energy1["f5"]= 824.4
data['apps'].append({
    'name':predict_app[0],
    'exec_time':'1230',
    })
data['apps'][0]["energy"] = energy1

'''
# these values of energies come also by extrapolation
these are for youtube application
'''
energy["f1"]= 391.7
energy["f2"]= 457.4
energy["f3"]= 523.0
energy["f4"]= 588.7
energy["f5"]= 607.3
data['apps'].append({
    'name':predict_app[1],
    'exec_time':'1320',
    })
data['apps'][1]["energy"] = energy

'''
# these values of energies come also by extrapolation
these are for chrome application
'''
energy2["f1"]= 429.8
energy2["f2"]= 506.2
energy2["f3"]= 581.8
energy2["f4"]= 657.7
energy2["f5"]= 679.3
data['apps'].append({
    'name':predict_app[2],
    'exec_time':'604',
    })
data['apps'][2]["energy"] = energy2   

with open('profile.json', 'w') as outfile:
    json.dump(data, outfile)

#computation of edps
edps = {}
for app in predict_app:
    edps[app] = []
    edp = []
    exec_time = int((data['apps'][predict_app.index(app)]['exec_time']))
    for energy in data['apps'][predict_app.index(app)]['energy']:
        edp.append(((data['apps'][predict_app.index(app)]['energy'][energy])*exec_time)/1000)
    edps[predict_app.index(app)] = edp

#print(data['apps'][predict_app.index("com.whatsapp")]['energy'])    
for app in predict_app:
    print(edps[predict_app.index(app)])

alpha = 0.5
bat = data['device']["battery_level"]/100
Bind = math.sqrt((1-bat))
ecis = {}

for app in predict_app:
    ecis[app] = []
    eci = []

    for edp in edps[predict_app.index(app)]:
        eci.append(alpha * Bind + (1-alpha)*edp)
    ecis[app] = eci

for app in predict_app:
    print(ecis[app])
#print(sum(dataset_apps2[dataset_apps2["Application"]=="com.whatsapp.w4b"]["Exec_time"])//len(dataset_apps2[dataset_apps2["Application"]=="com.whatsapp.w4b"]["Exec_time"].index))

'''
Here come the time to determine the CPU running
frequencies
the available budget is set 
We set also the execs as the execution time of the 3 apps
in that way we can change the available energy and the
execution time of the apps

if the frequency index is -1 the scheduling is not possible
otherwise, the frequency can be found from the index
knowing that we have:
    0:960 MHz 1:1094MHz 2:1228MHz 3:1362MHz 4:1400MHz
'''

available_budget= 2500;
execs = [1800,1320,900]
k = 3
for app in predict_app:
    edps[app] = []
    edp = []
    exec_time = execs[predict_app.index(app)]
    for energy in data['apps'][predict_app.index(app)]['energy']:
        edp.append(((data['apps'][predict_app.index(app)]['energy'][energy])*exec_time)/1000)
    edps[predict_app.index(app)] = edp


freq_ind = [-1]*k

def findFrequencyIndexes():
    for i in range(len(edps[k-1])-1,-1,-1):
        for j in range (len(edps[0])-1,-1,-1):
            for l in range(len(edps[0])-1,-1,-1):    
                if (available_budget - (edps[1][j] + edps[k-1][l])) >= edps[0][i]:
                    freq_ind[k-1]= l
                    freq_ind[1] = j
                    freq_ind[0] = i
                    return;
            
findFrequencyIndexes()   
print(freq_ind)
