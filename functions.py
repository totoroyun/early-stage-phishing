import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
import random
import itertools
from sklearn import preprocessing 
import math  
from collections import defaultdict
import _pickle as pickle
from scipy.io import savemat
import scipy.io as sio
import json
from sklearn import metrics
from sklearn.model_selection import train_test_split

def get_pos_info(pos_time_amount,pos_node,di_timestamp,G):
    pos = pos_node
    node_time = pos_time_amount

    train_pos=[]   
    test_pos=[] 
    for node in pos:
        if node_time[node][1]<di_timestamp:      
            train_pos.append(node)
        elif node_time[node][0]<=di_timestamp:  
            test_pos.append(node)
        
    #G1 is the graph ends before di_timestamp
    G1 = nx.MultiDiGraph()
    for u,v,d in G.edges(data=True):  
        if d['timestamp']<=di_timestamp:
            G1.add_edge(u,v,amount=d['amount'],timestamp=d['timestamp'])  
    
    P_nodes = np.array(train_pos + test_pos)
    sample_N_nodes = [node for node in G1 if G1.in_degree[node]>=5]
    all_pos = list(node_time.keys())
    N_nodes  = np.array(list(set(sample_N_nodes).difference(set(all_pos))) )
            
    return G1,P_nodes,N_nodes,np.array(train_pos), np.array(test_pos)


def get_in_amount(G,node):    
    Am = []
    alls = list(nx.all_neighbors(G,node)) 
    downs = list(nx.neighbors(G,node))
    for i in downs:
        alls.remove(i)
    ups = alls

    if ups != []:
        for nd in ups:
            for i in range(len(G[nd][node])):
                Am.append(G[nd][node][i]['amount'])

    if ups != []:
        amount = sum(Am)
    else:
        amount = 0
    return amount


def get_three_stages(test_amount, pos_time_amount):
    e = []
    m = []
    l = []

    i = 0
    for node in test_amount:
        if pos_time_amount[node][2] == 0:
            value = 0
        else:
            value = test_amount[node]/pos_time_amount[node][2]
        if value <= 0.33:
            e.append(i)
        elif value <= 0.66:
            m.append(i)
        else:
            l.append(i)
        i+=1
            
    return [e,m,l]



def save_efl_fs(j,G1):
    #pos_fs
    pos_node = sio.loadmat("./T/T"+str(j)+"/Node/pos_node.mat")
    pos = np.concatenate((pos_node['train'],pos_node['val'],pos_node['test']))
    
    start_time = datetime.datetime.now()
    pos_fs={}
    for node in pos:
        Am,Ts,ups = get_in_ts(G1,node)    
        in_Am = get_ts_f(Am)
        in_Ts = get_ts_f(Ts)
    
        Am,Ts,downs = get_out_ts(G1,node)    
        out_Am = get_ts_f(Am)
        out_Ts = get_ts_f(Ts)
        
        Nt_fs = get_nt_f(G1,node,ups)    
        pos_fs[node]  =  in_Am + in_Ts + out_Am + out_Ts + Nt_fs
        
        
    file = open('./T/T'+str(j)+'/FEL/pos_fs.json', 'w')
    jsonstr = json.dumps(pos_fs)
    file.write(jsonstr)
    file.close()  

    end_time = datetime.datetime.now()
    delta = (end_time-start_time).seconds
    pos_time = delta


    
    #neg_fs
    neg_time = []
    i=0
    while i <10:
        print(i)
        start_time = datetime.datetime.now()

        neg_node = sio.loadmat('./T/T'+str(j)+'/Node/neg_node'+str(i)+'.mat')
        neg = np.concatenate((neg_node['train'],neg_node['val'],neg_node['test']))
     
        neg_fs = {}
        for node in neg:
            Am,Ts,ups = get_in_ts(G1,node)    
            in_Am = get_ts_f(Am)
            in_Ts = get_ts_f(Ts)
        
            Am,Ts,downs = get_out_ts(G1,node)    
            out_Am = get_ts_f(Am)
            out_Ts = get_ts_f(Ts)
            
            Nt_fs = get_nt_f(G1,node,ups)    
            neg_fs[node] = in_Am + in_Ts + out_Am + out_Ts + Nt_fs
            #每个特征的第14表示节点的入度交易中最大的时间戳；第34表示出度交易中最大的时间戳
            
        #负特征写入
        file = open('./T/T'+str(j)+'/FEL/neg_fs'+str(i)+'.json', 'w')
        jsonstr = json.dumps(neg_fs)
        file.write(jsonstr)
        file.close()  

        end_time = datetime.datetime.now()
        delta = (end_time-start_time).seconds
        neg_time.append(delta)
     
        i+=1
    
    fel_time = pos_time + np.mean(neg_time)
    
    return fel_time
        
        
        
        
from scipy.stats import entropy    
def get_in_ts(G,node):
    Am=[]
    Ts=[]
    ups=[]
    for nd in nx.all_neighbors(G, node):
        if G.has_edge(nd,node)==True:   
            ups.append(nd)  
            for i in range(len(G[nd][node])):
                Am.append(G[nd][node][i]['amount'])
                Ts.append(G[nd][node][i]['timestamp'])

    return Am,Ts,ups


def get_out_ts(G,node):
    Am=[]
    Ts=[]
    downs = list(nx.neighbors(G,node))
    if downs != []:
        for nd in downs:
            for i in range(len(G[node][nd])):
                Am.append(G[node][nd][i]['amount'])
                Ts.append(G[node][nd][i]['timestamp'])
   
    return Am,Ts,downs

#features of time series
def get_ts_f(Ts):
    ts=list(np.diff(sorted(Ts))/3600)
    if ts==[]:
        x1=x2=x3=x4=x5=x6=x7=x8=x9=x10=0
    else:
        x1=len(ts) 
        x2=sum(ts) 
        x3=np.mean(ts)
        x4=max(ts)
        x5=min(ts)
        x6=np.median(ts)
        x7=np.std(ts)
        x8=pd.Series(ts).skew()
        x9=pd.Series(ts).kurt()
        if len(ts)<5:
            x10 = 0
        else:
            x10=entropy(pd.cut(ts,5,labels=range(5)))   
    return [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]


        
#features of local network
def get_nt_f(G,node,ups):
    x1=G.in_degree(node)         
    x2=G.out_degree(node)                            
    x3=G.degree(node)    
 
    downs=[n for n in G.neighbors(node)]
    alls=[n for n in nx.all_neighbors(G, node)]
    x4=len(ups) 
    x5=len(downs)
    x6=len(alls)
    
    x7=G.in_degree(node,weight='amount')       
    x8=G.out_degree(node,weight='amount')
    x9=G.degree(node,weight='amount') 
    
    if x4==0:
        x10=1
    else:
        x10=x1/(x4)
        
    if x5==0:
        x11=1
    else:
        x11= x2/x5 
        
    if x6==0:
        x12=1
    else:
        x12=x3/(x6)
        
    
    x13=x5+x4-x6 

    N2=list(itertools.combinations(alls, 2))
    x14=0
    for i,j in N2:
        if G.has_edge(i,j) or G.has_edge(j,i):
            x14+=1
            
    nt_f=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]
    return  nt_f




def input_fs(df_train,iid):

    if iid=='Ts':
        df_train=df_train[list(range(0,40))] 
        df_train=df_train.drop([10,30],axis=1)               

    if iid=='Ln':
        df_train=df_train[list(range(40,54))]    
        
    if iid=='All':
        #10-40-46:len(in-time),in_degree, in -strength
        #30-41-47:out
        df_train=df_train.drop([10,40,46,30,41,47],axis=1)
        
    if iid == 'Remove corrected':
        #selected features
        df_train=df_train.drop([10, 40, 3, 16, 30, 41, 23, 35, 48],axis=1)
    return df_train
  


def get_class_result(X_train, X_val, X_test, y_train, y_val, y_test,early, middle, late, clf):
    clf.fit(X_train,y_train)
    y_train_pred=clf.predict(X_train)
    scores1 = get_scores(y_train, y_train_pred)

    y_val_pred = clf.predict(X_val)
    scores2 = get_scores(y_val,y_val_pred)

    y_test_pred = clf.predict(X_test)
    scores3 = get_scores(y_test, y_test_pred)
    
    X_test_early = X_test[early]
    X_test_middle = X_test[middle]
    X_test_late = X_test[late]
    y_test_early = y_test[early]
    y_test_middle = y_test[middle]
    y_test_late = y_test[late]
    
    y_test_early_pred = clf.predict(X_test_early)
    y_test_middle_pred = clf.predict(X_test_middle)
    y_test_late_pred = clf.predict(X_test_late)
    
    scores4 = get_scores(y_test_early, y_test_early_pred)
    scores5 = get_scores(y_test_middle, y_test_middle_pred)
    scores6 = get_scores(y_test_late, y_test_late_pred)
    

    return scores1,scores2,scores3, scores4,scores5,scores6

def get_scores(actual,pred):
    scores=[]
    scores.append(metrics.roc_auc_score(actual,pred))
    scores.append(metrics.accuracy_score(actual,pred))
    scores.append(metrics.precision_score(actual,pred))
    scores.append(metrics.recall_score(actual,pred))
    scores.append(metrics.f1_score(actual,pred))
    return scores


