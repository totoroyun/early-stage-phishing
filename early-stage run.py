import time
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import _pickle as pickle
import datetime
from scipy.io import savemat
import scipy.io as sio
import json
from dateutil.parser import parse
import time
from sklearn import preprocessing 
from functions import *
import re
import math

#------------------------------------load data----------------------------------------------
f = open('./MulDiGraph.pkl','rb') 
G = pickle.load(f)  
len(G)

#----------------------------------extract features----------------------------------------------
#get timestamp of each given date       
dt = "2018-1-30 23:59:59"
timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
di = time.mktime(timeArray)

#add 15 days (time window is 15 days)
dts = [di]
for interval in range(1,12):
    dts.append(di + interval*15*24*3600)

#1. extract node features
for num in range(0,len(dts)):
    print(num)
    di = dts[num]
    date = time.strftime("%Y-%m-%d",time.localtime(di))         
    print(date)
    j = num+1

    #G1 is the graph ends before di
    pos_node = [node for node in pos_time_amount if G.in_degree(node)>=5]
    G1,P_nodes,N_nodes,train_pos,test_pos = get_pos_info(pos_time_amount,pos_node,di,G)
    test_amount = {node:get_in_amount(G1,node) for node in test_pos}
    stages = get_three_stages(test_amount, pos_time_amount)
    stages_num = np.array([date,len(train_pos),len(stages[0]),len(stages[1]),len(stages[2])])
   
    #save G1 and pos information
    with open("./T/T"+str(j)+"/G1.pkl", 'wb') as f:     # 将数据写入pkl文件
        pickle.dump(G1, f)
    #save pos info
    pos_info = {'P':P_nodes,'N':N_nodes,'train':train_pos,'test':test_pos,
                'early_pos':stages[0],'middle_pos':stages[1],'late_pos':stages[2]}
    savemat("./T/T"+str(j)+"/pos_info.mat", pos_info)
    
    #extract features
    fel_time = save_efl_fs(j,G1)



#----------------------------------test performance----------------------------------------------
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(penalty='l2',C = 0.5,solver='liblinear')
from sklearn.naive_bayes import BernoulliNB
clf2 = BernoulliNB()
from sklearn import svm
clf3= svm.SVC()
from sklearn import neighbors
clf4 = neighbors.KNeighborsClassifier(10)
from sklearn.ensemble import RandomForestClassifier
clf5 = RandomForestClassifier(n_estimators=50, random_state=0,max_depth=10)


iid = 'Remove corrected'
clf = clf1
for j in range(1,13):
#    print(j)
    clf_train = []
    clf_val = []
    clf_test = []
    clf_early = []
    clf_middle = []
    clf_late = []

    # load positive node in training, validation and test set
    pos_node = sio.loadmat("./T/T"+str(j)+"/Node/pos_node.mat")
    train_pos = pos_node['train']
    val_pos = pos_node['val']
    test_pos = pos_node['test']
    with open('./T/T'+str(j)+'/FEL/pos_fs.json','r',encoding = 'utf8') as f:        
        line = f.readline()
        pos_fs = json.loads(line)
    
    #load early, middle and late positive nodes in test set
    pos_info = sio.loadmat("./T/T"+str(j)+"/pos_info.mat")
    early_pos = pos_info['early_pos'][0]
    middle_pos = pos_info['middle_pos'][0]
    late_pos = pos_info['late_pos'][0]
    
    # combine negative nodes index
    early = np.concatenate((early_pos, early_pos + len(test_pos)))
    middle = np.concatenate((middle_pos, middle_pos + len(test_pos)))
    late = np.concatenate((late_pos, late_pos + len(test_pos)))
    
  
     
    i=0
    while i<10:
#        print(i)
        neg_node = sio.loadmat('./T/T'+str(j)+'/Node/neg_node'+str(i)+'.mat')
        train_neg = neg_node['train']
        val_neg = neg_node['val']
        test_neg = neg_node['test']
        with open('./T/T'+str(j)+'/FEL/neg_fs'+str(i)+'.json','r',encoding = 'utf8') as f:        
            line = f.readline()
            neg_fs = json.loads(line)
        
        #training set
        x_train_pos = [pos_fs[node] for node in train_pos]
        x_train_neg = [neg_fs[node] for node in train_neg]
        x_train = np.concatenate((x_train_pos,x_train_neg))
        y_train = np.array([1]*len(train_pos)+[0]*len(train_pos))
        
        #validation set
        x_val_pos = [pos_fs[node] for node in val_pos]
        x_val_neg = [neg_fs[node] for node in val_neg]
        x_val = np.concatenate((x_val_pos,x_val_neg))
        y_val = np.array([1]*len(val_pos)+[0]*len(val_pos))
               
        #test set
        x_test_pos = [pos_fs[node] for node in test_pos]
        x_test_neg = [neg_fs[node] for node in test_neg]
        x_test = np.concatenate((x_test_pos,x_test_neg))
        y_test = np.array([1]*len(test_pos)+[0]*len(test_pos))
         
        
        
        df_train = pd.DataFrame(x_train).fillna(0)
        df_val = pd.DataFrame(x_val).fillna(0)
        df_test = pd.DataFrame(x_test).fillna(0)
        
        #remove 
        df_train = input_fs(df_train,iid)     
        df_val = input_fs(df_val,iid)       
        df_test = input_fs(df_test,iid)      

        X_train = preprocessing.scale(df_train)
        X_val = preprocessing.scale(df_val)
        X_test = preprocessing.scale(df_test) 
    
        #train and predict
        scores1,scores2,scores3, scores4,scores5,scores6 = get_class_result(X_train, X_val, X_test, y_train, y_val, y_test,early, middle, late, clf)
        clf_train.append(scores1)
        clf_val.append(scores2)
        clf_test.append(scores3)
        clf_early.append(scores4)
        clf_middle.append(scores5)
        clf_late.append(scores6)      

        i+=1
        
    z1 = np.mean(clf_train,axis=0)
    z2 = np.mean(clf_val,axis=0)
    z3 = np.mean(clf_test,axis=0) 
    z4 = np.mean(clf_early,axis=0)
    z5 = np.mean(clf_middle,axis=0)
    z6 = np.mean(clf_late,axis=0)    
    
    df = pd.DataFrame([z1,z2,z3,z4,z5,z6],index = ['FELT_train','val','test','early','middle','late'],
                      columns = ['roc_auc','accuracy','precision','recall','F1'])
    print(df)


















