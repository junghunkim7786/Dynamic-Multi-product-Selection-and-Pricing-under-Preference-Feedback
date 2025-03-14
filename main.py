from Environment import *
from Algorithms import *
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import multiprocessing



def run(T,repeat,d,N,K,i):
    oracle_reward=0
    std=dict()
    index=dict()
    p=np.zeros(N)
    Env_dict=dict()
    S=dict()
    exp_reward=dict()
    oracle_reward=dict()
    avg_regret_sum=dict()
    regret_sum_list=dict()
    std_R=dict()
    print('repeat',i)
    seed=i

    algorithms=['DASP-MNL','UCB','TS','UCB-MNL','ONM','Etc']
    for name in algorithms:
        std[name]=np.zeros(T,float)
        regret_sum_list[name]=np.zeros((repeat,T),float)
        avg_regret_sum[name]=np.zeros(T,float)
        std_R[name]=np.zeros(T,float)
        exp_reward[name]=[]
        oracle_reward[name]=[]       
        Env_dict[name]=linear_Env(seed,d,N,K)
        x,x2, L0=Env_dict[name].observe_x()
        oracle_reward[name],_,_=Env_dict[name].oracle_policy()
        

    alg_UCB=UCB(seed,d, N, K, T)
    alg_TS=TS(seed,d, N, K, T)
    alg_DASP=DASP_MNL(seed,d,N,K,T,L0)
    alg_Etc=Etc(seed,d,N,K,T)
    alg_ONM=ONM(seed,d,N,K,T,L0)

    algorithms=[alg_UCB,alg_Etc,alg_DASP,alg_TS,alg_ONM]
    # algorithms=[alg_TS]

    for algorithm in algorithms:
        algorithm.reset()
        name=algorithm.name()
        print(name)
        for t in tqdm((np.array(range(T))+1)):
            
            if t==1:
                algorithm.run(t,x,x2,[0])
            else:
                algorithm.run(t,x,x2,index[name])
            p=algorithm.offer_price()
            S=algorithm.offer()
            exp_reward[name].append(Env_dict[name].exp_reward(S,p))
            index[name]=Env_dict[name].observe(S,p)

 
    for algorithm in algorithms:
        name=algorithm.name()
        reg=np.array(oracle_reward[name])-np.array(exp_reward[name])
        regret_sum=np.cumsum(reg)
        regret_sum_list[name][i,:]=regret_sum
        avg_regret_sum[name]+=regret_sum  

        filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'repeat'+str(i)+'Reff.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(regret_sum, f)
            f.close()



def run_multiprocessing(T, repeat, d, N, K, load):
    Path("./result").mkdir(parents=True, exist_ok=True)
    Path("./plot").mkdir(parents=True, exist_ok=True)

    std=dict()
    S=dict()

    exp_reward=dict()
    avg_regret_sum=dict()
    regret_sum_list=dict()
    std_R=dict()
    regret=dict()
    algorithms=['DASP-MNL','UCB','TS','UCB-MNL','ONM','Etc']

    for algorithm in algorithms:
        std[algorithm]=np.zeros(T,float)
        exp_reward[algorithm]=[]    
        regret_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_regret_sum[algorithm]=np.zeros(T,float)
        std_R[algorithm]=np.zeros(T,float)
    
    if load==False:    
        
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(run, [( T, repeat, d, N, K, i) for i in range(repeat)])
    else:
        print('load data without running')
    seed=0
    Env=linear_Env(seed,d,N,K)    
    alg_UCB=UCB(seed,d, N, K, T)
    alg_TS=TS(seed,d, N, K, T)
    alg_DASP=DASP_MNL(seed,d,N,K,T)
    alg_Etc=Etc(seed,d,N,K,T)
    alg_ONM=ONM(seed,d,N,K,T)

    algorithms=[alg_DASP,alg_UCB,alg_TS,alg_ONM,alg_Etc]

    # algorithms=[alg_UCB,alg_TS]

    for algorithm in  algorithms:
        name=algorithm.name()
        for i in range(repeat):

            filename_1=name+'T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'repeat'+str(i)+'Reff.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret[name]=objects[0]
            regret_sum_list[name][i,:]=objects[0]
            avg_regret_sum[name]+=objects[0]   

        regret[name]=avg_regret_sum[name]/repeat
        std_R[name]=np.std(regret_sum_list[name],axis=0)

    T_p=int(T/10)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    ax.tick_params(labelsize=22)
    plt.rc('legend',fontsize=22)
    ax.yaxis.get_offset_text().set_fontsize(22)
    ax.xaxis.get_offset_text().set_fontsize(22)
    plt.gcf().subplots_adjust(bottom=0.17)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    color=['violet','royalblue','limegreen','lightsalmon','gray' ,'tomato']
    marker_list=['^','s','v','o','P','D']


    for i, algorithm in enumerate(algorithms):
        name=algorithm.name()
        col=color[i]
        mark=marker_list[i]
        ax.plot(range(T),regret[name],color=col, marker=mark, label=name, markersize=10,markevery=T_p)
        ax.errorbar(range(T), regret[name], yerr=1.96*std_R[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6)

    plt.title('N={}, K={}'.format(N,K),fontsize=25)
    plt.xlabel('Time step '+r'$t$',fontsize=25)
    plt.ylabel(r'$\mathcal{R}(t)$',fontsize=25)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    plt.legend(lines, labels, loc='best')  
    plt.savefig('./plot/T'+str(T)+'d'+str(d)+'N'+str(N)+'K'+str(K)+'repeat'+str(repeat)+'Reff.pdf', bbox_inches = "tight")
    plt.clf()
    

if __name__=='__main__':
    T=50000
    repeat=5
    load=False
    d=4
    N=10 # 40, 60, 80
    K=5
    print(d, N,K, repeat)
    run_multiprocessing(T,repeat,d,N,K,load)
