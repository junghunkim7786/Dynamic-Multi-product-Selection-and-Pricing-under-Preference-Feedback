from Environment import *
from Algorithms import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.gridspec as gridspec


def plot(T,repeat,d):

    exp_reward=dict()
    avg_regret_sum=dict()
    regret_sum_list=dict()
    std=dict()
    regret=dict()
    algorithms=['TS','UCB','Etc','DASP-MNL','ONM']

    for algorithm in algorithms:
        exp_reward[algorithm]=[]    
        regret_sum_list[algorithm]=np.zeros((repeat,T),float)
        avg_regret_sum[algorithm]=np.zeros(T,float)
        std[algorithm]=np.zeros(T,float)
    gs = gridspec.GridSpec(1,3) 
    fig = plt.figure(figsize=(18, 5))
    for n,N in enumerate([10, 15, 20]):
        for k,K in enumerate([5]):
            for algorithm in   algorithms:
                name=algorithm
                avg_regret_sum[algorithm]=np.zeros(T,float)

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
                std[name]=np.std(regret_sum_list[name],axis=0)


            T_p=int(T/8)
            ax = fig.add_subplot(gs[k, n])
            ax.tick_params(labelsize=22)
            plt.rc('legend',fontsize=22)
            ax.yaxis.get_offset_text().set_fontsize(22)
            ax.xaxis.get_offset_text().set_fontsize(22)
            plt.gcf().subplots_adjust(bottom=0.20)
            ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])

            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            color=['violet','royalblue','limegreen','lightsalmon','gray' ,'tomato']
            marker_list=['^','s','v','o','P','D']

            if N==15:
                for i, algorithm in enumerate(algorithms):
                    name=algorithm
                    if name=='UCB':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='UCBA-LCBP (Algorithm 1)', markersize=11,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='TS':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='TSA-LCBP (Algorithm 2)', markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='Etc':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='ETC', markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='ONM':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='ONM', markersize=16,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='DASP-MNL':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='DASP-MNL', markersize=12,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    else:
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label=name, markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                plt.title('N={}'.format(N),fontsize=22)
                plt.xlabel('Time step '+r'$t$',fontsize=22)
                plt.ylabel(r'$\mathcal{R}(t)$',fontsize=22)

            else:
                for i, algorithm in enumerate(algorithms):
                    name=algorithm
                    if name=='UCB':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='UCBA-LCBP (Algorithm 1)', markersize=11,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='TS':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='TSA-LCBP (Algorithm 2)', markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='Etc':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='ETC', markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='ONM':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='ONM', markersize=16,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    elif name=='DASP-MNL':
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label='DASP-MNL', markersize=12,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)
                    else:
                        col=color[i]
                        mark=marker_list[i]
                        ax.plot(range(T),regret[name],color=col, marker=mark, label=name, markersize=10,markevery=T_p,zorder=6-i)
                        ax.errorbar(range(T), regret[name], yerr=1.96*std[name]/np.sqrt(repeat), color=col, errorevery=T_p, capsize=6,zorder=6-i)

                plt.title('N={}'.format(N),fontsize=22)
                plt.xlabel('Time step '+r'$t$',fontsize=22)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels = [labels[4],labels[3],labels[2], labels[1], labels[0]]
    lines=[lines[4],lines[3],lines[2],lines[1],lines[0]]
    fig.legend(lines, labels, loc='upper center', ncol=5,bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.savefig('./plot/T'+str(T)+'d'+str(d)+'K'+str(K)+'repeat'+str(repeat)+'.pdf', bbox_inches = "tight")
    plt.show()  
    

if __name__=='__main__':
    d=4
    T=50000
    repeat=10
    plot(T,repeat,d)
   


