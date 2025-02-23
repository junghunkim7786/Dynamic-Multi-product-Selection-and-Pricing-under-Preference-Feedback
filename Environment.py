import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from itertools import combinations
from itertools import product
from scipy.optimize import minimize
from itertools import combinations

class linear_Env:
    def __init__(self,seed,d,N,L):
        np.random.seed(seed)
        self.d=d
        self.N=N
        self.L=L
        self.x=np.zeros((self.N,self.d)) ## observed feature
        self.x2=np.zeros((self.N,self.d)) ## observed feature

        self.theta=np.zeros((self.d))
        self.theta_s=np.zeros((self.d))
        self.theta=np.random.uniform(0,1,self.d)
        self.theta=self.theta/np.sqrt(np.sum(self.theta**2))
        self.theta_s=np.random.uniform(0,1,self.d)
        self.theta_s=self.theta_s/np.sqrt(np.sum(self.theta_s**2))
        

        self.index=np.zeros(1)
        self.p=np.zeros(self.N)
        self.prev_t=0
        self.L0=0
    def observe_x(self):
        for n in range(self.N):
            self.x[n]=np.random.uniform(0,1,self.d)
            self.x[n]=self.x[n]/np.sqrt(np.sum(self.x[n]**2))
            self.x2[n]=np.random.uniform(0,1,self.d)
            self.x2[n]=self.x2[n]/np.sqrt(np.sum(self.x2[n]**2))
            
        self.L0=min(self.theta_s@self.x.T)

        
        return self.x,self.x2,self.L0
    
    def observe(self,S,p):
        index=[]
        S_=[i for i in S if p[i]<=max(self.x[i]@self.theta,0)]
        if len(S_)==0:
            index.append(None)
        else:
            prob=np.zeros(len(S_)+1)
            prob[1:1+len(S_)]=np.exp(self.x[S_]@self.theta-self.x2[S_]@self.theta_s*p[S_])/(1+np.sum(np.exp(self.x[S_]@self.theta-self.x2[S_]@self.theta_s*p[S_])))
            prob[0]=1- np.sum(prob[1:1+len(S_)])
            index_list=np.insert(S_, 0, self.N)
            x=np.random.choice(index_list, p=prob)
            index.append(x)
        return index  

    def exp_reward(self,S,p):
        R=0

        S_=[i for i in S if p[i]<=max(self.x[i]@self.theta,0)]
        R+=np.sum(p[S_]*np.exp(self.x[S_]@self.theta-self.x2[S_]@self.theta_s*p[S_]))/(1+np.sum(np.exp(self.x[S_]@self.theta-self.x2[S_]@self.theta_s*p[S_])))
        return R
    
    def exp_reward_obj(self,S,p):
        R=0

        R+=np.sum(p*np.exp(self.x[S]@self.theta-self.x2[S]@self.theta_s*p))/(1+np.sum(np.exp(self.x[S]@self.theta-self.x2[S]@self.theta_s*p)))
        return R
        

    def optimize_prices(self, S):
        """
        Optimize the prices for a given set of arms using gradient-based optimization.
        
        Parameters:
        - S: Indices of the arms in the set.
        - z: Feature vectors z for all arms.
        - x: Feature vectors x for all arms.
        - theta_star: Parameter vector theta^*.
        - theta_v: Parameter vector theta_v.
        
        Returns:
        - Optimal prices for the arms in S.
        """
        # Objective function to maximize (negative because we use minimize)
        def objective(p):
            return -self.exp_reward_obj(S, p)
        
        # Initial prices
        initial_prices = np.random.rand(len(S))
        
        # Constraints: prices must be non-negative
        bounds = [(0, max(np.dot(self.x[i], self.theta),0)) for i in S]
        
        result = minimize(objective, initial_prices, bounds=bounds, method='L-BFGS-B')


        return result.x if result.success else initial_prices
   
    def oracle_policy(self):

        # max_revenue = 0
        # optimal_set = []
        # optimal_prices = []

        max_revenue = 0
        optimal_set = []
        optimal_prices = []
        p=self.x@self.theta
        utility=p*np.exp(self.x@self.theta-self.x2@self.theta_s*p)
        largest_indices = np.argsort(utility)[-self.L:][::-1]
        S=largest_indices
        max_revenue = self.exp_reward(S, p)
        optimal_set = S
        optimal_prices = p

        return max_revenue, optimal_set, optimal_prices


