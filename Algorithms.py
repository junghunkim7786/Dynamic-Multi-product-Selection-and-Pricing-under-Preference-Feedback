import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from itertools import combinations
from tqdm import tqdm
from scipy.optimize import fmin_tnc
from scipy.optimize import minimize
from scipy.linalg import cholesky
from itertools import product
import torch
import copy
from scipy.stats import beta
from functools import reduce
import logging

#################    
    

class UCB:
    
    
    def construct_M(self):
        M=[]
        for size in [self.K]:

            for S in combinations(range(self.N), size):
                M.append(np.array(S))
        return M              

    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        return obj 
    
    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            
        norm_theta = np.linalg.norm(theta[self.d:2*self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[self.d:2*self.d] = theta[self.d:2*self.d] / norm_theta

        return theta
    
    def prob(self,i,S,theta):

        means = np.dot(self.z[S], theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        prob = np.exp(np.dot(self.z[i], theta)) / SumExp
        
        return prob

    def G(self,S,theta):
        gram=np.zeros((2*self.d,2*self.d))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.z[i],self.z[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.z[i],self.z[j])
        return gram
        
    def G_v(self,S,theta):
        gram=np.zeros((self.d,self.d))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.x[i],self.x[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.x[i],self.x[j])
        
        return gram
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+1/self.eta*np.linalg.inv(V)@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 5:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   


    def match_elements(self, S, index):

        matching_elements = [1 if element in index else 0 for element in S]

        return matching_elements
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
           
   
    def __init__(self,seed,d,N,K,T):
        print('UCB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=np.zeros((N,d))
        self.x2=np.zeros((N,d))

        self.S=[]
        self.S_hist=[]
        self.d=d
        self.alpha=10
        self.N=N
        self.K=K
        self.T=T
        self.eta=1/2*np.log(K+1)+3
        self.lamb=max(d*self.eta,self.eta)
        self.V=self.lamb*np.identity(2*self.d)
        self.V_v=self.lamb*np.identity(self.d)
        self.V_til=self.lamb*np.identity(2*self.d)
        self.z=np.zeros((self.N,2*self.d))
        self.p=np.zeros(N)
        self.theta=np.zeros(2*self.d)
        self.theta_v=np.zeros(self.d)
        self.v_upper=np.zeros(N)
        self.u_upper=np.zeros(N)
        self.g=np.zeros(2*self.d)
        self.det_V_fix=np.linalg.det(self.V)
        self.theta_fix=np.zeros(2*self.d)
        self.tau_max=self.d*math.log(self.T)
        self.C=1.01
    def run(self,t,x,x2,index):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            self.p=np.random.rand(self.N)

            self.x=x
            self.x=x2

            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            self.V_til=self.V+self.eta*self.G(self.S,self.theta) 
            self.V+= self.G(self.S,self.theta) 
            self.V_v+=self.G_v(self.S,self.theta) 

        else:
            y=self.match_elements(self.S, index)
            self.x=x
            self.x2=x2

            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            theta_prev=copy.deepcopy(self.theta)
            self.theta=self.fit(theta_prev,self.z,y,self.S,self.V_til)
            det=np.linalg.det(self.V)
            if det>(self.C)*self.det_V_fix:
                self.theta_fix=copy.deepcopy(self.theta)
                self.det_V_fix=copy.deepcopy(det)
            for n in range(self.N):
                self.u_upper[n]=self.z[n]@self.theta+self.alpha*np.sqrt(self.z[n]@np.linalg.inv(self.V)@self.z[n])+2*((self.C)**(1/2))*self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V_v)@self.x[n])
                self.v_upper[n]=self.x[n]@self.theta[0:self.d]+self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V_v)@self.x[n])
                self.p[n]=max(self.x[n]@self.theta_fix[0:self.d]-(self.C)**(1/2)*self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V_v)@self.x[n]),0)
                                  
            R=0
            tmp_R=0
            utility=self.v_upper*np.exp(self.u_upper)
            largest_indices = np.argsort(utility)[-self.K:][::-1]
            self.S=largest_indices
            
                
            self.V_til=self.V+self.eta*self.G(self.S,self.theta) 
            self.V+= self.G(self.S,self.theta) 
            self.V_v+=self.G_v(self.S,self.theta) 


    def offer(self):
        return self.S   
    def offer_price(self):
        return self.p   
    def name(self):
        return 'UCB'    
    
####################################


class TS:

    def construct_M(self):
        M=[]

        for size in [self.K]:
            for S in combinations(range(self.N), size):
                M.append(np.array(S))
        return M               

    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            
        norm_theta = np.linalg.norm(theta[self.d:2*self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[self.d:2*self.d] = theta[self.d:2*self.d] / norm_theta

        return theta
    
    def prob(self,i,S,theta):

        means = np.dot(self.z[S], theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        prob = np.exp(np.dot(self.z[i], theta)) / SumExp
        
        return prob

    def G(self,S,theta):
        gram=np.zeros((2*self.d,2*self.d))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.z[i],self.z[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.z[i],self.z[j])
        return gram
        
    def G_v(self,S,theta):
        gram=np.zeros((self.d,self.d))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.x[i],self.x[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.x[i],self.x[j])
        
        return gram
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+1/self.eta*np.linalg.inv(V)@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 5:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   
    
    def divide_into_groups(self, K, N):
        # Generate the input array [0, 1, ..., N]
        input_array = np.arange(N)
        # Split the shuffled array into K groups
        groups = np.array_split(input_array, K)

        return groups



    def match_elements(self, S, index):

            # Check if any element in the sublist matches any element in index
        matching_elements = [1 if element in index else 0 for element in S]
        # result.append(matching_elements)

        return matching_elements

        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,d,N, K,T):
        print('TS')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=np.zeros((N,d))
        self.x2=np.zeros((N,d))

        self.S=[]
        self.S_hist=[]
        self.d=d
        self.alpha=10
        self.N=N
        self.K=K
        self.T=T
        self.eta=1/2*np.log(K+1)+3
        self.lamb=max(d*self.eta,self.eta)
        self.V=self.lamb*np.identity(2*self.d)
        self.V_v=self.lamb*np.identity(self.d)
        self.V_til=self.lamb*np.identity(2*self.d)
        self.z=np.zeros((self.N,2*self.d))
        self.p=np.zeros(N)
        self.theta=np.zeros(2*self.d)
        self.theta_v=np.zeros(self.d)
        self.v_upper=np.zeros(N)
        self.u_upper=np.zeros(N)
        self.det_V_fix=np.linalg.det(self.V)
        self.theta_fix=np.zeros(2*self.d)
        self.C=1.01
    def run(self,t,x,x2,index):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            self.p=np.random.rand(self.N)
            self.x=x

            self.x2=x2
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            self.V_til=self.V+self.eta*self.G(self.S,self.theta) 
            self.V+= self.G(self.S,self.theta) 
            self.V_v+=self.G_v(self.S,self.theta) 
        else:
            y=self.match_elements(self.S, index) 
            self.x=x
            self.x2=x2
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            theta_prev=copy.deepcopy(self.theta)

            self.theta=self.fit(theta_prev,self.z,y,self.S,self.V_til)
            det=np.linalg.det(self.V)
            if det>(self.C)*self.det_V_fix:
                self.theta_fix=copy.deepcopy(self.theta)
                self.det_V_fix=copy.deepcopy(det)

            M=math.ceil(1-(np.log(2*self.N)/np.log(1-1/(4*np.log(math.e*math.pi)))))
            mean=self.theta
            cov=2*self.alpha**2*np.linalg.inv(self.V)
            theta_sample=np.random.multivariate_normal(mean, cov, M)
            mean_v=self.theta[0:self.d]
            cov_v=self.alpha**2*np.linalg.inv(self.V_v)
            theta_sample_v=np.random.multivariate_normal(mean_v, cov_v, M)
            for n in range(self.N):
                self.v_upper[n]=np.max(self.x[n]@theta_sample_v.T)
                self.u_upper[n]=np.max(self.z[n]@theta_sample.T)+(self.C)*(self.v_upper[n]-mean_v@self.x[n])
                self.p[n]=max(self.x[n]@self.theta_fix[0:self.d]-(self.C)**(1/2)*self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V_v)@self.x[n]),0)
            R=0
            tmp_R=0
            
            utility=self.v_upper*np.exp(self.u_upper)
            largest_indices = np.argsort(utility)[-self.K:][::-1]
            self.S=largest_indices

            self.V+= self.G(self.S,self.theta) 
            self.V_v+=self.G_v(self.S,self.theta) 
    def offer(self):
        return self.S   
    def offer_price(self):
        return self.p       
    def name(self):
        return 'TS'    


####################################


class Etc:
    
    
    def construct_M(self):
        M=[]
#         for size in range(1, self.L + 1):
        for size in [self.L]:
            for S in combinations(range(self.N), size):
                M.append(np.array(S))
        return M              

    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        return obj 
    
    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            
        norm_theta = np.linalg.norm(theta[self.d:2*self.d])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[self.d:2*self.d] = theta[self.d:2*self.d] / norm_theta

        return theta
    def compute_prob_loss(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means.astype(float))
        SumExp = np.sum(u)+1
        if 1 in y:
            prob = u / SumExp
        else: 
            prob = 1 / SumExp
             
        return prob    
    def compute_prob_grad(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        prob = u / SumExp
   

        return prob        
    def cost_function(self, theta, *args):
        x, y, S= args[0], args[1], args[2]
        loss=0
        for t in range(len(S)):
            if len(S[t])!=0:    
                x_=[x[t][i] for i in S[t]]
                y_=y[t]
                prob = self.compute_prob_loss(theta, x_, y_)
                loss+=-np.sum(np.multiply(y_, np.log(prob)))
        return loss + (1/2)*self.lamb* np.linalg.norm(theta)**2

    def gradient(self, theta, *args):
        x, y, S = args[0], args[1], args[2]
        m = 1
        grad=0
        for t in range(len(S)):
            if len(S[t])!=0:
                x_=x[t][S[t]]
                y_=y[t]
                prob = self.compute_prob_grad(theta, x_, y_)
                eps = (prob - y_)
                prod = eps[:, np.newaxis] * x_
                grad+=(1/m)*prod.sum(axis=0)
        grad=grad+self.lamb*theta
        return grad

    def fit(self, theta, *args):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=args, ftol=1e-6, disp=False,approx_grad=True)
        w = opt_weights[0]
        return w
    def match_elements(self, S, index):
        # result = []

            # Check if any element in the sublist matches any element in index
        matching_elements = [1 if element in index else 0 for element in S]

        return matching_elements
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
           
            
    def __init__(self,seed,d,N,L,T,pricing_strategy='l'):
        print('Etc')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=np.zeros((N,d))
        self.x2=np.zeros((N,d))

        self.S=[]
        self.S_hist=[]
        self.z_hist=[]    
        self.d=d
        self.N=N
        self.L=L
        self.T=T
        self.V=np.identity(2*self.d)
        self.V_v=np.identity(self.d)
        self.z=np.zeros((self.N,2*self.d))
        self.p=np.zeros(N)
        self.theta=np.zeros(2*self.d)
        self.theta_v=np.zeros(self.d)
        self.v_upper=np.zeros(N)
        self.u_upper=np.zeros(N)
        self.strategy=pricing_strategy
        self.g=np.zeros(2*self.d)
        self.lamb=1
    def optimize_prices(self, S):
    
        
        # Objective function to maximize (negative because we use minimize)
        def objective(p):
            return -self.exp_reward_obj(S, p)
        
        # Initial prices
        initial_prices = np.random.rand(len(S))
        
        # Constraints: prices must be non-negative
        bounds = [(0, max(np.dot(self.x[i], self.theta[0:self.d]),0)) for i in S]
        
        result = minimize(objective, initial_prices, bounds=bounds, method='L-BFGS-B')

        return result.x if result.success else initial_prices


    def exp_reward_obj(self,S,p):
        R=0

        R+=np.sum(p*np.exp(self.x[S]@self.theta[0:self.d]-self.x[S]@self.theta[self.d:2*self.d]*p))/(1+np.sum(np.exp(self.x[S]@self.theta[0:self.d]-self.x[S]@self.theta[self.d:2*self.d]*p)))
        return R

    def oracle_policy(self):

        max_revenue = 0
        optimal_set = []
        optimal_prices = []

        max_revenue = 0
        optimal_set = []
        optimal_prices = []
        p=self.x@self.theta[0:self.d]
        utility=p*np.exp(self.x@self.theta[0:self.d]-self.x@self.theta[self.d:2*self.d]*p)
        largest_indices = np.argsort(utility)[-self.L:][::-1]
        S=largest_indices
        max_revenue = self.exp_reward_obj(S, p[S])
        optimal_set = S
        optimal_prices = p

        return max_revenue, optimal_set, optimal_prices
    
    def run(self,t,x,x2,index):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            self.p=np.random.rand(self.N)

            self.x=x
            self.x=x2

        
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            for n in self.S:
                self.V+=np.outer(self.z[n],self.z[n])
                self.V_v+=np.outer(self.x[n],self.x[n])
        elif t<round(self.T**(2/3)):
            y=self.match_elements(self.S, index)

            self.y_hist.append(y)
            self.S_hist.append(self.S)     
            self.z_hist.append(self.z)     

            self.x=x
            self.x2=x2

            self.S=copy.deepcopy(random.choice(self.M))
            self.p=np.random.rand(self.N)
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            for n in self.S:
                self.V+=np.outer(self.z[n],self.z[n])
                self.V_v+=np.outer(self.x[n],self.x[n])

        elif t==round(self.T**(2/3)):
            y=self.match_elements(self.S, index) 
            self.y_hist.append(y)
            self.S_hist.append(self.S)
            self.z_hist.append(self.z)        
            self.theta_0=np.ones(int(2*self.d))
            self.theta=self.fit(self.theta_0,self.z_hist,self.y_hist,self.S_hist)
            
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)

            _, self.S, self.p= self.oracle_policy()
            
    def offer(self):
        return self.S   
    def offer_price(self):
        return self.p   
    def name(self):      
        return 'Etc'    


######
"""
We utilize the code provided by Erginbas, Y. E., Ramchandran, K., & Courtade, T. (2023). Available at https://openreview.net/forum?id=J7hbPeOZ39
"""

class DASP_MNL:

    def solve_assortment_and_pricing(self, K, alpha, beta):
        B = 1
        assortment = np.array([])
        B_min = 0
        B_max = 1000
        while B_max - B_min > 1e-6:
            B = (B_max + B_min)/2
            v_of_B = np.exp(alpha - beta * B - 1) / beta
            assortment = np.argpartition(v_of_B, -K)[-K:]
            B_achieved = np.sum(v_of_B[assortment])
            if B_achieved < B:
                B_max = B
            else:
                B_min = B

        prices = 1 / beta + B
        return assortment, prices


    def grad_reduce(self,input_grad, sum_grad):
        return sum_grad[0] + input_grad[0], sum_grad[1] + input_grad[1]


    def solve_mle(self,offered_contexts, selected_contexts, d, init_theta=None):
        if init_theta is None or np.isnan(init_theta).any():
            theta = 0.5 * np.ones(d)
        else:
            theta = init_theta
        iter = 0
        while True:
            def grad_map(input_contexts):
                offered_contexts, selected_contexts = input_contexts
                utilities = offered_contexts @ theta
                terms = np.exp(utilities)
                probs = terms / (1 + np.sum(terms))
                grad_new = - np.sum((probs * offered_contexts.T).T, axis=0) + selected_contexts
                W = np.diag(probs) - np.outer(probs, probs)
                hess_new = offered_contexts.T @ W @ offered_contexts
                return grad_new, hess_new
            results = map(grad_map, zip(offered_contexts, selected_contexts))
            grad, hess = reduce(self.grad_reduce, results)
            iter += 1
            update = np.linalg.inv(hess) @ grad
            theta = theta + update
            if np.linalg.norm(update) < 1e-5 or iter > 5:
                break
        print()
        X_data = np.concatenate(offered_contexts, axis=0)

        cov = X_data.T @ X_data / len(X_data)
        eigvals = np.linalg.eigvals(cov)
        return theta


    def mle_online_update(self, theta, V, offered_contexts, selected_contexts, kappa):
        utilities = offered_contexts @ theta
        terms = np.exp(utilities)
        probs = terms / (1 + np.sum(terms))
        G = - np.sum((probs * offered_contexts.T).T, axis=0) + selected_contexts
        theta = np.linalg.pinv(V) @ (V @ theta + G / kappa)
        return theta
    
    def update(self,t,  i_t, contexts,  contexts2,assortment, prices):
        x_tilde = np.concatenate([contexts[assortment], (- prices[assortment] * contexts2[assortment].T).T], axis=1)
        if t < self.T0:
            self.offered_contexts.append(x_tilde)
            if i_t is not None:
                self.selected_contexts.append(np.concatenate([contexts[i_t], - prices[i_t] * contexts2[i_t]]))
            else:
                self.selected_contexts.append(np.zeros(2 * self.d))
        elif t == self.T0:
            self.theta = self.solve_mle(self.offered_contexts, self.selected_contexts, 2 * self.d, init_theta=self.theta)
        else:
            offered_contexts = x_tilde
            if i_t is not None:
                selected_contexts = np.concatenate([contexts[i_t], - prices[i_t] * contexts2[i_t]])
            else:
                selected_contexts = np.zeros(2 * self.d)
            self.theta = self.mle_online_update(self.theta, self.V, offered_contexts, selected_contexts, self.kappa)



    def construct_M(self):
        M=[]      
        for size in range(1, self.L + 1):
            for S in combinations(range(self.N), size):
                M.append(np.array(S))
        return M              

    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        g=0
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            g += p
        obj=g@(theta-theta_prev)+(1/2)*np.sqrt((theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev))    
        return obj 

    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        if len(S)>0:
            obj_func = lambda theta: self.loss(theta,theta_prev, *args)
            constraint = ({'type': 'ineq', 'fun': lambda theta: 1 - np.linalg.norm(theta, ord=2)})
            result = minimize(obj_func, theta_prev, constraints=constraint, method='SLSQP')
            theta_update=result.x
        else:
            theta_update=theta_prev

        return theta_update


    def match_elements(self, S, index):

            # Check if any element in the sublist matches any element in index
        matching_elements = [1 if element in index else 0 for element in S]

        return matching_elements
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __init__(self,seed,d,N,L,T,L0=0.1):
        print('DASP-MNL')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=np.zeros((N,d))
        self.x2=np.zeros((N,d))

        self.offered_contexts = []
        self.selected_contexts = []
        self.S=[]
        self.S_hist=[]
        self.d=d
        self.kappa= 1/(L+1)**2
        self.N=N
        self.L=L
        self.T=T
        self.lamb=1
        self.V=self.lamb*np.identity(2*self.d)
        self.V_v=self.lamb*np.identity(self.d)
        self.z=np.zeros((self.N,2*self.d))
        self.p=np.zeros(N)
        self.theta=np.zeros(2*self.d)
        self.theta_v=np.zeros(self.d)
        self.v_upper=np.zeros(N)
        self.u_upper=np.zeros(N)
        self.L0 = L0
        self.P=1+(1+2*max(1,np.log(L)))/L0
        self.T0 = (d+np.log(T)+self.P**2)/(self.kappa**2)
        self.alpha= 10

    def run(self, t,x,x2,index):
        if t < self.T0:
            self.S = np.random.choice(self.N, size=self.L, replace=False)
            self.p = np.random.choice(2, size=self.N)
            self.x=x
            self.x2=x2
 
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            y=self.match_elements(self.S, index)
            if 1 in y:
                i_t=np.argmax(y)
            else:
                i_t=None
            self.update(t,i_t, self.x,self.x2, self.S, self.p)
            for n in self.S:
                self.V+=np.outer(self.z[n],self.z[n])
        else:
            y=self.match_elements(self.S, index)
            self.x=x
            self.x2=x2

            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            if 1 in y:
                i_t=np.argmax(y)
            else:
                i_t=None
            self.update(t,i_t, self.x,self.x2, self.S, self.p)
            
            psi, phi = self.theta[:self.d], self.theta[self.d:]
            contexts_double = np.concatenate([x, x], axis=1)
            g = np.zeros(self.N)
            V_inv = np.linalg.inv(self.V)
            for i in range(self.N):
                g[i] = self.alpha * np.inner(V_inv @ contexts_double[i], contexts_double[i])
            alpha = np.minimum(x @ psi + g, 1)
            beta = np.maximum(x @ phi - g, self.L0)
            self.S, self.p = self.solve_assortment_and_pricing(self.L, alpha, beta)
            for n in self.S:
                self.V+=np.outer(self.z[n],self.z[n])

    def offer(self):
        return self.S   
    def offer_price(self):
        return self.p   
    def name(self):
        return 'DASP-MNL'    
    

######
"""
We utilize the code provided by Erginbas, Y. E., Ramchandran, K., & Courtade, T. (2023). Available at https://openreview.net/forum?id=J7hbPeOZ39
"""


class ONM:

    def solve_assortment_and_pricing(self, K, alpha, beta):
        B = 1
        assortment = np.array([])
        B_min = 0
        B_max = 1000
        while B_max - B_min > 1e-6:
            B = (B_max + B_min)/2
            v_of_B = np.exp(alpha - beta * B - 1) / beta
            assortment = np.argpartition(v_of_B, -K)[-K:]
            B_achieved = np.sum(v_of_B[assortment])
            if B_achieved < B:
                B_max = B
            else:
                B_min = B

        prices = 1 / beta + B
        return assortment, prices


    def grad_reduce(self,input_grad, sum_grad):
        return sum_grad[0] + input_grad[0], sum_grad[1] + input_grad[1]


    def solve_mle(self,offered_contexts, selected_contexts, d, init_theta=None):
        if init_theta is None or np.isnan(init_theta).any():
            theta = 0.5 * np.ones(d)
        else:
            theta = init_theta
        utilities = offered_contexts @ theta
        terms = np.exp(utilities)
        probs = terms / (1 + np.sum(terms))
        grad = - np.sum((probs * offered_contexts.T).T, axis=0) + selected_contexts

        mu=1/(2*(1+2*math.sqrt(6*self.L)*2))
       
        update = (1/mu)*np.linalg.inv(self.V) @ grad
        theta = theta + update
        if np.linalg.norm(theta)**2>2:
            theta= math.sqrt(2)*theta/np.linalg.norm(theta)

        return theta


    def update(self, t,  i_t, contexts,contexts2, assortment, prices):
        x_tilde = np.concatenate([contexts[assortment], (- prices[assortment] * contexts2[assortment].T).T], axis=1)
        offered_contexts = x_tilde
        if i_t is not None:
            selected_contexts = np.concatenate([contexts[i_t], - prices[i_t] * contexts2[i_t]])
        else:
            selected_contexts = np.zeros(2*self.d)
        self.theta = self.solve_mle(offered_contexts, selected_contexts, 2*self.d)

    def prob(self,i,S,theta):

        means = np.dot(self.z[S], theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(2*self.d)
        # for i,n in enumerate(S):
        prob = np.exp(np.dot(self.z[i], theta)) / SumExp
        
        return prob

    def G(self,S,theta):
        gram=np.zeros((2*self.d,2*self.d))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.z[i],self.z[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.z[i],self.z[j])
        return gram
        
    def construct_M(self):
        M=[]
        for size in range(1, self.L + 1):
            for S in combinations(range(self.N), size):
                M.append(np.array(S))
        return M              

    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        g=0
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            g += p
        obj=g@(theta-theta_prev)+(1/2)*np.sqrt((theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev))    
        return obj 

    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        if len(S)>0:
            obj_func = lambda theta: self.loss(theta,theta_prev, *args)
            constraint = ({'type': 'ineq', 'fun': lambda theta: 1 - np.linalg.norm(theta, ord=2)})
            result = minimize(obj_func, theta_prev, constraints=constraint, method='SLSQP')
            theta_update=result.x
        else:
            theta_update=theta_prev

        return theta_update



    def match_elements(self, S, index):

            # Check if any element in the sublist matches any element in index
        matching_elements = [1 if element in index else 0 for element in S]

        return matching_elements
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
            
    def __init__(self,seed,d,N,L,T,L0=0.1):
        print('ONM')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=np.zeros((N,d))
        self.x2=np.zeros((N,d))

        self.offered_contexts = []
        self.selected_contexts = []
        self.S=[]
        self.S_hist=[]
        self.d=d
        self.kappa=1/(L+1)**2
        self.N=N
        self.L=L
        self.T=T
        self.lamb=1
        self.V=self.lamb*np.identity(2*self.d)
        self.V_v=self.lamb*np.identity(self.d)
        self.z=np.zeros((self.N,2*self.d))
        self.p=np.zeros(N)
        self.theta=np.zeros(2*self.d)
        self.theta_v=np.zeros(self.d)
        self.v_upper=np.zeros(N)
        self.u_upper=np.zeros(N)
        self.L0 = L0
        self.P=1+(1+2*max(1,np.log(L)))/L0
        self.T0 = (d+np.log(T)+self.P**2)/(self.kappa**2)

    def run(self, t,x,x2,index):
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            self.p=np.random.rand(self.N)
            self.x=x
            self.x2=x2

            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            self.V+= self.G(self.S,self.theta) 

        else:
            y=self.match_elements(self.S, index)
            self.x=x
            self.x2=x2
            self.z=np.concatenate((self.x,-self.p[:,np.newaxis]*self.x2),axis=1)
            if 1 in y:
                i_t=np.argmax(y)
            else:
                i_t=None
            self.update(t,i_t, self.x,self.x2, self.S, self.p)

            psi, phi = self.theta[:self.d], self.theta[self.d:]
            alpha = np.minimum(x @ psi, 1)
            beta = np.maximum(x @ phi, self.L0)
            self.S, self.p = self.solve_assortment_and_pricing(self.L, alpha, beta)
            for n in range(self.N):
                if random.uniform(0,1)>1/2:
                    self.p[n]=self.p[n]+1/(2*t**(1/4))
                else:
                    self.p[n]=self.p[n]-1/(2*t**(1/4))

            self.V+= self.G(self.S,self.theta) 




    def offer(self):
        return self.S   
    def offer_price(self):
        return self.p   
    def name(self):
        return 'ONM'    
    


