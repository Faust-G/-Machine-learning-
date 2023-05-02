import numpy as np
import random
import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import randint
from scipy.stats import bernoulli
from functions import *
from copy import deepcopy
from scipy.sparse.linalg import svds
from pathlib import Path

def get_grad(A, y, l2, x, idx = None):
    if idx is None:
        grad = np.divide(y, 1 + np.exp(y * (A @ x)))
        grad =  A.T @ grad / -len(y) + l2 * x
    else:
        grad = np.divide(y[idx], 1 + np.exp(y[idx] * (A[idx, :] @ x)))
        grad =  A[idx, :].T @ grad / -len(idx) + l2 * x
    return grad

def prox_R(x, lamb):
    p = np.abs(x) - lamb * np.ones_like(x)
    p[p <= 0] = 0
    return p * np.sign(x)

def gd(filename, x_init, A, y, gamma, 
         l2=0, sparse=True, l1=0, S=1000, max_t=np.inf,
         save_info_period=10, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    x = np.array(x_init)
    
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
   
    #метод
    for it in range(S):
        
        x = x - gamma * (get_grad(A, y, l2, x) + l1 * np.sign(x))
        num_of_data_passes += 1
        
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    #сохранение результатов в файл
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_GD_gamma_"+str(gamma)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def FISTA(filename, x_init, A, y, L, 
         mu, sparse=True, l1=0, S=1000, max_t=np.inf,
         save_info_period=10, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    
    k = L / mu
    xk = np.array(x_init)
    xk1 = np.array(x_init)
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(xk, [A, y, mu, sparse, l1])-f_star])
    sq_distances = np.array([norm(xk - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
   
    #метод
    for it in range(S):
        
        last = np.copy(xk1)
        xk1 = prox_R(xk - get_grad(A, y, mu, xk) / L, l1 / L)
        xk = xk1 + (np.sqrt(k) - 1) / (np.sqrt(k) + 1) * (xk1 - last)
        num_of_data_passes += 1
        
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(xk1, [A, y, mu, sparse, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(xk1 - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(xk1, [A, y, mu, sparse, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(xk1 - ref_point) ** 2)
    
    #сохранение результатов в файл
    res = {'last_iter':xk, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_FISTA_l2_"+str(mu)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res


def prox_gd(filename, x_init, A, y, gamma, 
         l2=0, sparse=True, l1=0, S=1000, max_t=np.inf,
         save_info_period=10, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    x = np.array(x_init)
    
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse, l1])])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
   
    #метод
    for it in range(S):

        g = get_grad(A, y, l2, x)
        x = prox_R(x - gamma * g, gamma * l1)
        num_of_data_passes += 1

        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    #сохранение результатов в файл
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_prox-GD_gamma_"+str(gamma)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def sgd_decr_stepsize(filename, x_init, A, y, gamma_schedule, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    x = np.array(x_init)
    
    gamma = gamma_schedule[0]
    decr_period = gamma_schedule[1]
    decr_coeff = gamma_schedule[2]
    # number_of_decreases = 0
    number_of_decreases = 1
    
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    #метод
    d = 0
    for it in range(int(S*m/batch_size)):

        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
        
        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size

        if d >= decr_period * number_of_decreases:
            gamma *= decr_coeff
            number_of_decreases += 1
        d = d + batch_size / m
        g = get_grad(A_for_batch, y, l2, x, batch_ind)
        x = prox_R(x - gamma * g, gamma * l1)

        num_of_data_passes += batch_size/m

        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    #сохранение результатов в файл
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SGD_decr_stepsize_gamma_"+str(gamma_schedule[0])+"_decr_period_"
              +str(decr_period)+"_decr_coeff_"+str(decr_coeff)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res


def sgd_const_stepsize(filename, x_init, A, y, gamma, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(int(S*m*1.0/batch_size), int(100000/batch_size))*batch_size)
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    x = np.array(x_init)
    
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0

    #метод
    for it in range(int(S*m/batch_size)):
        if indices_counter == indices_size:
            indices_counter = 0
            indices = randint.rvs(low=0, high=m, size=indices_size)
            

        batch_ind = indices[indices_counter:(indices_counter+batch_size)]
        indices_counter += batch_size  
        g = get_grad(A_for_batch, y, l2, x, batch_ind)
        x = prox_R(x - gamma * g, gamma * l1)
        num_of_data_passes += batch_size/m

        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    #сохранение результатов в файл
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SGD_const_stepsize_gamma_"+str(gamma)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def svrg(filename, x_init, A, y, gamma, 
         l2=0, sparse_full=True, sparse_stoch=False, l1=0, S=50, M=None, max_t=np.inf,
         batch_size=1, indices=None, save_info_period=100, x_star=None, f_star=None):
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    if M is None:
        M = int(2 * m / batch_size)
    if indices is None:
        indices = randint.rvs(low=0, high=m, size=min(M*batch_size*S, int(100000/batch_size)*batch_size))
    indices_size = len(indices)
    if x_star is None:
        x_star = np.zeros(n)
    ref_point = np.array(x_star) #если знаем решение, то ref_point поможет вычислять расстояние до него
    if f_star is None:
        f_star = 0
    x = np.array(x_init)
    
    #эти массивы мы будем сохранять в файл
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, l1]) - f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    
    t_start = time.time()
    num_of_data_passes = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0 #нужен для того, чтобы проходить массив индексов indices
    # filename = "dump/"+filename+"_L.txt"
    # with open(filename, 'rb') as file:
    #     L, _, _ = pickle.load(file)

    #метод    
    w = np.copy(x)
    for s in range(S):
    
        full_grad = get_grad(A, y, l2, x)
        for it in range(M):
            #если закончились индексы, то нужно ещё насэмплировать
            if indices_counter == indices_size:
                indices_counter = 0
                indices = randint.rvs(low=0, high=m, size=indices_size)
            batch_ind = indices[indices_counter:(indices_counter+batch_size)]
            indices_counter += batch_size
            
            g = get_grad(A_for_batch, y, l2, x, batch_ind) - get_grad(A_for_batch, y, l2, w, batch_ind)+ full_grad
            x = prox_R(x - gamma * g, gamma * l1)
            

            num_of_data_passes += 2.0*batch_size/m
            if ((s * M + it + 1) % save_info_period == 0):
                its = np.append(its, s * M + it + 1)
                tim = np.append(tim, time.time() - t_start)
                data_passes = np.append(data_passes, num_of_data_passes)
                func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
                sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        w = np.copy(x)
        if tim[-1] > max_t:
            break
    
    if ((s * M + it + 1) % save_info_period != 0):
        its = np.append(its, s * M + it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, l1])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
    
    #сохранение результатов в файл
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances}
    
    with open("dump/"+filename+"_SVRG_gamma_"+str(gamma)+"_l2_"+str(l2)+"_l1_"+str(l1)+"_num_of_epochs_"+str(S)
              +"_epoch_length_"+str(M)+"_batch_size_"+str(batch_size)+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res



