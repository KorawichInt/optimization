#!/usr/bin/env python3

from bees_algorithm.bees_algorithm import BeesAlgorithm
from bees_algorithm.bees_algorithm_parallel_algorithm import ParallelBeesAlgorithm, FullyParallelBeesAlgorithm
from bees_algorithm.bees_algorithm_parallel_testing import BeesAlgorithmTester
import time, math, logging
import benchmark_functions as bf
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


Griewank_bees_parameters=	{'ns':0,	'nb':18,	'ne':1,	'nrb':5,	'nre':10,	'stlim':5}
Ackley_bees_parameters=		{'ns':0,	'nb':8,		'ne':1,	'nrb':10,	'nre':20,	'stlim':5}
Easom_bees_parameters=		{'ns':0,	'nb':14,	'ne':1,	'nrb':5,	'nre':30,	'stlim':10}
Schwefel_bees_parameters=	{'ns':0,	'nb':14,	'ne':1,	'nrb':5,	'nre':30,	'stlim':10}

def test_on_function(test_function,lower_bound,upper_bound,bees_parameters,optimum_score,ba_class=BeesAlgorithm):
	n_runs=5
	print("Run\tIteration\tScore")
	print("="*30)
	results=[]
	for i in range(n_runs):
		a=ba_class(test_function,
							lower_bound,upper_bound,
							ns=bees_parameters['ns'],nb=bees_parameters['nb'],ne=bees_parameters['ne'],nrb=bees_parameters['nrb'],
							nre=bees_parameters['nre'],stlim=bees_parameters['stlim']) 
		it, score = a.performFullOptimisation(max_iteration=5000,max_score=optimum_score-0.001)
		print(str(i)+'\t'+str(it)+'\t'+str(score))
		results+=[(it,score)]
	mu_it=sum([float(r[0]) for r in results])/n_runs
	mu_score=sum([r[1] for r in results])/n_runs
	var_it=math.sqrt(sum([pow(r[0]-mu_it,2) for r in results])/n_runs)
	var_score=math.sqrt(sum([pow(r[1]-mu_score,2) for r in results])/n_runs)
	print('')
	print("Iterations Average "+str(mu_it)+" Standard Deviation "+str(var_it))
	print("Score Average "+str(mu_score)+" Standard Deviation "+str(var_score))

def test_on_function_parallel_tester(test_function,lower_bound,upper_bound,bees_parameters,optimum_score):
	tester=BeesAlgorithmTester(	test_function,
															lower_bound,upper_bound,
															ns=bees_parameters['ns'],nb=bees_parameters['nb'],ne=bees_parameters['ne'],nrb=bees_parameters['nrb'],
															nre=bees_parameters['nre'],stlim=bees_parameters['stlim'],useSimplifiedParameters=True)
	start_time = time.time()
	tester.run_tests(n_tests=200,max_iteration=5000,max_score=optimum_score-0.001,verbose=False)
	end_time = time.time() - start_time
	return end_time, tester
		
def test_on_function_parallel_algorithm(test_function,lower_bound,upper_bound,bees_parameters,optimum_score,fully=False):
	n_runs=50
	print("Run\tIteration\tScore")
	print("="*30)
	results=[]
	for i in range(n_runs):
		if not fully:
			a=ParallelBeesAlgorithm(test_function,
											lower_bound,upper_bound,
											ns=bees_parameters['ns'],nb=bees_parameters['nb'],ne=bees_parameters['ne'],nrb=bees_parameters['nrb'],
											nre=bees_parameters['nre'],stlim=bees_parameters['stlim'],useSimplifiedParameters=True) 
		else:
			a=FullyParallelBeesAlgorithm(test_function,
											lower_bound,upper_bound,
											nb=bees_parameters['nb'],nrb=bees_parameters['nrb'],
											stlim=bees_parameters['stlim'],useSimplifiedParameters=True)
		it,score = a.performFullOptimisation(max_iteration=5000,max_score=optimum_score-0.001)
		if i%5==0:
			print(str(i)+'\t'+str(it)+'\t'+str(score))
		results+=[(it,score)]
	mu_it=sum([float(r[0]) for r in results])/n_runs
	mu_score=sum([r[1] for r in results])/n_runs
	var_it=math.sqrt(sum([pow(r[0]-mu_it,2) for r in results])/n_runs)
	var_score=math.sqrt(sum([pow(r[1]-mu_score,2) for r in results])/n_runs)
	print('')
	print("Iterations Average",mu_it,"Standard Deviation",var_it)
	print("Score Average",mu_score,"Standard Deviation",var_score)


# ขอบเขตล่างและขอบเขตบน
lower_bound = [0,6]
upper_bound = [20,8]

data = pd.read_csv('fish_data.csv')

# Extract the x, y, and z columns from the CSV
x_data = data['average_weight'].values
y_data = data['ph_of_water'].values
z_data = data['life_span'].values
points = np.column_stack((x_data, y_data))

# A function that returns the z-value based on input params
def new_function(params):
    x, y = params  # Unpack the params (x and y values)
    z = griddata(points, z_data, (x, y), method='linear')
    if np.isnan(z):
        z = 0  
    return z
max_score = z_data.max()

def test_bees_algorithm():
	start_time_overall = time.time()
	start_time = time.time()
	print(f"Function Schwefel -----------------------------------)")
	test_on_function(new_function, lower_bound, upper_bound, Schwefel_bees_parameters, max_score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	start_time = time.time()
	print('')
	print("Function Easom -------------------------------------")
	test_on_function(new_function, lower_bound, upper_bound, Easom_bees_parameters, max_score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	start_time = time.time()
	print('')
	print("Function Ackley -----------------------------------------")
	test_on_function(new_function, lower_bound, upper_bound, Ackley_bees_parameters, max_score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	start_time = time.time()
	print('')
	print("Function Griewank -----------------------------------------")
	test_on_function(new_function, lower_bound, upper_bound, Griewank_bees_parameters, max_score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	end_time = time.time() - start_time_overall
	print('')
	print("Elapsed time (overall): "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))


# Griewank_bees_parameters
# Ackley_bees_parameters
# Easom_bees_parameters
# Schwefel_bees_parameters
# def new_function(params):
#     a = params[0]
#     b = params[1]
    
#     # Introducing complexity with multiple operations
#     # term1 = np.sin(a) * np.exp(b)  # Sine of 'a' multiplied by the exponential of 'b'
#     term2 = (a - b)**2  # Squared difference between 'a' and 'b'
#     # term3 = np.log(abs(a + 1)) + np.log(abs(b + 1))  # Logarithmic terms (ensuring no log of 0 or negative)
#     # term4 = np.cos(a * b)  # Cosine of the product of 'a' and 'b'

#     # Combine the terms in some manner (sum, product, etc.)
#     # result = term1 + term2 + term3 - term4
    
#     return term2


def test_visualisation():
	# b_func = bf.Easom(opposite=True)
	b_func = bf.Ackley(n_dimensions=2, opposite=True)
	print(b_func)
	suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
	# print(suggested_lowerbound)
	# print(suggested_upperbound)
	# alg = BeesAlgorithm(b_func,
	# 					suggested_lowerbound, suggested_upperbound,
	# 					**Ackley_bees_parameters)
	# alg.visualize_iteration_steps()
	alg = BeesAlgorithm(new_function,
						lower_bound, upper_bound,
						**Easom_bees_parameters)
	alg.visualize_iteration_steps()

def test_parallel_testing():
	print("Test of the parallel Bees Algorithm on the Ackley function..."),
	test_function=bf.Ackley(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = test_function.suggested_bounds()
	bees_parameters=Ackley_bees_parameters
	end_time, tester = test_on_function_parallel_tester(test_function,suggested_lowerbound,suggested_upperbound,bees_parameters,test_function.maximum().score)
	print("done.")
	print("Iterations 5-values summary "+str(tester.iterations5values))
	print("Scores 5-values summary "+str(tester.scores5values))
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	print('')
	print("Test of the parallel Bees Algorithm on the Griewank function..."),
	test_function=bf.Griewank(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = test_function.suggested_bounds()
	bees_parameters=Griewank_bees_parameters
	end_time, tester = test_on_function_parallel_tester(test_function,suggested_lowerbound,suggested_upperbound,bees_parameters,test_function.maximum().score)
	print("done.")
	print("Iterations 5-values summary "+str(tester.iterations5values))
	print("Scores 5-values summary "+str(tester.scores5values))
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))

def test_parallel_algorithm():
	print(">>> The Partial Parallel Bees Algorithm will now be tested <<<")
	start_time = time.time()
	print("Function Ackley (expected results: mean~128.82 std_dev~29.77)")
	b_func = bf.Ackley(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
	test_on_function(b_func, suggested_lowerbound, suggested_upperbound, Ackley_bees_parameters, b_func.maximum().score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	print('')
	print("Function Griewank (expected results: mean~2659.06 std_dev~1889.61)")
	b_func = bf.Griewank(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
	test_on_function(b_func, suggested_lowerbound, suggested_upperbound, Griewank_bees_parameters, b_func.maximum().score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	start_time = time.time()
	print('')
	print(">>> The Full Parallel Bees Algorithm will now be tested <<<")
	print("Function Ackley (expected results: mean~128.82 std_dev~29.77)")
	b_func = bf.Ackley(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
	test_on_function(b_func, suggested_lowerbound, suggested_upperbound, Ackley_bees_parameters, b_func.maximum().score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))
	print('')
	print("Function Griewank (expected results: mean~2659.06 std_dev~1889.61)")
	b_func = bf.Griewank(n_dimensions=10,opposite=True)
	suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
	test_on_function(b_func, suggested_lowerbound, suggested_upperbound, Griewank_bees_parameters, b_func.maximum().score)
	end_time = time.time() - start_time
	print("Elapsed time: "+str(time.strftime('%H:%M:%S',time.gmtime(end_time))))

if __name__=='__main__':
	# test_bees_algorithm()
	# test_parallel_testing()
	# test_parallel_algorithm()
	test_visualisation()