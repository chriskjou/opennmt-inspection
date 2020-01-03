# import time

# def basic_func(x):
# 	if x == 0:
# 		return 'zero'
# 	elif x%2 == 0:
# 		return 'even'
# 	else:
# 		return 'odd'
	
# starttime = time.time()
# for i in range(0,10):
# 	y = i*i
# 	# time.sleep(2)
# 	print('{} squared results in a/an {} number'.format(i, basic_func(y)))
	
# print('That took {} seconds'.format(time.time() - starttime))


### EXAMPLE

import multiprocessing as mp
import time

list_a = [[1, 2, 3], [5, 6, 7, 8], [10, 11, 12], [20, 21]]
list_b = [[2, 3, 4, 5], [6, 9, 10], [11, 12, 13, 14], [21, 24, 25]]

def get_commons(list_1, list_2):
    return list(set(list_1).intersection(list_2))

print(mp.cpu_count())
start = time.time()
pool = mp.Pool(mp.cpu_count())
results = [pool.apply(get_commons, args=(l1, l2)) for l1, l2 in zip(list_a, list_b)]
pool.close()    
end = time.time()
# print(results[:10])
print('That took {} seconds'.format(end - start))

###

# import time
# import multiprocessing 

# def basic_func(x):
#     if x == 0:
#         return 'zero'
#     elif x%2 == 0:
#         return 'even'
#     else:
#         return 'odd'

# def multiprocessing_func(x):
#     y = x*x
#     time.sleep(2)
#     print('{} squared results in a/an {} number'.format(x, basic_func(y)))
    
# if __name__ == '__main__':
#     starttime = time.time()
#     processes = []
#     for i in range(0,10):
#         p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
#         processes.append(p)
#         p.start()
        
#     for process in processes:
#         process.join()
        
#     print('That took {} seconds'.format(time.time() - starttime))

