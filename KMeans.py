#	...................................................................................
'''
author: Palash Nandi.
'''
#	...................................................................................

import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#	import is done.


'''
read_dataset():
input: a path to the dataset
output: returns a MinMax normalized df 
'''
def read_dataset(path):
	df = pd.read_csv(path)
	
	total_columns = df.shape[1]
	default_col_names = np.arange(total_columns)
	df.columns = default_col_names
	
	df.reset_index(inplace=True)
	index_col = list(df['index'])
	df.drop(['index'], axis=1,inplace=True)
	# print(df)

	scaler_ob = MinMaxScaler()
	_ = scaler_ob.fit_transform(df)
	mod_df = pd.DataFrame(_)
	mod_df.insert(0,'index', index_col)

	ele_wise_cent_id = [None]* mod_df.shape[0]

	# print(mod_df)
	return mod_df


#	...................................................................................

'''
get_batches():
input_1: a dataframe object
input_2: batch_size [ optional | default size = 64 ]
output:  An array of batches
'''

def get_batches(df, batch_size=64):
    # print(f'df: {df.shape}, batch_size: {batch_size}')
    start_position = 0
    upto_position  = batch_size

    batch_list = []
    while  upto_position <= df.shape[0]:
        # print(f'upto_position: {upto_position}')
        try:
            batch_list.append(df.loc[start_position:upto_position-1, ].to_numpy())
        except Exception as e:
            # print(f'Exception in get_batches(_): {e}')
            batch_list.append(df.loc[upto_position, ].to_numpy())
        
        start_position = upto_position
        upto_position += batch_size

        # print(f'upto_position updated to: {upto_position}')

        if (df.shape[0]-upto_position) < batch_size:
            tail_amount = (df.shape[0]-upto_position)
            batch_list.append(df.tail(tail_amount).to_numpy())
            
            # print(f'Last {tail_amount} rows are added too.')
            break

    if len(batch_list) == 0:
        batch_list.append(df.to_numpy())
        
        
    # print(df)
    return batch_list

#	...................................................................................
'''
initialize_centroids():
input_1: number of centroids
input_2: the dataframe object
input_3: number of attributes [ optional | default == None ]
input_4: The type of initialization. [ optional | default == 'random' ]
output:  initialized centroids.
'''
def initialize_centroids(num_c, df, num_attr=None, mode='random'):
    centroids = None
    
    if mode == 'random':
        centroids = np.random.rand(num_c,num_attr)
        return centroids
    
    if mode == 'kmeans++':
        # print(f'{df}\n')
        
        random_pos = np.random.randint(low=0, high=df.shape[1])
        centroids = list([df.loc[random_pos]])
        centroid_positions = [random_pos]

        for num_centroid in range(1,num_c):
            max_dist = None
            selected_centroid = None
            selected_centroid_location = None
            temp_centroids = np.array(df.loc[random_pos])
            
            # print(f'For searching centroid_{num_centroid+1}: \n{np.array(centroids)}\n')
            for pos_i,d in df.iterrows():
                data = np.array(d)

                if pos_i in centroid_positions:
                    # print(f'Skipping as it is one of the selected centroid')
                    # print(f'{np.array(centroids)}')
                    # print(f'{data}')
                    # print('\n')
                    continue
                
                diff_ = ( data - temp_centroids)**2
                sum_ = np.sum(diff_)
                
                # print(f'\tFor pos_{pos_i} dist: {sum_} and max_dist: {max_dist}')
                if max_dist == None:
                    selected_centroid = data
                    max_dist = sum_
                    selected_centroid_location = pos_i
            
                if max_dist < sum_ and max_dist != None:
                    selected_centroid = data
                    max_dist = sum_
                    selected_centroid_location = pos_i

                # print(f'\t{pos_i} is selected with: {data}\n')
                    

            centroids.append(selected_centroid)
            centroid_positions.append(selected_centroid_location)
            
            print(f'Updated centroids:\n{np.array(centroids).shape} \n {np.array(centroids)}')
            # print(f'Updated centroids positions: \n {centroid_positions}')
            # print('\n\n\n')    
        
        return np.array(centroids)

    if mode == 'naive_shard':
        sum2id= {}
        id_list = []
        centroids = np.zeros((num_c,df.shape[1]))
        # print(f'zero centroids: {centroids.shape}')
        step_jump = math.ceil(df.shape[0]/num_c)
        index_list = list(df.index)
        
        df['row_wise_sum'] = np.sum(df, axis=1)
        id2sum = df.to_dict('dict')['row_wise_sum']
                
        for id_,sum_ in id2sum.items():
            try:
                _ = sum2id[sum_]
                _.append(id)
                sum2id[sum_] = _ 
            
            except:
                sum2id[sum_] = [id_]
                

        sum_list = list(sum2id.keys())
        sum_list.sort(reverse=False)
        
        for sum_i in sum_list:
            id_list += sum2id[sum_i]

        
        # print(sum2id)
        # print(f'{len(id_list)}: {id_list}')
        # print(f'step_jump: {step_jump}')

        centroid_processed = 0
        st_index = 0
        end_index = step_jump
        total_processed = 0
        df = df.drop(['row_wise_sum'], axis=1)
        
        # while total_processed < df.shape[0]:
        while len(id_list) > 0:    
            ids = id_list[st_index:end_index]
            id_list = id_list[end_index:]
            # print(f'\nitems in id_list: {len(id_list)}')

            ids = list(set(index_list).intersection(set(ids)))
            temp_df = df.loc[ids].copy(deep=True)
            abstract_centroid = np.sum(temp_df.to_numpy(), axis=0)/step_jump

            temp_df['sq_sum_distance'] = np.sum((temp_df.to_numpy() - abstract_centroid)**2, axis=1)
            min_val = min( list( temp_df['sq_sum_distance']))

            abstract_centroid = temp_df.loc[temp_df['sq_sum_distance'] == min_val, :]
            # print( abstract_centroid.shape)
            abstract_centroid.drop(['sq_sum_distance'], inplace=True, axis=1)
            abstract_centroid = np.array( abstract_centroid)
            # print( abstract_centroid)

            # # print(f'col names: {df.columns}')
            # # print(f'{len(ids)}: {ids}')
            # # print(temp_df)
            # # print(np.sum(temp_df.to_numpy(), axis=0))
            # # print(np.sum(temp_df[[0]].to_numpy(), axis=0))
            # print(abstract_centroid)
            # print(temp_df)
            # print(f'min_val: {min_val}')
            # print(_)
            
            centroids[centroid_processed] = abstract_centroid
            centroid_processed += 1
            total_processed += temp_df.shape[0]

            # print(f'centroids: {centroids}')
            # print(f'total_processed: {total_processed}')
            # print(f'centroids: {np.array(centroids).shape}')
            # print(f'{np.array(centroids)}')


            # break
        return np.array(centroids)


#	...................................................................................
'''
update_centroids():
input_1 : dictionary of data_id to assigned cluster_id
input_2 : dictionary of data_id to assigned cluster_id's intermidiate distance
input_3 : centroid wise cumulative distance i.e. inertia
input_4 : the datafframe object
input_5 : the type of update required [options: normal/from_dataset]
output:   updated centroids
'''

def update_centroids( id2c_id, id2c_id_distance, cumulative_centroid_values, df, mode='normal'):
    ele_wise_cent_id_counter = Counter(list(id2c_id.values()))
    # print(ele_wise_cent_id_counter)
    
    for i in ele_wise_cent_id_counter.keys():
            cumulative_centroid_values[i] /= ele_wise_cent_id_counter[i] 
    
    if str(mode).strip().lower() == 'normal':
        return cumulative_centroid_values
    
    if str(mode).strip().lower() == 'from_dataset':
        max_cluster = max(list(id2c_id.values()))
        c_id2ids = {}

        for pos_i, c_id in id2c_id.items():
            try:
                _ = c_id2ids[c_id]
                _.append(pos_i)
                c_id2ids[c_id] = _
            except:
                c_id2ids[c_id] = [pos_i]
        
        # print(id2c_id)
        # print( c_id2ids)
        # print(f'Total cluster: {min(list(c_id2ids.keys()))}-{max(list(c_id2ids.keys()))} vs expected_val: {cumulative_centroid_values.shape[0]}')

        # print(cumulative_centroid_values)
        for c_id, id_list in c_id2ids.items():
            data = df.loc[id_list,]
            data.reset_index(drop=True, inplace=True)
            # print(data)
            
            chosen_c_id = cumulative_centroid_values[c_id]

            sum_ = np.sum((chosen_c_id - data)**2, axis=1)
            min_sum = min(sum_)
            min_index = list(sum_).index(min_sum)
            
            # print(sum_)
            # print(min_index)

            chosen_centroid = data.loc[min_index,]
            # print(f'{np.array(chosen_centroid)} for {c_id}')

            cumulative_centroid_values[c_id] = chosen_centroid
            # print(cumulative_centroid_values)
            # break

        return cumulative_centroid_values
    
    return None
    
#	...................................................................................
'''
save_figure(...)
input_1: x
input_2.1: y1
input_2.2: y2
input_2.3: y3
input_3: dataset_name
input_4: image_path
input_5: batch_size
input_6: title
output:  it saves the image in system
'''
def save_figure_3(x,y1,y2,y3,dataset_name, batch_size, title,l1,l2,l3):
	image_path = '/home/palash/ML_GitHub/K_Means/images/'
	png_name = image_path + dataset_name + '_3_' + str(batch_size) + '_Scratch_vs_sklearn.png'
	svg_name = image_path + dataset_name + '_3_' + str(batch_size) + '_Scratch_vs_sklearn.svg'

	png_name = png_name.replace('.csv','')
	svg_name = svg_name.replace('.csv','')

	plt.plot(x, y1, 'r')
	plt.plot(x, y2, 'g')
	plt.plot(x, y3, 'm')

	plt.xlabel("cluster_size")
	plt.ylabel("cost")
	plt.title(title)
	plt.legend([l1,l2,l3])

	plt.savefig(png_name)
	plt.savefig(svg_name)

	plt.clf()



#	...................................................................................
    
'''
save_figure(...)
input_1: x
input_2.1: y1
input_2.2: y2
input_3: dataset_name
input_4: image_path
input_5: batch_size
input_6: title
input_7: legend_1
input_7: legend_2
output:  it saves the image in system
'''
def save_figure_2(x,y1,y2,dataset_name, batch_size, title, l1, l2):
	image_path = '/home/palash/ML_GitHub/K_Means/images/'
	png_name = image_path + dataset_name + '_2_' + str(batch_size) + title + '.png'
	svg_name = image_path + dataset_name + '_2_' + str(batch_size) + title + '.svg'

	png_name = png_name.replace('.csv','')
	svg_name = svg_name.replace('.csv','')
	png_name = png_name.replace('\n','')
	svg_name = svg_name.replace('\n','')

	plt.plot(x, y1, 'r')
	plt.plot(x, y2, 'g')
	# plt.plot(x, y3, 'm')

	plt.xlabel("cluster_size")
	plt.ylabel("cost")
	plt.title(title)
	plt.legend([l1,l2])

	plt.savefig(png_name)
	plt.savefig(svg_name)

	plt.clf()



#	...................................................................................
'''
save_figure(...)
input_1: x
input_2: y1
input_3: dataset_name
input_4: image_path
input_5: batch_size
input_6: title
input_7: legend_1
input_7: legend_2
output:  it saves the image in system
'''
def save_figure_1(x,y1,dataset_name, batch_size, title, l1):
	image_path = '/home/palash/ML_GitHub/K_Means/images/'
	png_name = image_path + dataset_name + '_1_' + str(batch_size) + title + '.png'
	svg_name = image_path + dataset_name + '_1_' + str(batch_size) + title + '.svg'

	png_name = png_name.replace('.csv','')
	svg_name = svg_name.replace('.csv','')
	png_name = png_name.replace('\n','')
	svg_name = svg_name.replace('\n','')

	plt.plot(x, y1)
	# plt.plot(x, y2, 'g')
	# plt.plot(x, y3, 'm')

	plt.xlabel("cluster_size")
	plt.ylabel("cost")
	plt.title(title)
	plt.legend([l1])

	plt.savefig(png_name)
	plt.savefig(svg_name)

	plt.clf()

#	...................................................................................

# 	global variables

total_epoch = 150
start_cluster = 2
maximum_cluster = 70
x_cluster = []
y_error = []

x_kmeans = []
error_kmeans = []
custom_centroids = {}
sklearn_centroids = {}
tolerance = 7
batch_size = 64

dataset_name = 'iris_dataset.csv'
dataset_path = '/home/palash/ML_GitHub/K_Means/' + dataset_name

mod_df = read_dataset(dataset_path)
batch_list = get_batches( df=mod_df, batch_size = 32)

for num_centroids in range(start_cluster, maximum_cluster+1):
	# Kmeans using library is started.
    loss_status = 'unaltered'
    kmeans = KMeans(n_clusters=num_centroids, random_state=0)
    kmeans.fit_transform(np.array(mod_df[mod_df.columns[1:]]))
    error_kmeans.append(kmeans.inertia_)
    x_kmeans.append(num_centroids)
    sklearn_centroids[num_centroids] = kmeans.cluster_centers_
    # print(sklearn_centroids)
    
    try:
        if error_kmeans[-1] < error_kmeans[-2]:
            loss_status = 'reduced'
        elif error_kmeans[-1] > error_kmeans[-2]:
            loss_status = 'increased'
        else:
            loss_status = 'unaltered' 
        print(f'For n: {num_centroids}, Error of k-means_sklearn: {error_kmeans[-3:]}: {loss_status}')
    
    except:
        pass 
    
    # Kmeans using library is finished.

    # Kmeans from scratch is started.

    # initialization is 'random'
    # centroids = initialize_centroids(num_c=num_centroids, num_attr=df.shape[1], df=mod_df, mode='random')
    
    # initialization is 'kmeans++'
    centroids = initialize_centroids(num_c=num_centroids, df=mod_df[mod_df.columns[1:]], mode='naive_shard')
    
    x_cluster.append(num_centroids)
    
    entropy_record = []
    unchanged_itr = 0
    for epoch_i in range(total_epoch):
        loss_status = 'unaltered'
        id2c_id = {}
        id2c_id_distance = {}
       
        entropy_sum = 0 # [None]* mod_df.shape[0]
        temp_centroids = np.full((centroids.shape[0], centroids.shape[1]),0.0)
        
        batch_entropy_record = []
        # calculate nearest centroid_id for each row
        for batch_i, data in enumerate(get_batches( df=mod_df, batch_size = batch_size)):
            for _, data in enumerate(data):
                pos_i = data[0]
                data = data[1:] 
                diff_ = data - centroids
                diff_2 = np.square(diff_)
                sum_ = np.sum(diff_2, axis=1) # np.sqrt( np.sum(diff_2, axis=1))
                min_sum = min(sum_)
                temp_cent_id = list(sum_).index(min_sum)
            
                id2c_id[pos_i] = temp_cent_id
                id2c_id_distance[pos_i] = min_sum
                entropy_sum += min_sum
                temp_centroids[temp_cent_id] += data 
        
            centroids = update_centroids( id2c_id=id2c_id, id2c_id_distance=id2c_id_distance, cumulative_centroid_values=temp_centroids, df=mod_df[mod_df.columns[1:]], mode='from_dataset')
            batch_entropy_record.append(entropy_sum)
            # break # 3
        
        # update new ele_wise_cent_id
        entropy_record.append( sum(batch_entropy_record)/len(batch_entropy_record))
        
        try:
            if entropy_record[-1] < entropy_record[-2]:
                loss_status = 'reduced'
                unchanged_itr = 0
            elif entropy_record[-1] > entropy_record[-2]:
                loss_status = 'increased'
                unchanged_itr = 0
            else:
                unchanged_itr += 1
                if unchanged_itr == tolerance:
                    break
        except:
            pass

        print(f'scratch_:For {num_centroids}.{epoch_i}, errors: {entropy_record[-3:]} => {loss_status}')

        # centroids = update_centroids( id2c_id=id2c_id, id2c_id_distance=id2c_id_distance, cumulative_centroid_values=temp_centroids, df=mod_df, mode='from_dataset')
        # break
    y_error.append(entropy_record[-1])
    custom_centroids[num_centroids] = centroids
    # break # 1

#	...................................................................................

#	...................................................................................
'''
Here we will reproduce another version of kmeans implemented from scratch by 
author 'Antonis Maronikolakis' for comparison purpose.

Github link: https://github.com/antmarakis/Machine-Learning/tree/master/Clustering/kMeans%20-%20Standard
GeekforGeeks link: https://www.geeksforgeeks.org/k-means-clustering-introduction/
'''
#	...................................................................................
def FindColMinMax(items):
    n = len(items[0]);
    minima = [pow(10,5) for i in range(n)];
    maxima = [pow(10,-5) for i in range(n)];
    
    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f];
            if (item[f] > maxima[f]):
                maxima[f] = item[f];
    
    return minima,maxima;

def UpdateMean(n,mean,item):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+item[i])/float(n);
        mean[i] = round(m, 5);
        
    return mean;

def EuclideanDistance(x, y):
    # print(f'Received,\nX:{x}, \nY:{y}')
    S = 0; # The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i]-y[i], 2)
        # print(f'\t_diff: {math.pow(x[i]-y[i], 1)}')
        # print(f'\t_diff^2: {math.pow(x[i]-y[i], 2)}')    
    
    # print(f'\tsum: {S}')
    # print(f'\treturning: {math.sqrt(S)}')
    
    #The square root of the sum
    return math.sqrt(S)

def Classify(means,item):
    minimum = pow(10,7);
    index = -1;
    
    for i in range(len(means)):
        # print(f'For mean: {means[i]}')
        dis = EuclideanDistance(item, means[i]);
        
        if (dis < minimum):
            minimum = dis;
            index = i;
    return index;

def calculate_inertia(items, belongsTo, means):
    _inertia = 0
    # print(f'items: {items}')
    # print(f'belongsTo: {belongsTo}')
    # print(f'means: {means}')

    for i, data_i in enumerate(items):
        _centroid_id = belongsTo[i]
        _centroid = means[_centroid_id]

        _sum = 0
        for j, val_j in enumerate(data_i):
            _diff = abs(_centroid[j]-val_j)
            _sum += pow(_diff,2)
        
        _inertia += _sum
        # print(_inertia)
    
    return _inertia
    
def InitializeMeans(items, k, cMin, cMax):
    f = len(items[0]); # number of features
    means = [[0 for i in range(f)] for j in range(k)];
    
    for mean in means:
        for i in range(len(mean)):
            mean[i] = random.uniform(cMin[i]+1, cMax[i]-1);
    
    return means;

def FindClusters(means,items):
	clusters = [[] for i in range(len(means))]; # Init clusters
	
	for item in items:
		# Classify item into a cluster
		index = Classify(means,item);
		# Add item to cluster
		clusters[index].append(item);

	# return clusters;

tolerance = 7
belongsTo = []
inertia_records = []
epoch2inertia = {}

def CalculateMeans(k,items,maxIterations=100000):
    # Find the minima and maxima for columns
    cMin, cMax = FindColMinMax(items);
	
    # return cMin, cMax
    
    # Initialize means at random points
    means = InitializeMeans(items,k,cMin,cMax);
    # print(f'random means: {means}')
    # return None 
    
    # Initialize clusters, the array to hold the number of items in a class
    clusterSizes= [0 for i in range(len(means))];
    # print(f'clusterSizes: {clusterSizes}')
    # return None
    
    # An array to hold the cluster an item is in
    belongsTo = [0 for i in range(len(items))];
    _inertia = None
    # Calculate means
    for e in range(maxIterations):
        for i in range(len(items)):
            item = items[i];
            # Classify item into a cluster and update the
            # corresponding means.	
            index = Classify(means,item);
            clusterSizes[index] += 1;
            cSize = clusterSizes[index];
            means[index] = UpdateMean(cSize,means[index],item);
            
            # Item changed cluster
            if(index != belongsTo[i]):
                noChange = False;
            belongsTo[i] = index;

            
        

        _inertia = calculate_inertia(items, belongsTo, means)
        inertia_records.append(_inertia)
        epoch2inertia[e] = _inertia

        loss_status = 'unaltered'
        
        try:
            if inertia_records[-1] < inertia_records[-2]:
                loss_status = 'reduced'
                unchanged_itr = 0
            elif inertia_records[-1] > inertia_records[-2]:
                loss_status = 'increased'
                unchanged_itr = 0
            else:
                unchanged_itr += 1
                if unchanged_itr == tolerance:
                    break
        except:
            pass

        print(f'g4gks_:For {k}.{e}, errors: {inertia_records[-3:]} => {loss_status}')


    return means, _inertia;


# min_centroids = 2
# max_centroids = 70

items = []
num_cent_2_inertia = {}

for i in mod_df.to_numpy():
	items.append(list(i))

for num_centroids in range(start_cluster, maximum_cluster+1):
    _, _inertia= CalculateMeans(k=num_centroids, items=items, maxIterations= 150)
    num_cent_2_inertia[num_centroids] = _inertia

# ....
''' save the image. '''
# ....
gfg_vals = list(num_cent_2_inertia.values())
# print(f'x_cluster: {len(x_cluster)}')
# print(f'y_error: {len(y_error)}')
# print(f'gfg_vals: {len(gfg_vals)}')


title = "k-means by sklearn vs scratch vs GeeksforGeeks \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_3(x_cluster, y_error, error_kmeans, gfg_vals, dataset_name, batch_size, title, l1,l2,l3)

title = "k-means by sklearn vs scratch\n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_2(x_cluster, y_error, error_kmeans, dataset_name, batch_size, title, l1,l2)


title = "k-means by sklearn vs GeeksforGeeks \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_2(x_cluster, error_kmeans, gfg_vals, dataset_name, batch_size, title, l2,l3)

title = "k-means from scratch vs GeeksforGeeks \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_2(x_cluster, y_error, gfg_vals, dataset_name, batch_size, title, l1,l3)


title = "k-means from scratch \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_1(x_cluster, y_error, dataset_name, batch_size, title,l1)

title = "k-means by sklearn \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_1(x_cluster, error_kmeans, dataset_name, batch_size, title, l2)

title = "k-means from GeeksforGeeks \n\n\n"
l1 = 'k-means_from_scratch'
l2 = 'k-means_from_sklearn' 
l3 = 'k-means_from_geeksforgeeks'

save_figure_1(x_cluster, gfg_vals, dataset_name, batch_size, title,l3)



