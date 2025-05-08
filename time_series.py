from myutils import Utils
import numpy as np
from baseline.DeepSAD.src.run import DeepSAD
import os
import argparse
from math import ceil
from myutils import Utils
from scipy import spatial
import pandas as pd
from baseline.Supervised import supervised
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('--la', type=float, default=0.1,
                    help='la')
parser.add_argument('--ratio', type=float, default=1.0,
                    help='ratio')
parser.add_argument('--mixup_alpha', type=float, default=0.2,
                    help='ratio')
parser.add_argument('--mixup_beta', type=float, default=0.2,
                    help='ratio')
parser.add_argument('--cutout_alpha', type=float, default=0.1,
                    help='ratio')
parser.add_argument('--cutout_beta', type=float, default=0.3,
                    help='ratio')
parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument("--method", type=str, default='nng_mix')
parser.add_argument("--alg", type=str, default='DeepSAD')
parser.add_argument('--use_anomaly_only', action='store_true')
parser.add_argument('--use_uniform', action='store_true')
parser.add_argument('--nn_k', type=int, default=10,
                    help='nn_k')
parser.add_argument('--nn_mix_gaussian', action='store_true')
parser.add_argument('--nn_mix_gaussian_std', type=float, default=1.0,
                    help='nn_mix_gaussian_std')
parser.add_argument('--adjust_nn_k', action='store_true')
parser.add_argument('--adjust_nn_k_n', type=int, default=2,
                    help='adjust_nn_k_n')
parser.add_argument("--appen", type=str, default='')
parser.add_argument('--gaussian_var', type=float, default=1.0,
                    help='gaussian_var')
parser.add_argument('--adjust_nn_k_anomaly', action='store_true')
parser.add_argument('--adjust_nn_k_n_anomaly', type=float, default=0.3,
                    help='adjust_nn_k_n_anomaly')
parser.add_argument('--nn_k_anomaly', type=int, default=10,
                    help='nn_k_anomaly')
args = parser.parse_args()

seed = args.seed
utils = Utils()
utils.set_seed(args.seed)

# Load the dataset
dataset = pd.read_csv("train.csv")

# Sort the ascending order of timestamp but also keep the building_id ascending
dataset = dataset.sort_values(by=['building_id', 'timestamp'], ascending=[True, True])

# Perform Median Imputation to handle missing values
def median_imputation(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Find all unique building IDs
    unique_building_ids = df_copy['building_id'].unique()
    
    for i in unique_building_ids:
        # For each building ID, find the median of the 'meter_reading' column
        median_value = df_copy[df_copy['building_id'] == i]['meter_reading'].median()
        
        # Fill missing values in the 'meter_reading' column with the median value
        df_copy.loc[(df_copy['building_id'] == i) & (df_copy['meter_reading'].isnull()), 'meter_reading'] = median_value
    
    return df_copy

# Apply median imputation to the dataset
dataset = median_imputation(dataset)

# Extract the unique building IDs
unique_building_ids = dataset['building_id'].unique()

class DataGenerator():
    def __init__(self, 
                 seed: int = 42, test_size:float=0.3, generate_duplicates:bool=True, n_samples_treshold:int=1000):
    
        self.seed = seed
        self.test_size = test_size
        self.generate_duplicates = generate_duplicates
        self.n_samples_treshold = n_samples_treshold
        self.utils = Utils()
    
    def data_generator(self, dataset, minmax = True, 
                    la = None, 
                    at_least_one_labeled=False,
                    realistic_synthetic_mode=None, alpha:int=5, percentage:float=0.1,
                    noise_type=None, duplicate_times:int=2, noise_ratio:float=0.05):
        '''
            la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
            at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
            '''
        self.utils.set_seed(self.seed)
        
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1].copy()
        
        if type(la) == float:
            if at_least_one_labeled:
                # n_labeled_anomalies = ceil(sum(y) * (1 - self.test_size) * la)
                n_labeled_anomalies = ceil(sum(y) * la)
            else:
                n_labeled_anomalies = int(sum(y) * (1 - self.test_size) * la)
        elif type(la) == int:
            n_labeled_anomalies = la
        else:
            raise NotImplementedError
    

        # if len(y) < self.n_samples_treshold and self.generate_duplicates:
        #     self.utils.set_seed(self.seed)
        #     idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
        #     X = X.iloc[idx_duplicate]
        #     y = y.iloc[idx_duplicate]
        
        # if len(y) > 10000:
        #     # How to choose for time series data?
        #     self.utils.set_seed(self.seed)
        #     idx_duplicate = np.random.choice(np.arange(len(y)), 10000, replace=True)
        #     X = X.iloc[idx_duplicate]
        #     y = y.iloc[idx_duplicate]
        
        if noise_type is None:
            pass
        else:
            raise NotImplementedError
        
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle = True, stratify=y)
        
        # if minmax:
        #     scaler = MinMaxScaler().fit(X_train)
        #     X_train = scaler.transform(X_train)
        #     X_test = scaler.transform(X_test)

        
        # idx_normal = np.where(y_train == 0)[0]
        # idx_anomaly = np.where(y_train == 1)[0]
        
        # idx_normal = np.where(y == 0)[0] 
        # idx_anomaly = np.where(y == 1)[0]
        
        # if type(la) == float:
        #     if at_least_one_labeled:
        #         idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
        #     else:
        #         idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)
        # elif type(la) == int:
        #     if la > len(idx_anomaly):
        #         raise AssertionError(f'the number of labeled anomalies are greater than the total anomalies: {len(idx_anomaly)} !')
        #     else:
        #         idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        # else:
        #     raise NotImplementedError

        # idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        
        # idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        # del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        # y_train[idx_unlabeled] = 0
        # y_train[idx_labeled_anomaly] = 1
        
        # y.loc[idx_unlabeled] = 0
        # y.loc[idx_labeled_anomaly] = 1

        # return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}, scaler
        return X, y
        
log_name = "log_%s_%s_%s_%s_%s"%('train', args.alg, args.method, str(args.la), str(args.ratio))
if args.method == 'mixup' or args.method == 'nng_mix':
    log_name = log_name + '_%s_%s'%(str(args.mixup_alpha), str(args.mixup_beta))
if args.method == 'cutout' or args.method == 'cutmix':
    log_name = log_name + '_%s_%s'%(str(args.cutout_alpha), str(args.cutout_beta))
if args.method == 'nng_mix' and (not args.adjust_nn_k):
    log_name = log_name + '_k_%s'%(str(args.nn_k))
if (args.method == 'nng_mix') and (not args.adjust_nn_k_anomaly):
    log_name = log_name + '_k_anomaly_%s'%(str(args.nn_k_anomaly))
if args.method == 'gaussian_noise':
    log_name = log_name + '_%s'%(str(args.gaussian_var))
if args.adjust_nn_k:
    log_name = log_name + '_adjust_nn_k_%s'%(str(args.adjust_nn_k_n))
if args.adjust_nn_k_anomaly:
    log_name = log_name + '_adjust_nn_k_anomaly_%s'%(str(args.adjust_nn_k_n_anomaly))
if args.nn_mix_gaussian:
    log_name = log_name + '_gaussian_std_%s'%(str(args.nn_mix_gaussian_std))
if args.use_uniform:
    log_name = log_name + '_use_uniform'
if args.use_anomaly_only:
    log_name = log_name + '_use_anomaly_only'

log_name = log_name + '_seed_%s'%(str(args.seed)) 

if args.appen:
    log_name = log_name + '_' + args.appen

log_name = log_name + '.csv'
base_path = "logs/"
log_path = base_path + log_name

dataset_name = "LEAD_DATASET"

num_times = [0, 1, 5, 10]


with open(log_path, 'w') as f:
    f.write("{},{},{},{},{},{}\n".format('Building Id', 'Original_Anomaly', 0, 1, 5, 10))
    f.flush()

    for i in unique_building_ids:
        building_dataset = dataset[dataset['building_id'] == i]
        building_dataset = building_dataset.drop(columns=['building_id', 'timestamp'])
        building_dataset = building_dataset.reset_index(drop=True)
        nu = 0
        
        # Calculate original anomaly ratio
        original_anomaly_ratio = building_dataset['anomaly'].sum() / len(building_dataset)
        
        num_anomaly_list = []
        
        for num in num_times:
            added_synthetic_data = set()
            data_gen = DataGenerator(seed = args.seed)
            X,y = data_gen.data_generator(building_dataset, la=args.la, at_least_one_labeled=True)
            
            # X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
            
            # anomaly_data = X_train[y_train == 1]
            # unlabeled_data = X_train[y_train == 0]
            
            anomaly_data = X[y == 1]
            unlabeled_data = X[y == 0].copy()
            unlabeled_indices = np.where(y == 0)[0]
            
            # if args.ratio != 1.0:
            #     idx_choose_anomaly = np.random.choice(anomaly_data.shape[0], ceil(args.ratio * anomaly_data.shape[0]), replace=False)
            #     anomaly_data = anomaly_data[idx_choose_anomaly]
            
            gen_anomaly_files_n = anomaly_data.shape[0] * num

            
            if args.method == 'nng_mix':
                dim = anomaly_data.shape[1]
                tree2 = spatial.KDTree(anomaly_data)
                tree = spatial.KDTree(unlabeled_data)
                added_synthetic_data = set()
                
                for j in range(gen_anomaly_files_n):
                    if np.random.uniform(0,1) > 0.5:
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        if args.adjust_nn_k:
                            dis, ind = tree.query(anomaly_data.iloc[index1], k=anomaly_data.shape[0]*args.adjust_nn_k_n)
                        else:
                            dis, ind = tree.query(anomaly_data.iloc[index1], k=args.nn_k)
                        index2 = np.random.choice(ind[0])

                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)
                    
                        if args.nn_mix_gaussian:
                            gaussian_noise1 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            gaussian_noise2 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            anomaly_data_sample = (
                            lam * (gaussian_noise1 + anomaly_data.iloc[index1]) + 
                            (1 - lam) * (gaussian_noise2 + unlabeled_data.iloc[index2])
                            )
                        else:
                            anomaly_data_sample = (lam * anomaly_data[index1] + (1 - lam) * unlabeled_data[index2])
                    
                    else:
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        if args.adjust_nn_k_anomaly:
                            query_k = int(anomaly_data.shape[0]*args.adjust_nn_k_n_anomaly)
                        else:
                            query_k = args.nn_k_anomaly
                        query_k = max(query_k, 1)
                        query_k = min(query_k, anomaly_data.shape[0])

                        dis, ind = tree2.query(anomaly_data.iloc[index1], k=query_k)
                    
                        index2 = np.random.choice(ind[0])

                        if ind.shape[1] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(ind[0])

                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)

                        if args.nn_mix_gaussian:
                            gaussian_noise1 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            gaussian_noise2 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            anomaly_data_sample = (
                            lam * (gaussian_noise1 + anomaly_data.iloc[index1]) + 
                            (1 - lam) * (gaussian_noise2 + anomaly_data.iloc[index2])
                            )
                        else:
                            anomaly_data_sample = (lam * anomaly_data[index2] + (1 - lam) * anomaly_data[index1])
                    
                    # Randomly sample an unlabeled data point
                    unlabeled_idx = np.random.choice(unlabeled_data.shape[0], 1)[0]
                    
                    # This is to ensure that the generated synthetic data is not replaced
                    while unlabeled_idx in added_synthetic_data:
                        unlabeled_idx = np.random.choice(unlabeled_data.shape[0], 1)[0]
                    # Replace the unlabeled data point with the generated anomaly data
                    unlabeled_data.iloc[unlabeled_idx] = anomaly_data_sample
                    
                    # Add the sample to added_synthetic_data set to avoid replacing it again
                    added_synthetic_data.add(unlabeled_idx)
                    
                    # Doubt: whether to change the label of the generated synthetic data to 1
                    y[unlabeled_indices[unlabeled_idx]] = 1
                    
            # X_train = np.vstack([unlabeled_data, anomaly_data])
            # anomaly_labels = np.ones((anomaly_data.shape[0], 1))
            # unlabeled_labels = np.zeros((unlabeled_data.shape[0], 1))
            
            # y_train = np.concatenate([unlabeled_labels, anomaly_labels], axis=0)
            
            # if args.alg == 'DeepSAD':
            #     model = DeepSAD(seed=args.seed)
            # elif args.alg == 'MLP':
            #     model = supervised(seed = args.seed, model_name = 'MLP')
            
            # model.fit(X_train = X_train, y_train = y_train[:, 0])
            # score = model.predict_score(X_test)
            
            # utils = Utils()
            # result = utils.metric(y_true= y_test, y_score=score)
            
            # results[nu] = result['aucroc']
            # nu = nu + 1
            print(unlabeled_data)
            # X = np.vstack([unlabeled_data, anomaly_data])
            X.loc[y==0,'meter_reading']=unlabeled_data.values.reshape(-1)   
            X.loc[y==1,'meter_reading']=anomaly_data.values.reshape(-1)

            
            # Calculate ratio of anomalies in changed dataset
            new_anomaly_ratio = y.sum() / len(y)
            num_anomaly_list.append(new_anomaly_ratio)
            
            # Write the data out
        
        f.write("{},{},{},{},{},{}\n".format(i, original_anomaly_ratio, num_anomaly_list[0], num_anomaly_list[1], num_anomaly_list[2], num_anomaly_list[3]))
        f.flush()