import os
import os.path as path
import numpy as np
from glob import glob
import random
import pandas as pd
import torch
import torch.nn as nn
#====================================================================================================================================#

def count_yield(df):

    Groups = df.groupby(by=["cultivar"])
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    dict_ = {}
    for state, frame in Groups:        
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        
        dict_[state] = count_list, count_sum
    
    return dict_

def data_weight_sampler_eval(save_dir):


    train_df = pd.read_csv(os.path.join(save_dir, 'coords') +'/train.csv', index_col=0) 
    train_df.reset_index(inplace = True, drop = True)

    val_df = pd.read_csv(os.path.join(save_dir, 'coords') +'/val.csv', index_col=0)
    val_df.reset_index(inplace = True, drop = True)

    test_df = pd.read_csv(os.path.join(save_dir, 'coords') +'/test.csv', index_col=0)
    test_df.reset_index(inplace = True, drop = True)
    
    
    train_weights = train_df['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = val_df['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    

    test_weights = test_df['NormWeight'].to_numpy() 
    test_weights = torch.DoubleTensor(test_weights)
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights)) 
    
    
    return train_sampler, val_sampler, test_sampler


def make_weights_for_yield_ranges(df):
    
    dict_ = count_yield(df)
    
    weight = []#np.zeros((len(df))) 
    
    for idx, row in df.iterrows():  
        patch_cultivar = row['cultivar']
        patch_mean     = row['patch_mean'] 
        #yield_bin = get_range_bins(patch_mean)
        

        get_patch_count = dict_[patch_cultivar][0][patch_mean]
        get_cultivar_sum = dict_[patch_cultivar][1]
        row_weight = get_patch_count / get_cultivar_sum
        
        #weight[idx] = row_weight
        weight.append(row_weight)
    weight = np.array(weight)
        
    return weight
     
def make_norm_weight(df):
    
    weights = make_weights_for_yield_ranges(df)
    df['weight'] = weights
    
    list_sum = df.groupby(by=["cultivar"])['weight'].transform('sum')
    NormWeights = df['weight']/list_sum
    df['NormWeight'] = NormWeights
    
    return df



def df_weight_sampler(df):
    
    Groups = df.groupby(by=["cultivar"])
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    dict_ = {}
    for state, frame in Groups:        
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        
        dict_[state] = count_list, count_sum

    weight = []#np.zeros((len(df))) 
    
    for idx, row in df.iterrows():  
        patch_cultivar = row['cultivar']
        patch_mean = row['patch_mean']     

        get_patch_count = dict_[patch_cultivar][0][patch_mean]
        get_cultivar_sum = dict_[patch_cultivar][1]
        row_weight = get_patch_count / get_cultivar_sum
        weight.append(row_weight)
        
    weight = np.array(weight)
    df['weight'] = weight
    list_sum = df.groupby(by=["cultivar"])['weight'].transform('sum')
    NormWeights = df['weight']/list_sum
    df['NormWeight'] = NormWeights
    
    return df


def yield_weight_sampler(train, val, test, save_dir):
    
    tr_df   = df_weight_sampler(train)
    tr_df.reset_index(inplace = True, drop = True)
    tr_df.to_csv(os.path.join(save_dir,'train.csv'))
    
    val_df  = df_weight_sampler(val)
    val_df.reset_index(inplace = True, drop = True)
    val_df.to_csv(os.path.join(save_dir,'val.csv'))
    
    
    test_df = df_weight_sampler(test)
    test_df.reset_index(inplace = True, drop = True)
    test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    
    return tr_df, val_df, test_df
    


def data_weight_sampler(train, val, test, save_dir):
    
    save_dir = os.path.join(save_dir, 'coords')
    train_df, val_df, test_df = yield_weight_sampler(train, val, test, save_dir)
    
    
    train_weights = train_df['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = val_df['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    

    test_weights = test_df['NormWeight'].to_numpy() 
    test_weights = torch.DoubleTensor(test_weights)
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights)) 
    
    
    return train_sampler, val_sampler, test_sampler


def wn(df):

    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    count_list = df['patch_mean'].value_counts(bins=bins, sort=False)
    count_sum = np.sum(count_list)

    weight = []

    for idx, row in df.iterrows():  
        patch_cultivar = row['cultivar']
        patch_mean     = row['patch_mean'] 
        #yield_bin = get_range_bins(patch_mean)


        get_patch_count = count_list[patch_mean]
        row_weight = get_patch_count / count_sum

        #weight[idx] = row_weight
        weight.append(row_weight)
    weight = np.array(weight)

    df['w'] = weight

    list_sum = df['w'].sum()
    NormWeights = df['w']/list_sum
    df['NW'] = NormWeights
    
    return df



class ReadData(object):
    def __init__(self, npy_dir, csv_dir, category = None, patch_size = None, in_channels = None, spatial_resolution = None, run_status = None):

        self.npy_dir      = npy_dir
        self.csv_dir      = csv_dir
        self.wsize        = patch_size
        self.in_channels  = in_channels
        self.spatial_resolution = spatial_resolution


        if category    == 'train': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/train.csv', index_col=0) 
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'val': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/val.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'test': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/test.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        
        self.images = sorted(glob(os.path.join(self.npy_dir , 'imgs') +'/*.npy'))
        self.labels = sorted(glob(os.path.join(self.npy_dir , 'labels') +'/*.npy'))
        
        if self.spatial_resolution == 1: 
            self.number_of_random_value = 100
            self.depth    = 15
            self.range    = 6

            if run_status == 'train':
                self.random_vector1   = np.random.randint(1,self.number_of_random_value, self.number_of_random_value)
                np.save(self.csv_dir + '/RVs/RV1.npy', self.random_vector1)
                self.random_vector2   = np.random.randint(1, self.number_of_random_value, self.number_of_random_value)
                np.save(self.csv_dir + '/RVs/RV2.npy', self.random_vector2)
                self.random_vector3   = np.random.randint(1, self.number_of_random_value, self.number_of_random_value)
                np.save(self.csv_dir + '/RVs/RV3.npy', self.random_vector3)
                self.random_vector4   = np.random.randint(1, self.number_of_random_value, self.number_of_random_value)
                np.save(self.csv_dir + '/RVs/RV4.npy', self.random_vector4)

            elif run_status == 'eval': 
                self.random_vector1 = np.load(self.csv_dir + '/RVs/RV1.npy', allow_pickle=True)
                self.random_vector2 = np.load(self.csv_dir + '/RVs/RV2.npy', allow_pickle=True)
                self.random_vector3 = np.load(self.csv_dir + '/RVs/RV3.npy', allow_pickle=True)
                self.random_vector4 = np.load(self.csv_dir + '/RVs/RV4.npy', allow_pickle=True)


        

    def __getitem__(self, idx):

        xcoord      = self.NewDf.loc[idx]['X'] 
        ycoord      = self.NewDf.loc[idx]['Y'] 
        block_id    = self.NewDf.loc[idx]['block']
        cultivar    = self.NewDf.loc[idx]['cultivar']
        cultivar_id = self.NewDf.loc[idx]['cultivar_id']
        rw_id       = self.NewDf.loc[idx]['row']
        sp_id       = self.NewDf.loc[idx]['space']
        t_id        = self.NewDf.loc[idx]['trellis_id']
        
        WithinBlockMean    = self.NewDf.loc[idx]['win_block_mean']
        WithinBlockStd     = self.NewDf.loc[idx]['win_block_std']
        WithinCultivarMean = self.NewDf.loc[idx]['win_cultivar_mean']
        WithinCultivarStd  = self.NewDf.loc[idx]['win_cultivar_std']
        
        
        img_path   = self.NewDf.loc[idx]['IMG_PATH']
        label_path = self.NewDf.loc[idx]['LABEL_PATH']

        image = self.crop_gen(img_path, xcoord, ycoord) 
        mask  = self.crop_gen(label_path, xcoord, ycoord) 

        CulMatrix = self.patch_cultivar_matrix(cultivar_id)  
        RWMatrix  = self.patch_rw_matrix(rw_id) 
        SpMatrix  = self.patch_sp_matrix(sp_id)
        TMatrix   = self.patch_tid_matrix(t_id)  
        EmbMat    = np.concatenate([CulMatrix, RWMatrix, SpMatrix, TMatrix], axis = 0)

            
            
        image = np.swapaxes(image, -1, 0)    
        
        if self.in_channels == 5: 
            bc_mean_mtx = self.add_input_within_bc_mean(WithinBlockMean)
            image = np.concatenate([image, bc_mean_mtx], axis = 0)


        if self.in_channels == 6: 
            bc_mean_mtx = self.add_input_within_bc_mean(WithinBlockMean)
            block_timeseries_encode = self.time_series_encoding(block_id)
            image = np.concatenate([image, bc_mean_mtx, block_timeseries_encode], axis = 0)
        
        
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image / 255.


        mask  = np.swapaxes(mask, -1, 0)
        mask  = torch.as_tensor(mask)
         
        sample = {"image": image, "mask": mask, "EmbMatrix": EmbMat, "block": block_id, "cultivar": cultivar, "X": xcoord, "Y": ycoord, "win_block_mean":WithinBlockMean, 
                    "win_block_std": WithinBlockStd, "win_cultivar_mean": WithinCultivarMean, "win_cultivar_std":WithinCultivarStd}
             
        return sample

    def __len__(self):
        return len(self.NewDf)
       
    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
  
    
    def patch_cultivar_matrix(self, cul_id):

        if self.spatial_resolution == 1: 
            zeros_matrix = np.zeros(self.number_of_random_value)
            idxs = self.random_vector1[(cul_id-1)*self.range:cul_id*self.range]
            zeros_matrix[idxs] = 1
            cultivar_matrix = zeros_matrix.reshape(1,int(self.wsize/8),int(self.wsize/8))

        elif self.spatial_resolution == 10: 
            zeros_matrix       = np.full(4, (1/cul_id))
            cultivar_matrix    = zeros_matrix.reshape(1, int(self.wsize/8), int(self.wsize/8))
        
        return cultivar_matrix
    
    def patch_rw_matrix(self, rw):

        if self.spatial_resolution == 1: 
            zeros_matrix = np.zeros(self.number_of_random_value) 
            idxs = self.random_vector2[(rw-1)*self.range:rw*self.range]
            zeros_matrix[idxs] = 1
            rw_matrix = zeros_matrix.reshape(1,int(self.wsize/8),int(self.wsize/8))

        elif self.spatial_resolution == 10: 
            zeros_matrix       = np.full(4, (1/rw))
            rw_matrix          = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return rw_matrix    
    
    def patch_sp_matrix(self, sp):

        if self.spatial_resolution == 1: 
            zeros_matrix = np.zeros(self.number_of_random_value) 
            idxs = self.random_vector3[(sp-1)*self.range:sp*self.range]
            zeros_matrix[idxs] = 1
            sp_matrix = zeros_matrix.reshape(1,int(self.wsize/8),int(self.wsize/8))

        elif self.spatial_resolution ==10: 
            zeros_matrix       = np.full(4, (1/sp))
            sp_matrix          = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return sp_matrix
        
    def patch_tid_matrix(self, tid):

        if self.spatial_resolution == 1: 
            zeros_matrix = np.zeros(self.number_of_random_value) 
            idxs = self.random_vector4[(tid-1)*self.range:tid*self.range]
            zeros_matrix[idxs] = 1
            tid_matrix = zeros_matrix.reshape(1,int(self.wsize/8),int(self.wsize/8))

        elif self.spatial_resolution ==10: 
            zeros_matrix = np.full(4, (1/tid))
            tid_matrix = zeros_matrix.reshape(1,int(self.wsize/8), int(self.wsize/8))
        
        return tid_matrix
    
    def add_input_within_bc_mean(self, bloks_mean):
        
        fill_matrix_bmean = np.full((1, self.wsize, self.wsize, 15), bloks_mean) 
        
        return fill_matrix_bmean
    
    def time_series_encoding(self, block_id):
        timeseries = None

        name_split = os.path.split(str(block_id))[-1]
        year       = name_split[-4:]

        if year == '2016': 
            days = [91, 95, 107, 116, 135, 136, 141, 150, 161, 166, 171, 176, 182, 195, 202]
        elif year =='2017':
            days = [90, 109, 119, 134, 140, 156, 169, 176, 179, 184, 190, 192, 195, 197, 202]
        elif year =='2018':
            days = [91, 105, 112, 115, 121, 131, 135, 142, 152, 155, 165, 175, 185, 191, 194]
        elif year =='2019':
            days = [91, 101, 112, 115, 121, 124, 131, 145, 152, 155, 164, 171, 181, 192, 202]
        
        for day in days: 
            this_week_matrix = np.full((1, self.wsize, self.wsize), 1 - np.sin(day/(365*np.pi))) 
            this_week_matrix = np.expand_dims(this_week_matrix, axis = -1)
            if timeseries is None:
                timeseries = this_week_matrix
            else:
                timeseries = np.concatenate([timeseries, this_week_matrix], axis = -1)

        return timeseries  
    



class ReadData_S1(object):
    def __init__(self, npy_dir, csv_dir, category = None, window_size = None):
        self.npy_dir      = npy_dir
        self.csv_dir      = csv_dir
        self.wsize = window_size


         
        if category    == 'train': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/train.csv', index_col=0) 
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'val': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/val.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        elif category  == 'test': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/test.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)
        
        self.images = sorted(glob(os.path.join(self.npy_dir, 'imgs') +'/*.npy'))
        self.labels = sorted(glob(os.path.join(self.npy_dir, 'labels') +'/*.npy'))
    

    def __getitem__(self, idx):

        xcoord      = self.NewDf.loc[idx]['X'] 
        ycoord      = self.NewDf.loc[idx]['Y'] 
        block_id    = self.NewDf.loc[idx]['block']
        cultivar    = self.NewDf.loc[idx]['cultivar']
        
        
        img_path   = self.NewDf.loc[idx]['IMG_PATH']
        label_path = self.NewDf.loc[idx]['LABEL_PATH']

        image = self.crop_gen(img_path, xcoord, ycoord) 
        mask  = self.crop_gen(label_path, xcoord, ycoord) 
        
        image = np.swapaxes(image, -1, 0)   
        #block_timeseries_encode = self.time_series_encoding(block_id)
        #image = np.concatenate([image, block_timeseries_encode], axis = 0)
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image / 255. 
        

        mask  = np.swapaxes(mask, -1, 0)
        mask  = torch.as_tensor(mask)

        
        sample = {"image": image, "mask": mask, "block": block_id, "cultivar": cultivar, "X": xcoord, "Y": ycoord}

             
        return sample

    def __len__(self):
        return len(self.NewDf)
       
    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 

    def time_series_encoding(self, block_id):
        timeseries = None

        name_split = os.path.split(str(block_id))[-1]
        year       = name_split[-4:]

        if year == '2016': 
            days = [91, 95, 107, 116, 135, 136, 141, 150, 161, 166, 171, 176, 182, 195, 202]
        elif year =='2017':
            days = [90, 109, 119, 134, 140, 156, 169, 176, 179, 184, 190, 192, 195, 197, 202]
        elif year =='2018':
            days = [91, 105, 112, 115, 121, 131, 135, 142, 152, 155, 165, 175, 185, 191, 194]
        elif year =='2019':
            days = [91, 101, 112, 115, 121, 124, 131, 145, 152, 155, 164, 171, 181, 192, 202]
        
        for day in days: 
            this_week_matrix = np.full((1, self.wsize, self.wsize), 1 - np.sin(day/(365*np.pi))) 
            this_week_matrix = np.expand_dims(this_week_matrix, axis = -1)
            if timeseries is None:
                timeseries = this_week_matrix
            else:
                timeseries = np.concatenate([timeseries, this_week_matrix], axis = -1)

        return timeseries  
    
  
