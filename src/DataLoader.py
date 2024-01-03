import os
import os.path as path
import numpy as np
from glob import glob
import random
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from src.configs import blocks_information_dict

#====================================================================================================================================#

def cost_sensitive_weight_sampler(df):
    
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

def return_cost_sensitive_weight_sampler(train, val, test, save_dir, run_status: str):
    
    save_dir = os.path.join(save_dir, 'coords')

    if run_status == 'train':
        train_df   = cost_sensitive_weight_sampler(train)
        train_df.reset_index(inplace = True, drop = True)
        train_df.to_csv(os.path.join(save_dir,'train.csv'))
        
        valid_df  = cost_sensitive_weight_sampler(val)
        valid_df.reset_index(inplace = True, drop = True)
        valid_df.to_csv(os.path.join(save_dir,'val.csv'))
        
        
        test_df = cost_sensitive_weight_sampler(test)
        test_df.reset_index(inplace = True, drop = True)
        test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    
    elif run_status == 'eval':
        train_df = pd.read_csv(os.path.join(save_dir,'train.csv'), index_col=0) 
        train_df.reset_index(inplace = True, drop = True)

        valid_df = pd.read_csv(os.path.join(save_dir,'val.csv'), index_col=0)
        valid_df.reset_index(inplace = True, drop = True)

        test_df = pd.read_csv(os.path.join(save_dir, 'test.csv'), index_col=0)
        test_df.reset_index(inplace = True, drop = True)

    train_weights = train_df['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = valid_df['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    

    test_weights = test_df['NormWeight'].to_numpy() 
    test_weights = torch.DoubleTensor(test_weights)
    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights)) 
    
    
    return train_sampler, val_sampler, test_sampler

class dataloader_(object):
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

        img_path   = self.NewDf.loc[idx]['IMG_PATH']
        label_path = self.NewDf.loc[idx]['LABEL_PATH']

        # return cropped input image using each patch coordinates
        image = self.crop_gen(img_path, xcoord, ycoord) 
        image = np.swapaxes(image, -1, 0)    
        if self.in_channels == 5: 
            block_timeseries_encode = self.time_series_encoding(block_id)
            image = np.concatenate([image, block_timeseries_encode], axis = 0)

        if self.in_channels == 6: 
            bc_mean_mtx = self.add_input_within_bc_mean(WithinBlockMean)
            block_timeseries_encode = self.time_series_encoding(block_id)
            image = np.concatenate([image, bc_mean_mtx, block_timeseries_encode], axis = 0)

        image = torch.as_tensor(image, dtype=torch.float32)
        image = image / 255.

        # return embedding tensor: 
        CulMatrix = self.patch_cultivar_matrix(cultivar_id)  
        RWMatrix  = self.patch_rw_matrix(rw_id) 
        SpMatrix  = self.patch_sp_matrix(sp_id)
        TMatrix   = self.patch_tid_matrix(t_id)  
        EmbMat    = np.concatenate([CulMatrix, RWMatrix, SpMatrix, TMatrix], axis = 0)

        # return crooped mask tensor: 
        mask  = self.crop_gen(label_path, xcoord, ycoord) 
        mask  = np.swapaxes(mask, -1, 0)
        mask  = torch.as_tensor(mask)
         

        sample = {"image": image, "mask": mask, "EmbMatrix": EmbMat, "block": block_id, "cultivar": cultivar, "X": xcoord, "Y": ycoord}
             
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
    
class pixel_hold_out_dataloader(object):

    def __init__(self, npy_dir, csv_dir, category = None, patch_size = None):
        self.npy_dir      = npy_dir
        self.csv_dir      = csv_dir
        self.wsize        = patch_size

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



def dataloaders(spatial_resolution: int, 
                scenario: str,
                batch_size:int, 
                in_channels:int, 
                patch_size:int, 
                patch_offset: int, 
                cultivar_list: list,
                year_list: list,
                resmapling_status: False,
                exp_name: str): 
    

    if spatial_resolution == 1: 
        data_dir           = '/data2/hkaman/Livingston/data/1m/'
        botneck_size = 10
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/1m/' + 'EXP_' + exp_name

    elif spatial_resolution == 10:
        data_dir           = '/data2/hkaman/Livingston/data/10m/'
        botneck_size = 2
        exp_output_dir = '/data2/hkaman/Livingston/EXPs/10m/' + 'EXP_' + exp_name



    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(exp_output_dir + '/coords')
        os.makedirs(exp_output_dir + '/loss')
        os.makedirs(exp_output_dir + '/RVs')

    best_model_name      = exp_output_dir + '/best_model_' + exp_name + '.pth'
    loss_fig_name        = exp_output_dir + '/loss/loss_'  + exp_name + '.png'
    loss_df_name         = exp_output_dir + '/loss/loss_'  + exp_name + '.csv' 
    loss_weekly_fig_name = exp_output_dir + '/loss/loss_'  + exp_name + '_w.png'
    loss_weekly_df_name  = exp_output_dir + '/loss/loss_'  + exp_name + '_w.csv' 
    #==============================================================================================================#
    #================================================== csv file generator=========================================#
    #==============================================================================================================#
    train_csv, val_csv, test_csv = data_generator(eval_scenario = scenario, spatial_resolution = spatial_resolution, 
                                                                patch_size = patch_size, patch_offset = patch_offset,  
                                                                cultivar_list = cultivar_list, 
                                                                year_list = year_list).return_split_dataframe()

    #==============================================================================================================#
    #============================================      Data Weight Generation     =================================#
    #==============================================================================================================#
    train_sampler, valid_sampler, test_sampler  = return_cost_sensitive_weight_sampler(train_csv, val_csv, test_csv, exp_output_dir, run_status = 'train')

    #==============================================================================================================#
    #============================================ Imprical Data Weight Generation =================================#
    #==============================================================================================================#
    
    train_weights = train_csv['NormWeight'].to_numpy() 
    train_weights = torch.DoubleTensor(train_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, 
                                                                   len(train_weights), replacement=True)    

    val_weights   = val_csv['NormWeight'].to_numpy() 
    val_weights   = torch.DoubleTensor(val_weights)
    val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(val_weights, 
                                                                   len(val_weights), replacement=True)    
    
    test_weights   = test_csv['NormWeight'].to_numpy() 
    test_weights   = torch.DoubleTensor(test_weights)
    test_sampler   = torch.utils.data.sampler.WeightedRandomSampler(test_weights, 
                                                                   len(test_weights), replacement=True)  
 
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#

    if scenario == 'pixel_hold_out': 
        dataset_training = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'train', patch_size = patch_size)
        dataset_validate = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'val',  patch_size = patch_size)
        dataset_test     = pixel_hold_out_dataloader(data_dir, exp_output_dir, category = 'test',  patch_size = patch_size)
    else: 
        dataset_training = dataloader_(data_dir, exp_output_dir, category = 'train', patch_size = patch_size, in_channels = in_channels, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'valid')
        dataset_validate = dataloader_(data_dir, exp_output_dir, category = 'val',  patch_size = patch_size, in_channels = in_channels, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'valid')
        dataset_test     = dataloader_(data_dir, exp_output_dir, category = 'test',  patch_size = patch_size, in_channels = in_channels, 
                                                                                    spatial_resolution = spatial_resolution, run_status = 'eval')     
    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
    # define training and validation data loaders
            # define training and validation data loaders
    if resmapling_status: 
        print(f"resampling is calculating!")
        data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                        shuffle=False,  sampler=train_sampler, num_workers=8)  
        data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                        shuffle=False, sampler= valid_sampler, num_workers=8) 
        data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                        shuffle=False, num_workers=8)  #
    else: 
        data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                        shuffle=True,  num_workers=8) 
        data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                        shuffle=False, num_workers=8)  
        data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                        shuffle=False, num_workers=8) 


    return data_loader_training, data_loader_validate, data_loader_test

class data_generator():
    def __init__(self, eval_scenario: str, 
                    spatial_resolution: int, 
                    patch_size: int, 
                    patch_offset: int,  
                    cultivar_list: list, 
                    year_list: list):

        self.eval_scenario      = eval_scenario
        self.spatial_resolution = spatial_resolution
        self.patch_size         = patch_size
        self.patch_offset       = patch_offset
        self.cultivar_list      = cultivar_list
        self.year_list          = year_list


        if self.cultivar_list is None: 
            self.cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 
                                    'CABERNET_SAUVIGNON','SYMPHONY', 
                                    'MERLOT', 'CHARDONNAY', 
                                    'SYRAH', 'RIESLING']

        if self.spatial_resolution == 1: 
            self.npy_dir = '/data2/hkaman/Livingston/data/1m/'

        else: 
            self.npy_dir = '/data2/hkaman/Livingston/data/10m/'

        self.images_dir  = os.path.join(self.npy_dir, 'imgs')
        self.image_names = os.listdir(self.images_dir)
        self.image_names.sort() 

        self.label_dir   = os.path.join(self.npy_dir, 'labels')
        self.label_names = os.listdir(self.label_dir)
        self.label_names.sort() 


    def return_split_dataframe(self):

        full_dataframe = self.return_dataframe_patch_info()


        if self.eval_scenario == 'pixel_hold_out': 
            train, valid, test = self.pixel_hold_out(full_dataframe)
        elif self.eval_scenario == 'year_hold_out':
            train, valid, test = self.year_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_hold_out': 
            train, valid, test = self.block_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_year_hold_out': 
            train, valid, test = self.block_year_hold_out(full_dataframe)

        print(f"Training Patches: {len(train)}, Validation: {len(valid)} and Test: {len(test)}")
        print("============================= Train =========================================")
        _ = print_df_summary(train)
        print("============================= Validation ====================================")
        _ = print_df_summary(valid)
        print("============================= Test ==========================================")
        _ = print_df_summary(test)
        print("=============================================================================")

        return train, valid, test


    def return_dataframe_patch_info(self): 

        df = pd.DataFrame()

        Block, Cultivar, CID, Trellis, TID, RW, SP = [], [], [], [], [], [], []
        P_means, YEAR, X_COOR, Y_COOR, IMG_P, Label_P  = [], [], [], [], [], []
        
        
        generated_cases = 0
        removed_cases = 0 
        
        
        for idx, name in enumerate(self.label_names):
            # Extract Image path
            image_path = os.path.join(self.images_dir, self.image_names[idx])

            name_split  = os.path.split(name)[-1]
            block_name  = name_split.replace(name_split[12:], '')
            root_name   = name_split.replace(name_split[7:], '')
            year        = name_split.replace(name_split[0:8], '').replace(name_split[12:], '')
            
            res           = {key: blocks_information_dict[key] for key in blocks_information_dict.keys() & {root_name}}
            list_d        = res.get(root_name)
            block_variety = list_d[0]
            block_id      = list_d[1]
            block_rw      = list_d[2]
            block_sp      = list_d[3]
            block_trellis = list_d[5]
            block_tid     = list_d[6]

            label_npy = os.path.join(self.label_dir, name)
            label = np.load(label_npy, allow_pickle=True)
            label = label[0,:,:,0]
            width, height = label.shape[1], label.shape[0]
            
            
            
            for i in range(0, height - self.patch_size, self.patch_offset):
                for j in range(0, width - self.patch_size, self.patch_offset):
                    crop_label = label[i:i+ self.patch_size, j:j+ self.patch_size]
                    
                    if np.any((crop_label < 0)):
                        removed_cases += 1
                        
                    elif np.all((crop_label >= 0)): 

                        generated_cases += 1
                        
                        patch_mean       = np.mean(crop_label)
                        P_means.append(patch_mean)
                    
                        Block.append(block_name)
                        CID.append(block_id)
                        Cultivar.append(block_variety)
                        Trellis.append(block_trellis)
                        TID.append(block_tid)
                        RW.append(int(block_rw))
                        SP.append(int(block_sp))
                        YEAR.append(year)
                        X_COOR.append(i)
                        Y_COOR.append(j)
                        IMG_P.append(image_path)
                        Label_P.append(label_npy)                                        

                        
        df['block']       = Block
        df['X']           = X_COOR
        df['Y']           = Y_COOR
        df['year']        = YEAR
        df['cultivar_id'] = CID
        df['cultivar']    = Cultivar
        df['trellis']     = Trellis
        df['trellis_id']  = TID
        df['row']         = RW
        df['space']       = SP
        df['patch_mean']     = P_means
        df['IMG_PATH']    = IMG_P
        df['LABEL_PATH']  = Label_P
        
        if self.cultivar_list is None:
            Customized_df = df
            
        else: 
            Customized_df = df[df['cultivar'].isin(self.cultivar_list)]
            
        return Customized_df

    def pixel_hold_out(self, df): 

        BlockList = df.groupby(by=["block"])

        train_df_list, valid_df_list, test_df_list = [], [], []

        for block_name, block_df in BlockList:
            if block_df.shape[0] > 2:
                train0, test = train_test_split(block_df, train_size = 0.8, test_size = 0.2, shuffle = True)
                train, valid = train_test_split(train0, train_size = 0.8, test_size = 0.2, shuffle = True)

            train_df_list.append(train)
            valid_df_list.append(valid)
            test_df_list.append(test)

        train = pd.concat(train_df_list)
        valid = pd.concat(valid_df_list)
        test  = pd.concat(test_df_list)

        return train, valid, test

    def year_hold_out(self, df): 

        NewGroupedDf = df.groupby(by=["year"])

        Group1 = NewGroupedDf.get_group(self.year_list[0])
        Group2 = NewGroupedDf.get_group(self.year_list[1])
        Group3 = NewGroupedDf.get_group(self.year_list[2])
        Group4 = NewGroupedDf.get_group(self.year_list[3])

        frames = [Group1, Group2]
        train = pd.concat(frames)
        valid = Group3
        test  = Group4

        return train, valid, test
    
    def block_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block'] = block_root_name
        new_dataframe_basedon_block_mean['cultivar'] = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
            
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 
        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 
            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if n_blocks <= 1: 
                name_2016  = frame['block'].iloc[0] + '_2016'
                name_2017  = frame['block'].iloc[0] + '_2017'
                name_2018  = frame['block'].iloc[0] + '_2018'
                name_2019  = frame['block'].iloc[0] + '_2019'
                training_blocks_names.extend((name_2016, name_2017, name_2018, name_2019))
                
            elif n_blocks == 2:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[1] + '_2016'
                name_2017_1  = frame['block'].iloc[1] + '_2017'
                name_2018_1  = frame['block'].iloc[1] + '_2018'
                name_2019_1  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))
                
            elif n_blocks == 3:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[2] + '_2016'
                name_2017_1  = frame['block'].iloc[2] + '_2017'
                name_2018_1  = frame['block'].iloc[2] + '_2018'
                name_2019_1  = frame['block'].iloc[2] + '_2019'
                
                testing_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))  
                
                name_2016_2  = frame['block'].iloc[1] + '_2016'
                name_2017_2  = frame['block'].iloc[1] + '_2017'
                name_2018_2  = frame['block'].iloc[1] + '_2018'
                name_2019_2  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_2, name_2017_2, name_2018_2, name_2019_2)) 
                
            elif n_blocks > 3:
                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
 

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):
                    name_2016  = block_mean_yield_2017[i][0] + '_2016'
                    name_2017  = block_mean_yield_2017[i][0] + '_2017'
                    name_2018  = block_mean_yield_2017[i][0] + '_2018'
                    name_2019  = block_mean_yield_2017[i][0] + '_2019'

                    if i == te: 
                        testing_blocks_names.append(name_2016)
                        testing_blocks_names.append(name_2017)
                        testing_blocks_names.append(name_2018)
                        testing_blocks_names.append(name_2019)
                        te = te + 3
                    elif i == val: 
                        validation_blocks_names.append(name_2016)
                        validation_blocks_names.append(name_2017)
                        validation_blocks_names.append(name_2018)
                        validation_blocks_names.append(name_2019)

                        val = val + 3
                    else:
                        training_blocks_names.append(name_2016)
                        training_blocks_names.append(name_2017)
                        training_blocks_names.append(name_2018)
                        training_blocks_names.append(name_2019)

        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test

    def block_year_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block']      = block_root_name
        new_dataframe_basedon_block_mean['cultivar']   = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
        
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 

        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 

            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if frame.shape[0] == 3:
                

                name_0  = frame['block'].iloc[0] + '_' + self.year_list[0]
                name_1  = frame['block'].iloc[0] + '_' + self.year_list[1]
                training_blocks_names.append(name_0)
                training_blocks_names.append(name_1)
                
                name_2  = frame['block'].iloc[1] + '_' + self.year_list[2]
                validation_blocks_names.append(name_2) 

                name_3  = frame['block'].iloc[2] + '_' + self.year_list[3]
                testing_blocks_names.append(name_3) 


                
            elif frame.shape[0] > 3:

                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
                #print(block_mean_yield_2017)
                #print("============================")

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):

                    if i == te: 
                        name_3  = block_mean_yield_2017[i][0] + '_' + self.year_list[3]
                        testing_blocks_names.append(name_3)
                        te = te + 3
                        #print(f"{cul}: {name_3}")
                    elif i == val: 
                        name_2  = block_mean_yield_2017[i][0] + '_' + self.year_list[2]
                        validation_blocks_names.append(name_2)
                        val = val + 3
                        #print(f"{cul}: {name_2}")
                    else:
                        name_0  = block_mean_yield_2017[i][0] + '_' + self.year_list[0]
                        name_1  = block_mean_yield_2017[i][0] + '_' + self.year_list[1]
                        #print(f"{cul}: {name_0, name_1}")
                        training_blocks_names.append(name_0)
                        training_blocks_names.append(name_1)

                    
                    
                    
                    #print(f"with MORE than 3: {name_0, name_1, name_2, name_3}")

        #print(validation_blocks_names)
        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test

def print_df_summary(df):

    cultivars = df.groupby(by = 'cultivar')
    print(f"There are {len(cultivars)} cultivar types.")

    for cul, df in cultivars:
        
        blocks_within_cultivar = df.groupby(by = 'block')
        print(f"Cultivar {cul} has {len(blocks_within_cultivar)} blocks and {df.shape[0]} samples:")
        for b, df2 in blocks_within_cultivar:
            print(f"     Block {b} has {df2.shape[0]} samples.")
