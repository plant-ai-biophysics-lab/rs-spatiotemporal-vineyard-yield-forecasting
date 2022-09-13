import os
import os.path as path
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import cv2
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio 
from rasterio.plot import show, show_hist
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.merge import merge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import random 
from scipy.signal import convolve2d

#from skimage.measure import compare_ssim
root_dir = '/data2/hkaman/Livingston/' 


params = {'legend.fontsize': 14,
         'axes.labelsize': 14,
         'axes.titlesize':14,
         'xtick.labelsize':13,
         'ytick.labelsize':13}
plt.rcParams.update(params)






def ndvi_(img):
    
    ndvi = (img[3,:,:]-img[2,:,:])/(img[3,:,:]+img[2,:,:])

    return ndvi


def ndvi_concat(img):
    out = None
    for i in range(img.shape[3]):
        this_image = img[:,:,:, i]
        print()
        ndvi = (this_image[3,:,:]-this_image[2,:,:])/(this_image[3,:,:]+this_image[2,:,:])
        ndvi = np.expand_dims(ndvi, axis = 0)
        this_image = np.concatenate([this_image, ndvi], axis = 0) 
        this_image = np.expand_dims(this_image, axis = -1) 
        if out is None: 
            out = this_image
        else: 
            out = np.concatenate([out, this_image], axis = -1)
        
    return out 


def sentinel_image_stretch(image): 
    # normalize into 0-255
    red = image[0,:,:]
    red_n = ((255*red/np.max(red))).astype(np.uint8)
    red_n = np.expand_dims(red_n, axis = 0)

    green = image[1,:,:]
    green_n = ((255*green/np.max(green))).astype(np.uint8)
    green_n = np.expand_dims(green_n, axis = 0)


    blue = image[2,:,:]
    blue_n = ((255*blue/np.max(blue))).astype(np.uint8)
    blue_n = np.expand_dims(blue_n, axis = 0)

    nir = image[3,:,:]
    nir_n = ((255*nir/np.max(nir))).astype(np.uint8)
    nir_n = np.expand_dims(nir_n, axis = 0) 

    image_norm = np.concatenate([red_n, green_n, blue_n, nir_n], axis = 0)

    return image_norm

def img_crop(img, polygon):
    with rasterio.open(img) as inimg:
        in_cropped, out_transform_in = mask(inimg,
        [polygon],crop=True)
        in_cropped_meta = inimg.meta.copy()
        in_cropped_meta.update({"driver": "GTiff",
            "height": in_cropped.shape[1],
            "width": in_cropped.shape[2], 
            "transform": out_transform_in})
    return in_cropped_meta, in_cropped

def reverse_coordinates(pol):
    """
    Reverse the coordinates in pol
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    Returns [[y1,x1],[y2,x2],...,[yN,xN]]
    """
    #return [list(f[-1::-1]) for f in pol]
    list = []
    for f in pol:
        f2 = f[-1::-1]
        list.append(f2)
    return list

def list_to_coord (list, src):
    coord = []
    for p in list:
        cor = src.transform*p
        coord.append(cor)

    return coord

def to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [[wind_.row_off,wind_.col_off],
            [wind_.row_off,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off],
            [wind_.row_off,wind_.col_off]]

def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
            [bbox[2],bbox[1]],
            [bbox[2],bbox[3]],
            [bbox[0],bbox[3]],
            [bbox[0],bbox[1]]]

def pol_to_np(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    return np.array([list(l) for l in pol])

def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:,0]),
                    np.min(arr[:,1]),
                    np.max(arr[:,0]),
                    np.max(arr[:,1]))   


def block_cultivar_mean_std(df):

    
    block_groups = df.groupby(by = 'block')
    
    block_mean_vector, block_std_vector = [], []
    block_root_name1, block_name, block_cultivar1, block_mean, n_patches_p_block  = [], [], [], [], []
    block_root_name2, block_cultivar2, block_mean2, block_std2 = [], [], [], []
    

    df1 = pd.DataFrame()
    block_groups = df.groupby(by = 'block')

    
    for block_id, block_df in block_groups:
        split_name = os.path.split(block_id)[-1]
        root_name = split_name.replace(split_name[7:], '')
        block_root_name1.append(root_name)
        
        block_name.append(block_id)
        this_n_patches_p_block = len(block_df)
        n_patches_p_block.append(this_n_patches_p_block)

        this_block_mean= block_df['patch_mean'].mean()
        block_mean.append(this_block_mean)
        
        
        b_mean_vector  = np.array(this_n_patches_p_block*[this_block_mean], dtype=np.float32)
        block_mean_vector.append(b_mean_vector)
        this_block_std = block_df['patch_mean'].std()
        b_std_vector   = np.array(this_n_patches_p_block*[this_block_std])
        block_std_vector.append(b_std_vector)
        
        
        block_cultivar1.append(block_df['cultivar'].iloc[0])

    block_mean_vector = np.concatenate(block_mean_vector)    
    block_std_vector  = np.concatenate(block_std_vector) 
       
    
        
    df1['block_root'] = block_root_name1
    df1['block']      = block_name
    df1['cultivar']   = block_cultivar1
    df1['block_mean'] = block_mean
    df1['n_patches']  = n_patches_p_block

    
    
    df2 = pd.DataFrame()

    df1_groups = df1.groupby(by = 'block_root')
    within_block_mean, within_block_std = [], []
    all_block_patches = []
    
    for b, dfbbb in df1_groups: 
        
        block_root_name2.append(b)
        block_cultivar2.append(dfbbb['cultivar'].iloc[0])
        blocks_mean = dfbbb['block_mean'].mean()
        block_mean2.append(blocks_mean)
        blocks_std = dfbbb['block_mean'].std()
        block_std2.append(blocks_std)
        
        all_blocks_patches = dfbbb['n_patches'].sum()
        all_block_patches.append(all_blocks_patches)
        
        within_block_mean_vector = np.array(all_blocks_patches*[blocks_mean])
        within_block_mean.append(within_block_mean_vector)
        
        within_block_std_vector = np.array(all_blocks_patches*[blocks_std])
        within_block_std.append(within_block_std_vector)
        
    within_block_mean = np.concatenate(within_block_mean)    
    within_block_std  = np.concatenate(within_block_std) 
    
    
    df2['block_root'] = block_root_name2
    df2['cultivar']   = block_cultivar2
    df2['block_mean'] = block_mean2
    df2['block_std']  = block_std2
    df2['blocks_patches'] = all_block_patches

    cultivar_groups = df2.groupby(by = 'cultivar')
    within_cultivar_mean, within_cultivar_std = [], []
    cultivar, cultivars_mean, cultivars_std = [], [], []
    cultivar_patches = []
    
    df3 = pd.DataFrame()
    for cultivar_id, cultivar_df in cultivar_groups:
        cultivar.append(cultivar_id)
        
        n_patches_cultivar = cultivar_df['blocks_patches'].sum()
        cultivar_patches.append(n_patches_cultivar)
        
        cultivar_m = cultivar_df['block_mean'].mean()
        cultivars_mean.append(cultivar_m)
        within_cultivar_mean_vector = np.array(n_patches_cultivar*[cultivar_m])
        within_cultivar_mean.append(within_cultivar_mean_vector)
        
        cul_std = cultivar_df['block_mean'].std()
        if np.isnan(cul_std):
            cul_std = cultivar_df['block_std'].iloc[0]
            cultivars_std.append(cul_std)
            within_cultivar_std_vector = np.array(n_patches_cultivar*[cul_std])
            within_cultivar_std.append(within_cultivar_std_vector)
        else:
            cultivars_std.append(cul_std)
            within_cultivar_std_vector = np.array(n_patches_cultivar*[cul_std])
            within_cultivar_std.append(within_cultivar_std_vector)


    df3['cultivar']      = cultivar
    df3['cultivar_mean'] = cultivars_mean
    df3['cultivar_std']  = cultivars_std
    df3['cultivar_patches'] = cultivar_patches
    
    within_cultivar_mean = np.concatenate(within_cultivar_mean)
    within_cultivar_std = np.concatenate(within_cultivar_std)
    df['block_mean']        = block_mean_vector
    df['block_std']         = block_std_vector
    df['win_block_mean']    = within_block_mean
    df['win_block_std']     = within_block_std
    df['win_cultivar_mean'] = within_cultivar_mean
    df['win_cultivar_std']  = within_cultivar_std

    return df, df1, df2, df3 

class split_train_valid_test(): 
    def __init__(self, df = None, scenario = None, year_list = None):
        self.df = df
        self.scenario = scenario
        self.year_list = year_list

    
    def __train_valid_test__(self):

        if self.scenario == 1: 
            train, valid, test = self.split_data_scenario_1()

        elif self.scenario == 2:
            train, valid, test = self.split_data_scenario_2()

        elif self.scenario == 3:
            train, valid, test = self.split_data_scenario_3()

        elif self.scenario == 4:
            train, valid, test = self.split_data_scenario_4()

        #elif self.scenario == 'within_cultivar':
        #    train, valid, test = self.split_data_scenaerio_4()


        return train, valid, test 

    def split_data_scenario_1(self):

        BlockList = self.df.groupby(by=["block"])

        train_df_list, valid_df_list, test_df_list = [], [], []

        for block_name, block_df in BlockList:
            #print(f"{block_name}: {block_df.shape}")
            train0, test = train_test_split(block_df, train_size = 0.8, test_size = 0.2, shuffle = True)
            train, valid = train_test_split(train0, train_size = 0.8, test_size = 0.2, shuffle = True)

            train_df_list.append(train)
            valid_df_list.append(valid)
            test_df_list.append(test)

        train = pd.concat(train_df_list)
        valid = pd.concat(valid_df_list)
        test  = pd.concat(test_df_list)

        return train, valid, test


    def split_data_scenario_2(self):

        NewGroupedDf = self.df.groupby(by=["year"])

        '''Group2016 = NewGroupedDf.get_group('2016')
        Group2017 = NewGroupedDf.get_group('2017')
        Group2018 = NewGroupedDf.get_group('2018')
        Group2019 = NewGroupedDf.get_group('2019')

        frames = [Group2016, Group2019]
        train = pd.concat(frames)
        val   = Group2017
        test  = Group2018'''

        Group1 = NewGroupedDf.get_group(self.year_list[0])
        Group2 = NewGroupedDf.get_group(self.year_list[1])
        Group3 = NewGroupedDf.get_group(self.year_list[2])
        Group4 = NewGroupedDf.get_group(self.year_list[3])

        frames = [Group1, Group2]


        train = pd.concat(frames)
        valid = Group3
        test  = Group4

        return train, valid, test


    def split_data_scenario_3(self): 
        
        groupby_year = self.df.groupby(by = 'year')
        year2017 = groupby_year.get_group('2017')
        
        df1 = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        block2017_groups = year2017.groupby(by = 'block')
        for block, blockdf in block2017_groups:
            name_split = os.path.split(block)[-1]
            root_name = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        df1['block'] = block_root_name
        df1['cultivar'] = cultivar
        df1['block_mean'] = b_mean
            
        
        GroupList = df1.groupby(by=["cultivar"]) 
        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in GroupList: 
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
                # Print out the feature and importances 
                #[print('Block: {:20} Mean: {}'.format(*pair)) for pair in block_mean_yield]

                sorted_block_names = [block_mean_yield_2017[i][0] for i in range(len(block_mean_yield_2017))]
                sorted_block_yield = [block_mean_yield_2017[i][1] for i in range(len(block_mean_yield_2017))]

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


        train = self.df[self.df['block'].isin(training_blocks_names)]
        valid = self.df[self.df['block'].isin(validation_blocks_names)]
        test  = self.df[self.df['block'].isin(testing_blocks_names)]


        return train, valid, test

    def split_data_scenario_4(self):
        
        df1 = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        block_groups = self.df.groupby(by = 'block')
        for block, blockdf in block_groups:
            name_split = os.path.split(block)[-1]
            root_name = name_split.replace(name_split[12:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        df1['block'] = block_root_name
        df1['cultivar'] = cultivar
        df1['block_mean'] = b_mean
            
        
        GroupList = df1.groupby(by=["cultivar"]) 
        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in GroupList:
            
            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if n_blocks <= 1: 
                training_blocks_names.append(frame['block'].iloc[0])
            elif n_blocks == 2:
                training_blocks_names.append(frame['block'].iloc[0])
                validation_blocks_names.append(frame['block'].iloc[1]) 
            elif n_blocks == 3:   
                training_blocks_names.append(frame['block'].iloc[0])
                validation_blocks_names.append(frame['block'].iloc[2])
                testing_blocks_names.append(frame['block'].iloc[1])
                
            elif n_blocks > 3:    
                blocks_     = frame['block']
                blocks_mean_ = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_, blocks_mean_)]

                block_mean_yield = sorted(block_mean_yield, key = lambda x: x[1], reverse = True)
                # Print out the feature and importances 
                #[print('Block: {:20} Mean: {}'.format(*pair)) for pair in block_mean_yield]

                sorted_block_names = [block_mean_yield[i][0] for i in range(len(block_mean_yield))]
                sorted_block_yield = [block_mean_yield[i][1] for i in range(len(block_mean_yield))]

                te  = 1
                val = 2
                for i in range(len(block_mean_yield)):
                    name_b  = block_mean_yield[i][0] 

                    if i == te: 
                        testing_blocks_names.append(name_b)
                        te = te + 4
                    elif i == val: 
                        validation_blocks_names.append(name_b)
                        val = val + 4
                    else:
                        training_blocks_names.append(name_b)


        train = self.df[self.df['block'].isin(training_blocks_names)]
        valid = self.df[self.df['block'].isin(validation_blocks_names)]
        test  = self.df[self.df['block'].isin(testing_blocks_names)]


        return train, valid, test



def scenario1_split_data(df):

    BlockList = df.groupby(by=["block"])

    train_df_list, valid_df_list, test_df_list = [], [], []

    for block_name, block_df in BlockList:
        #print(f"{block_name}: {block_df.shape}")
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



def scenario3_split_data(df): 
    
    groupby_year = df.groupby(by = 'year')
    year2017 = groupby_year.get_group('2017')
    
    df1 = pd.DataFrame()
    block_root_name, cultivar, b_mean = [], [], []
    
    block2017_groups = year2017.groupby(by = 'block')
    for block, blockdf in block2017_groups:
        name_split = os.path.split(block)[-1]
        root_name = name_split.replace(name_split[7:], '')
        block_root_name.append(root_name)
        
        cultivar.append(blockdf['cultivar'].iloc[0])
        b_mean.append(blockdf['patch_mean'].mean())
        
    df1['block'] = block_root_name
    df1['cultivar'] = cultivar
    df1['block_mean'] = b_mean
        
    
    GroupList = df1.groupby(by=["cultivar"]) 
    training_blocks_names = []
    validation_blocks_names = []
    testing_blocks_names = []
    
    for cul, frame in GroupList: 
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
            # Print out the feature and importances 
            #[print('Block: {:20} Mean: {}'.format(*pair)) for pair in block_mean_yield]

            sorted_block_names = [block_mean_yield_2017[i][0] for i in range(len(block_mean_yield_2017))]
            sorted_block_yield = [block_mean_yield_2017[i][1] for i in range(len(block_mean_yield_2017))]

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

            
    return training_blocks_names, validation_blocks_names, testing_blocks_names


def each_cultivar_split_data(root_dir, cultivar):

    label_dir = os.path.join(root_dir, 'labels')
    label_names = os.listdir(label_dir)
    label_names.sort() 
    
    
    df = pd.DataFrame()

    Block,  Cultivar, means  = [], [], []

    for idx, name in enumerate(label_names):
        # Extract Image path
        label_npy = os.path.join(label_dir, name)
        label = np.load(label_npy, allow_pickle=True)
        label = label[0,:,:,0]
        
        #Block mean value
        label_mean_value = np.mean(label)
    
        name_split = os.path.split(name)[-1]
        root_name = name_split.replace(name_split[12:], '')
        block_name = name_split.replace(name_split[7:], '')
        year = name_split.replace(name_split[0:8], '').replace(name_split[12:], '')

        res = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d = res.get(block_name)
        block_variety = list_d[0]
        block_id      = list_d[1]
        block_rw      = list_d[2]
        block_sp      = list_d[3]
        block_trellis = list_d[5]
        block_tid     = list_d[6]
        
        Block.append(root_name)
        Cultivar.append(block_variety)
        means.append(label_mean_value) 
        
    df['block']    = Block
    df['cultivar'] = Cultivar
    df['mean']     = means
    
    GroupList = df.groupby(by=["cultivar"]) 
    
    df_new = GroupList.get_group(cultivar[0])
    train_blocks_names = []
    validation_blocks_names = []
    testing_blocks_names = []
        
    block_names = df_new['block'] 
    
    means = df_new['mean']  
    #newinfo
    block_mean_yield = [(blocks, mean) for blocks, 
                           mean in zip(df_new['block'], df_new['mean'])]

    block_mean_yield = sorted(block_mean_yield, key = lambda x: x[1], reverse = True)
    te =1
    val = 2

    for i in range(len(block_mean_yield)):
        name_b  = block_mean_yield[i][0] 
        yield_b = block_mean_yield[i][1]

        if i == te: 
            testing_blocks_names.append(name_b) 
            te = te + 4
        elif i == val: 
            validation_blocks_names.append(name_b) 
            val = val + 4
        else:
            train_blocks_names.append(name_b)
    
    
    return df, train_blocks_names, validation_blocks_names, testing_blocks_names


#==============================================================================================#
#====================================  .npy map generation V1    ==============================#
#==============================================================================================#

def categ_crop_coord_csv(npy_dir, img_size = None, offset = 0, cultivar_list = None): 
    if cultivar_list is None: 
        cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 'CABERNET_SAUVIGNON','SYMPHONY',  
                        'MERLOT', 'CHARDONNAY', 'SYRAH', 'RIESLING']#'PINOT_GRIS',
    # first calculate the csv file for percentile range of each block: 
    block_percentiles = blocklevel_categ_csv(npy_dir)
    
    #===========
    images_dir  = os.path.join(npy_dir, 'imgs')
    image_names = os.listdir(images_dir)
    image_names.sort() 

    label_dir   = os.path.join(npy_dir, 'labels')
    label_names = os.listdir(label_dir)
    label_names.sort() 

    df = pd.DataFrame(columns = ['block','cultivar', 'cultivar_id', 'trellis', 'trellis_id', 'row', 'space', 'patch_mean', 'year', 'X', 'Y', 'IMG_PATH', 'LABEL_PATH'])

    Block, Cultivar, CID, Trellis, TID, RW, SP                               = [], [], [], [], [], [], []
    P_means, Cmean, Cstd, Bmean, Bstd, YEAR, X_COOR, Y_COOR, IMG_P, Label_P  = [], [], [], [], [], [], [], [], [], []
    p_lower_perc, p_upper_perc, p_01gamma, p_99gamma, p_minyield, p_maxyield, p_mingamma, p_maxgamma = [], [], [], [], [], [], [], []
    block_l_p, block_u_p, block_min, block_max = [], [], [], []
    
    
    
    generated_cases = 0
    removed_cases = 0 
    
    
    for idx, name in enumerate(label_names):
        # Extract Image path
        image_path = os.path.join(images_dir, image_names[idx])

        name_split  = os.path.split(name)[-1]
        block_name  = name_split.replace(name_split[12:], '')
        root_name   = name_split.replace(name_split[7:], '')
        year        = name_split.replace(name_split[0:8], '').replace(name_split[12:], '')
        
        res           = {key: Categ_[key] for key in Categ_.keys() & {root_name}}
        list_d        = res.get(root_name)
        block_variety = list_d[0]
        block_id      = list_d[1]
        block_rw      = list_d[2]
        block_sp      = list_d[3]
        block_trellis = list_d[5]
        block_tid     = list_d[6]

        label_npy = os.path.join(label_dir, name)
        label = np.load(label_npy, allow_pickle=True)
        label = label[0,:,:,0]
        width, height = label.shape[1], label.shape[0]
        
        block_information = block_percentiles.loc[block_percentiles['block'] == block_name] 
        
        block_lower_percentile = block_information['yield01'].values[0]
        block_upper_percentile = block_information['yield99'].values[0]
        
        block_min_yield = block_information['yieldmin'].values[0]
        block_max_yield = block_information['yieldmax'].values[0]
        
        
        for i in range(0, height-img_size, offset):
            for j in range(0, width-img_size, offset):
                crop_label = label[i:i+img_size, j:j+img_size]
                
                if np.any((crop_label < 0)):
                    removed_cases += 1
                    
                elif np.all((crop_label >= 0)): 

                    generated_cases += 1
                    # calculating the percentile of each block and each patch: 
                    # a) block lower and upper percentile: 
                    block_l_p.append(block_lower_percentile)
                    block_u_p.append(block_upper_percentile)
                    # b) patch lower and upper percentile 
                    patch_lower_perc = np.percentile(crop_label, 1)
                    patch_upper_perc = np.percentile(crop_label, 99)
                    p_lower_perc.append(patch_lower_perc)
                    p_upper_perc.append(patch_upper_perc)
                    # c) calculating the gamma based on percentile 
                    p_01gamma.append((patch_lower_perc - block_lower_percentile)/(block_upper_percentile - block_lower_percentile))
                    p_99gamma.append((patch_upper_perc - block_lower_percentile)/(block_upper_percentile - block_lower_percentile))
                    
                    # calculating the minmax of each block and each patch: 
                    # a) block lower and upper percentile: 
                    block_min.append(block_min_yield)
                    block_max.append(block_max_yield)
                    # b) patch lower and upper percentile 
                    patch_min_yield = np.min(crop_label)
                    patch_max_yield = np.max(crop_label)
                    p_minyield.append(patch_lower_perc)
                    p_maxyield.append(patch_upper_perc)
                    # c) calculating the gamma based on percentile 
                    p_mingamma.append((patch_min_yield - block_min_yield)/(block_max_yield - block_min_yield))
                    p_maxgamma.append((patch_max_yield - block_min_yield)/(block_max_yield - block_min_yield))
                    
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
    df['b_lower_perc']   = block_l_p
    df['b_upper_perc']   = block_u_p
    df['b_minyield']     = block_min
    df['b_maxyield']     = block_max
    
    df['patch_mean']     = P_means
    df['p_lower_perc']   = p_lower_perc
    df['p_upper_perc']   = p_upper_perc
    df['p_01gamma']      = p_01gamma
    df['p_99gamma']      = p_99gamma
    df['p_minyield']     = p_minyield
    df['p_maxyield']     = p_maxyield
    df['p_mingamma']     = p_mingamma
    df['p_maxgamma']     = p_maxgamma


    df['IMG_PATH']    = IMG_P
    df['LABEL_PATH']  = Label_P
    
    
    if cultivar_list is None:
        Newdf = df
        
    else: 
        Newdf = df[df['cultivar'].isin(cultivar_list)]

        
    return Newdf


def scenario_csv_generator(scenario = None, spatial_resolution = None, img_size = None, offset = None,  cultivar_list = None, year_list = None): 

    if spatial_resolution == 1: 
        npy_dir           = '/data2/hkaman/Livingston/data/1m/'
        bsize = 10
    elif spatial_resolution == 10:
        npy_dir           = '/data2/hkaman/Livingston/data/10m/'
        bsize = 2

    
    #path = os.path.join(npy_dir, 'coords')
    #isdir = os.path.isdir(path) 
    
    #if isdir is False:
    #    os.mkdir(path)
        

    df = categ_crop_coord_csv(npy_dir, img_size = img_size, offset = offset, cultivar_list = cultivar_list)
    df, _, _, _ = block_cultivar_mean_std(df)


    if scenario == 1: 
        #shuffled_df = df.sample(frac = 1)
        #train1, test = train_test_split(shuffled_df, test_size=0.2, shuffle=True, stratify = None)
        #train, val = train_test_split(train1, test_size=0.2, random_state=42, shuffle=True)
        train, val, test = scenario1_split_data(df)

           
    elif scenario == 2: 
        NewGroupedDf = df.groupby(by=["year"])

        Group1 = NewGroupedDf.get_group(year_list[0])
        Group2 = NewGroupedDf.get_group(year_list[1])
        Group3 = NewGroupedDf.get_group(year_list[2])
        Group4 = NewGroupedDf.get_group(year_list[3])

        frames = [Group1, Group2]
        train = pd.concat(frames)
        val   = Group3
        test  = Group4
        
        
    elif scenario == 3: 
        
        training_blocks_names, validation_blocks_names, testing_blocks_names = scenario3_split_data(df)
        
        train = df[df['block'].isin(training_blocks_names)]
        val = df[df['block'].isin(validation_blocks_names)]
        test = df[df['block'].isin(testing_blocks_names)]

    elif scenario == 4:
        
        training_blocks_names, validation_blocks_names, testing_blocks_names = scenario4_split_data(df)
        
        train = df[df['block'].isin(training_blocks_names)]
        val   = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)]
        
    elif scenario == 5:
        
        thisdf, training_blocks_names, validation_blocks_names, testing_blocks_names = each_cultivar_split_data(root, cultivar)
        
        train = df[df['block'].isin(training_blocks_names)]
        val = df[df['block'].isin(validation_blocks_names)]
        test = df[df['block'].isin(testing_blocks_names)]
        
       
    print(f"Training Patches: {len(train)}, Validation: {len(val)} and Test: {len(test)}")
    print("============================= Train =========================================")
    _ = print_df_summary(train)
    print("============================= Validation ====================================")
    _ = print_df_summary(val)
    print("============================= Test ==========================================")
    _ = print_df_summary(test)
    print("=============================================================================")
    
    return train, val, test

#==============================================================================================#
#====================================  .npy map generation V2    ==============================#
#==============================================================================================#

def sentinel_timeseries_crop_gen(image_names, image_dir, roi_polygon_src_coords, spatial_resolution, h, w):

    timeseries_image = None

    for img in image_names:

        img_name =  image_dir + '/' + img 
        # crop the sentinel image based on the corner of block shapefile corrdinates
        _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
        # sentinel image normalization! 
        image_cropped = sentinel_image_stretch(image_cropped) 
        # change the order of channels.
        image_cropped = np.moveaxis(image_cropped, 0, 2)
        # convert nan to zero
        image_cropped = np.nan_to_num(image_cropped)
        # adding one more dimension

        

        if spatial_resolution != 10: 

            scale = (10/spatial_resolution)
            up10x_img = cv2.resize(image_cropped, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            up10x_img = up10x_img[0:h, 0:w,:]
            up_img_ex = np.expand_dims(up10x_img, axis = 0)

        elif spatial_resolution == 10:
            up10x_img = image_cropped[0:h, 0:w,:]
            up_img_ex = np.expand_dims(up10x_img, axis = 0)


        if timeseries_image is None:
            timeseries_image = up_img_ex
        else:
            timeseries_image = np.concatenate((timeseries_image, up_img_ex), axis = 0) 

    return timeseries_image



def label_data_gen(label_data, spatial_resolution): 

    if spatial_resolution == 1:
        label_data = np.expand_dims(label_data, axis = 0)
        #label_data = np.expand_dims(label_data, axis = -1)

    elif spatial_resolution !=1: 
        scale = 1/spatial_resolution
        label_data = cv2.resize(label_data, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        label_data = np.expand_dims(label_data, axis = 0)
        label_data = np.expand_dims(label_data, axis = -1)

    h = label_data.shape[1]
    w = label_data.shape[2]

    return label_data, h, w

def label_crop(label, polygon):
    with rasterio.open(label) as input:
        cropped_img, out_transform = mask(input,
        [polygon],crop=True)
        meta = input.meta.copy()
        meta.update({"driver": "GTiff",
            "height": cropped_img.shape[1],
            "width": cropped_img.shape[2], 
            "transform": out_transform})  
    return meta, cropped_img 

def blocklevel_csv_gen(block_name, label, img_size = None, offset = None, cultivar_list = None): 

    if cultivar_list is None: 
        cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 'CABERNET_SAUVIGNON','SYMPHONY', 'PINOT_GRIS', 
                        'MERLOT', 'CHARDONNAY', 'SYRAH', 'RIESLING'] #'PINOT_GRIS', 
    
    df = pd.DataFrame(columns = ['block','cultivar', 'cultivar_id', 'trellis', 'trellis_id', 'row', 'space', 'patch_mean', 'year', 'X', 'Y'])

    Block, Cultivar, CID, Trellis, TID, RW, SP    = [], [], [], [], [], [], []
    P_means,YEAR, X_COOR, Y_COOR = [], [], [], []
    p_lower_perc, p_upper_perc, p_01gamma, p_99gamma, p_minyield, p_maxyield, p_mingamma, p_maxgamma = [], [], [], [], [], [], [], []
    block_l_p, block_u_p, block_min, block_max = [], [], [], []
    
    generated_cases = 0
    removed_cases = 0 
 
    root_name   = block_name.replace(block_name[7:], '')
    year        = block_name.replace(block_name[0:8], '')
    
    res           = {key: Categ_[key] for key in Categ_.keys() & {root_name}}
    list_d        = res.get(root_name)
    block_variety = list_d[0]
    block_id      = list_d[1]
    block_rw      = list_d[2]
    block_sp      = list_d[3]
    block_trellis = list_d[5]
    block_tid     = list_d[6]
    

    label = label[0,:,:,0]
    width, height = label.shape[1], label.shape[0]
    

    block_lower_percentile = np.percentile(label, int(1))
    block_upper_percentile = np.percentile(label, int(99))
    
    block_min_yield = np.min(label)
    block_max_yield = np.max(label)
    
    
    for i in range(0, height-img_size, offset):
        for j in range(0, width-img_size, offset):
            crop_label = label[i:i+img_size, j:j+img_size]
            
            if np.any((crop_label < 0)):
                removed_cases += 1
                
            elif np.all((crop_label >= 0)): 
                generated_cases += 1
                # calculating the percentile of each block and each patch: 
                # a) block lower and upper percentile: 
                block_l_p.append(block_lower_percentile)
                block_u_p.append(block_upper_percentile)
                # b) patch lower and upper percentile 
                patch_lower_perc = np.percentile(crop_label, 1)
                patch_upper_perc = np.percentile(crop_label, 99)
                p_lower_perc.append(patch_lower_perc)
                p_upper_perc.append(patch_upper_perc)
                # c) calculating the gamma based on percentile 
                p_01gamma.append((patch_lower_perc - block_lower_percentile)/(block_upper_percentile - block_lower_percentile))
                p_99gamma.append((patch_upper_perc - block_lower_percentile)/(block_upper_percentile - block_lower_percentile))
                
                # calculating the minmax of each block and each patch: 
                # a) block lower and upper percentile: 
                block_min.append(block_min_yield)
                block_max.append(block_max_yield)
                # b) patch lower and upper percentile 
                patch_min_yield = np.min(crop_label)
                patch_max_yield = np.max(crop_label)
                p_minyield.append(patch_lower_perc)
                p_maxyield.append(patch_upper_perc)
                # c) calculating the gamma based on percentile 
                p_mingamma.append((patch_min_yield - block_min_yield)/(block_max_yield - block_min_yield))
                p_maxgamma.append((patch_max_yield - block_min_yield)/(block_max_yield - block_min_yield))
                
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
                                  

                    
    df['block']          = Block
    df['X']              = X_COOR
    df['Y']              = Y_COOR
    df['year']           = YEAR
    df['cultivar_id']    = CID
    df['cultivar']       = Cultivar
    df['trellis']        = Trellis
    df['trellis_id']     = TID
    df['row']            = RW
    df['space']          = SP
    df['b_lower_perc']   = block_l_p
    df['b_upper_perc']   = block_u_p
    df['b_minyield']     = block_min
    df['b_maxyield']     = block_max
    df['patch_mean']     = P_means
    df['p_lower_perc']   = p_lower_perc
    df['p_upper_perc']   = p_upper_perc
    df['p_01gamma']      = p_01gamma
    df['p_99gamma']      = p_99gamma
    df['p_minyield']     = p_minyield
    df['p_maxyield']     = p_maxyield
    df['p_mingamma']     = p_mingamma
    df['p_maxgamma']     = p_maxgamma

    
    
    if cultivar_list is None:
        Newdf = df
        
    else: 
        Newdf = df[df['cultivar'].isin(cultivar_list)]
        
    return Newdf     

def block_image_label_gen(img_dir, spatial_resolution = None, img_size = None, offset = None,  cultivar_list = None, block_list = None, year_list = None):

    if block_list is None: 
                block_list = ['LIV_003_2016','LIV_003_2017','LIV_003_2018','LIV_003_2019',
                    'LIV_004_2016','LIV_004_2017','LIV_004_2018','LIV_004_2019',
                    'LIV_005_2016','LIV_005_2017','LIV_005_2018', 
                    'LIV_006_2016','LIV_006_2017','LIV_006_2018','LIV_006_2019',
                    'LIV_007_2016','LIV_007_2017','LIV_007_2018','LIV_007_2019',
                    'LIV_008_2016','LIV_008_2017','LIV_008_2018', 
                    'LIV_009_2016','LIV_009_2017','LIV_009_2018','LIV_009_2019',  
                    'LIV_010_2016','LIV_010_2017','LIV_010_2018',
                    'LIV_011_2016','LIV_011_2017','LIV_011_2018','LIV_011_2019', 
                    'LIV_012_2016','LIV_012_2017','LIV_012_2018','LIV_012_2019',
                    'LIV_013_2016','LIV_013_2017','LIV_013_2018','LIV_013_2019',
                    'LIV_014_2016','LIV_014_2017','LIV_014_2018','LIV_014_2019',
                    'LIV_016_2016','LIV_016_2017','LIV_016_2018','LIV_016_2019',
                    'LIV_017_2016','LIV_017_2017','LIV_017_2018',
                    'LIV_018_2016','LIV_018_2017','LIV_018_2018','LIV_018_2019', 
                    'LIV_019_2016','LIV_019_2017','LIV_019_2018','LIV_019_2019',
                    'LIV_021_2017','LIV_021_2018','LIV_021_2019',
                    'LIV_022_2016','LIV_022_2017','LIV_022_2018',
                    'LIV_025_2016','LIV_025_2017','LIV_025_2018','LIV_025_2019', 
                    'LIV_028_2016','LIV_028_2017','LIV_028_2018','LIV_028_2019',
                    'LIV_032_2016','LIV_032_2017','LIV_032_2018', 
                    'LIV_038_2016','LIV_038_2017','LIV_038_2018', 
                    'LIV_050_2016','LIV_050_2017','LIV_050_2018','LIV_050_2019',
                    'LIV_058_2016','LIV_058_2017','LIV_058_2018', 
                    'LIV_061_2016','LIV_061_2017','LIV_061_2018','LIV_061_2019',
                    'LIV_062_2016','LIV_062_2017','LIV_062_2018', 
                    'LIV_063_2016','LIV_063_2017','LIV_063_2018','LIV_063_2019', 
                    'LIV_066_2016','LIV_066_2017','LIV_066_2019',
                    'LIV_068_2016','LIV_068_2017',
                    'LIV_070_2016','LIV_070_2017','LIV_070_2018',
                    'LIV_073_2016','LIV_073_2017','LIV_073_2018', 
                    'LIV_076_2016','LIV_076_2017','LIV_076_2018','LIV_076_2019', 
                    'LIV_077_2016','LIV_077_2017','LIV_077_2018','LIV_077_2019', 
                    'LIV_089_2016','LIV_089_2017','LIV_089_2018','LIV_089_2019', 
                    'LIV_090_2016','LIV_090_2017','LIV_090_2018','LIV_090_2019',
                    'LIV_102_2016','LIV_102_2017','LIV_102_2018','LIV_102_2019', 
                    'LIV_103_2016','LIV_103_2017','LIV_103_2018','LIV_103_2019', 
                    'LIV_105_2016','LIV_105_2017','LIV_105_2018','LIV_105_2019', 
                    'LIV_107_2016','LIV_107_2017','LIV_107_2018','LIV_107_2019', 
                    'LIV_111_2016','LIV_111_2017','LIV_111_2018','LIV_111_2019', 
                    'LIV_114_2016','LIV_114_2017','LIV_114_2018','LIV_114_2019',
                    'LIV_123_2016','LIV_123_2017','LIV_123_2018','LIV_123_2019',
                    'LIV_125_2016','LIV_125_2017','LIV_125_2018','LIV_125_2019',
                    'LIV_128_2016','LIV_128_2017','LIV_128_2018','LIV_128_2019', 
                    'LIV_135_2016','LIV_135_2017','LIV_135_2018','LIV_135_2019', 
                    'LIV_136_2016','LIV_136_2017','LIV_136_2018','LIV_136_2019',
                    'LIV_172_2016','LIV_172_2017','LIV_172_2018','LIV_172_2019',
                    'LIV_176_2016','LIV_176_2017','LIV_176_2018','LIV_176_2019',
                    'LIV_177_2016','LIV_177_2017','LIV_177_2018','LIV_177_2019',
                    'LIV_178_2016','LIV_178_2017','LIV_178_2018','LIV_178_2019',
                    'LIV_181_2016','LIV_181_2017','LIV_181_2018','LIV_181_2019',
                    'LIV_182_2016','LIV_182_2017','LIV_182_2018','LIV_182_2019', 
                    'LIV_186_2016','LIV_186_2017','LIV_186_2018','LIV_186_2019', 
                    'LIV_193_2016','LIV_193_2017','LIV_193_2018','LIV_193_2019']
    block_df_list = []
    block_list_imgs_labels = {}
    if year_list is None: 
        year_list = ['2016', '2017', '2018', '2019']

    for year in year_list: 

        LIV_year = 'LIV' +  year


        image_dir   = os.path.join(img_dir + LIV_year + '/image/Sentinel2/')
        image_names = os.listdir(image_dir)
        image_names.sort()

        label_dir   = os.path.join(img_dir+ LIV_year+'/yield/')
        label_names = os.listdir(label_dir)
        label_names.sort()


        if year == '2016':
            # because the format of 2016 is not tif!
            format = ".asc"
        else:
            format = ".tif"

        for name in label_names: 
            if name.endswith(format):
                name_split = os.path.split(name)[-1]
                name_root = name_split.replace(name_split[-4:], '')

                if name_root in block_list:
                    label_file_name = label_dir + '/' + name  

                    label_src = rasterio.open(label_file_name)
                    label_data = label_src.read()
                    label_data = np.moveaxis(label_data, 0, 2)
                    # convert all the non values including the background to -1 
                    label_data[label_data<0] = -1 
                    # adding one more dimension to be able for concatenation 
                    h = label_src.height # hight of label image
                    w = label_src.width  # width of label image 
                    slice_ = (slice(0, h), slice(0, w))
                    window_slice = windows.Window.from_slices(*slice_)
                    #print(window_slice)
                    # Window to list of index (row,col) 
                    pol_index = to_index(window_slice)
                    # Convert list of index (row,col) to list of coordinates of lat and long 
                    pol = [list(label_src.transform*p) for p in reverse_coordinates(pol_index)]
                    roi_polygon_src_coords = warp.transform_geom(label_src.crs,
                                                    {'init': 'epsg:32610'},                                          
                                                    {"type": "Polygon",
                                                    "coordinates": [pol]})

                    # update the label_data based on desired spatial resolution, also addign dimension to the image for concatenation: 
                    new_label_data, new_h, new_w = label_data_gen(label_data, spatial_resolution)

                    # generating timeseries sentinel 2 images after cropping, normalizing between (0, 255):  
                    timeseries_image = sentinel_timeseries_crop_gen(image_names, image_dir, roi_polygon_src_coords, spatial_resolution, new_h, new_w)

                    block_list_imgs_labels[name_root] = [timeseries_image, new_label_data]


                    # creating csv file dataframe for each block of all possible patches: 
                    block_df = blocklevel_csv_gen(name_root, new_label_data, img_size = img_size, offset = offset, cultivar_list = cultivar_list)
                    block_df_list.append(block_df)

    df = pd.concat(block_df_list)

    return block_list_imgs_labels, df



def npy_csv_generator(scenario = None, spatial_resolution = None, img_size=None, offset = None, cultivar_list = None, block_list = None,  year_list = None): 
    

        
    img_dir = '/data2/hkaman/Livingston/LIV_tif/'

    block_list_imgs_labels, df = block_image_label_gen(img_dir, spatial_resolution = spatial_resolution, img_size = img_size, 
                                                        offset = offset, 
                                                        cultivar_list = cultivar_list,
                                                        block_list = block_list,
                                                        year_list = year_list)


    df, df1, df2, df3 = block_cultivar_mean_std(df)

    split_obj = split_train_valid_test(df = df, scenario = scenario, year_list = year_list)
    train, val, test = split_obj.__train_valid_test__()
                    

    print(f"Training Patches: {len(train)}, Validation: {len(val)} and Test: {len(test)}")
    #print("============================= Train =========================================")
    #_ = print_df_summary(train)
    #print("============================= Validation ====================================")
    #_ = print_df_summary(val)
    #print("============================= Test ==========================================")
    #_ = print_df_summary(test)
    #print("=============================================================================")


    return train, val, test, block_list_imgs_labels

def print_df_summary(df):
    cultivars = df.groupby(by = 'cultivar')
    print(f"There are {len(cultivars)} cultivar types.")

    for cul, df in cultivars:
        
        blocks_within_cultivar = df.groupby(by = 'block')
        print(f"Cultivar {cul} has {len(blocks_within_cultivar)} blocks and {df.shape[0]} samples:")
        for b, df2 in blocks_within_cultivar:
            print(f"     Block {b} has {df2.shape[0]} samples.")


def block_npy_gen(data_dir, img_save_dir, label_save_dir, year = None, spatial_res = None):
    LIV_year = 'LIV' + year

    image_dir = os.path.join(data_dir + LIV_year+ '/image/Sentinel2/')
    image_names = os.listdir(image_dir)
    image_names.sort()

    label_dir = os.path.join(data_dir+ LIV_year+'/yield/')
    label_names = os.listdir(label_dir)
    label_names.sort()


    output_list = []

    if year == '2016':
        # because the format of 2016 is not tif!
        format = ".asc"
    else:
        format = ".tif"


    for name in label_names: 
        if name.endswith(format):
            label_file_name = label_dir + '/' + name  
            label_src = rasterio.open(label_file_name)
            label_data = label_src.read()
            label_data = np.moveaxis(label_data, 0, 2)
            # convert all the non values including the background to -1 
            label_data[label_data<0] = -1 
            # adding one more dimension to be able for concatenation 
            
            #print(label_data.shape)

            h = label_src.height # hight of label image
            w = label_src.width  # width of label image 
            slice_ = (slice(0, h), slice(0, w))
            window_slice = windows.Window.from_slices(*slice_)
            # Window to list of index (row,col) 
            pol_index = to_index(window_slice)

            # Convert list of index (row,col) to list of coordinates of lat and long 
            pol = [list(label_src.transform*p) for p in reverse_coordinates(pol_index)]
            
            roi_polygon_src_coords = warp.transform_geom(label_src.crs,
                                            {'init': 'epsg:32610'},                                          
                                            {"type": "Polygon",
                                            "coordinates": [pol]})
            

            if spatial_res == 1: 
                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)

                timeseries_image = None

                for img in image_names:
                    img_name =  image_dir + '/'+ img
                    # crop the sentinel image based on the corner of block shapefile corrdinates
                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    # sentinel image normalization! 
                    image_cropped = sentinel_image_stretch(image_cropped) 
                    # change the order of channels.
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    # convert nan to zero
                    image_cropped = np.nan_to_num(image_cropped)
                    # adding one more dimension
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
   
                    
                    up10x_img = cv2.resize(image_cropped, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)
                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 
                #print(np.sum(np.array(up10x_img) == 0)

            elif spatial_res == 10: 
                # aggregate the yeild map to 10m resolution 
                label_data = cv2.resize(label_data, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)

                timeseries_image = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)
                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped = image_cropped[0:h, 0:w,:]
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)
                    if timeseries_image is None:
                        timeseries_image = image_cropped_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, image_cropped_ex), axis = 0)

            elif spatial_res == 20: 

                label_data = cv2.resize(label_data, None, fx = 0.05, fy = 0.05, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)


                timeseries_image = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)

                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)


                    up10x_img = cv2.resize(image_cropped, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)


                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 

            elif spatial_res == 40: 

                label_data = cv2.resize(label_data, None, fx = 0.025, fy = 0.025, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)
                
                
                timeseries_image = None
                img_concate = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)

                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)

                    up10x_img = cv2.resize(image_cropped, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)
                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 




            #print(f"{label_data.shape}|{timeseries_image.shape}")

            name_split = os.path.split(name)[-1]
            name_root = name_split.replace(name_split[-4:], '')
            new_img_name = name_root + '_img.npy'
            img_npy_name = img_save_dir + new_img_name
            _ = saveList(timeseries_image, img_npy_name)

            new_label_name = name_root + '_label.npy'
            label_npy_name = label_save_dir + new_label_name
            _ = saveList(label_data, label_npy_name) 

#==============================================================================================#
#====================================  N                         ==============================#
#==============================================================================================#

def tranfer_file(input_list, filter_list, src_dir, dst_dir):
    for f in input_list:
        
        if f in filter_list:
            src = src_dir+f
            dst = dst_dir+f
            shutil.move(src,dst)
            
            
def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename, myList)
    #print("Saved successfully!")

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename, allow_pickle=True)
    return tempNumpyArray.tolist()



#==============================================================================================#
#====================================       Evaluation           ==============================#
#==============================================================================================#


def npy_block_names(npy_array):

    block_names = []
    for i in range(len(npy_array)):
        blocks = npy_array[i]['block']
        block_names.append(blocks)

    np_arr = np.concatenate(block_names)
    out = np.unique(np_arr)
    
    return out


def time_series_eval_csv(pred_npy, blocks_list, wsize = None):
    #blocks_list = get_blocks_from_patches(pred_npy)
    
    OutDF = pd.DataFrame()
    out_ytrue, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], []
    out_ypred_w1, out_ypred_w2,out_ypred_w3,out_ypred_w4,out_ypred_w5 = [], [], [], [], []
    out_ypred_w6, out_ypred_w7,out_ypred_w8,out_ypred_w9,out_ypred_w10 = [], [], [], [], []
    out_ypred_w11, out_ypred_w12,out_ypred_w13,out_ypred_w14,out_ypred_w15 = [], [], [], [], []
    
    for block in blocks_list:  
        
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name  = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id   = root_name
        
        res           = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d        = res.get(block_name)
        cultivar_id   = list_d[1]

        
        for l in range(len(pred_npy)):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            if len(tb_pred_indices) !=0:   
                for index in tb_pred_indices:

                    x0                = pred_npy[l]['X'][index]
                    y0                = pred_npy[l]['Y'][index]
                    x_vector, y_vector = xy_vector_generator(x0, y0, wsize)
                    out_x.append(x_vector)
                    out_y.append(y_vector)
       
                    tb_ytrue         = pred_npy[l]['ytrue'][index]
                    tb_flatten_ytrue = tb_ytrue.flatten()
                    out_ytrue.append(tb_flatten_ytrue)
                    

                    tb_ypred_w1    = pred_npy[l]['ypred_w1'][index]
                    tb_flatten_ypred_w1 = tb_ypred_w1.flatten()
                    out_ypred_w1.append(tb_flatten_ypred_w1)
                    
                    tb_ypred_w2    = pred_npy[l]['ypred_w2'][index]
                    tb_flatten_ypred_w2 = tb_ypred_w2.flatten()
                    out_ypred_w2.append(tb_flatten_ypred_w2)
                    
                    tb_ypred_w3    = pred_npy[l]['ypred_w3'][index]
                    tb_flatten_ypred_w3 = tb_ypred_w3.flatten()
                    out_ypred_w3.append(tb_flatten_ypred_w3)
                    
                    tb_ypred_w4    = pred_npy[l]['ypred_w4'][index]
                    tb_flatten_ypred_w4 = tb_ypred_w4.flatten()
                    out_ypred_w4.append(tb_flatten_ypred_w4)
                    
                    tb_ypred_w5    = pred_npy[l]['ypred_w5'][index]
                    tb_flatten_ypred_w5 = tb_ypred_w5.flatten()
                    out_ypred_w5.append(tb_flatten_ypred_w5)

                    tb_ypred_w6    = pred_npy[l]['ypred_w6'][index]
                    tb_flatten_ypred_w6 = tb_ypred_w6.flatten()
                    out_ypred_w6.append(tb_flatten_ypred_w6)
                    
                    tb_ypred_w7    = pred_npy[l]['ypred_w7'][index]
                    tb_flatten_ypred_w7 = tb_ypred_w7.flatten()
                    out_ypred_w7.append(tb_flatten_ypred_w7)
                    
                    tb_ypred_w8    = pred_npy[l]['ypred_w8'][index]
                    tb_flatten_ypred_w8 = tb_ypred_w8.flatten()
                    out_ypred_w8.append(tb_flatten_ypred_w8)
                    
                    tb_ypred_w9    = pred_npy[l]['ypred_w9'][index]
                    tb_flatten_ypred_w9 = tb_ypred_w9.flatten()
                    out_ypred_w9.append(tb_flatten_ypred_w9)
                    
                    tb_ypred_w10    = pred_npy[l]['ypred_w10'][index]
                    tb_flatten_ypred_w10 = tb_ypred_w10.flatten()
                    out_ypred_w10.append(tb_flatten_ypred_w10)
                    
                    tb_ypred_w11    = pred_npy[l]['ypred_w11'][index]
                    tb_flatten_ypred_w11 = tb_ypred_w11.flatten()
                    out_ypred_w11.append(tb_flatten_ypred_w11)
                    
                    tb_ypred_w12    = pred_npy[l]['ypred_w12'][index]
                    tb_flatten_ypred_w12 = tb_ypred_w12.flatten()
                    out_ypred_w12.append(tb_flatten_ypred_w12)
                    
                    tb_ypred_w13    = pred_npy[l]['ypred_w13'][index]
                    tb_flatten_ypred_w13 = tb_ypred_w13.flatten()
                    out_ypred_w13.append(tb_flatten_ypred_w13)
                    
                    tb_ypred_w14    = pred_npy[l]['ypred_w14'][index]
                    tb_flatten_ypred_w14 = tb_ypred_w14.flatten()
                    out_ypred_w14.append(tb_flatten_ypred_w14)
                    
                    tb_ypred_w15    = pred_npy[l]['ypred_w15'][index]
                    tb_flatten_ypred_w15 = tb_ypred_w15.flatten()
                    out_ypred_w15.append(tb_flatten_ypred_w15)
                    
                    tb_block_id   = np.array(len(tb_flatten_ytrue)*[block_id], dtype=np.int32)
                    out_blocks.append(tb_block_id)

                    tb_cultivar_id = np.array(len(tb_flatten_ytrue)*[cultivar_id], dtype=np.int8)
                    out_cultivars.append(tb_cultivar_id)


                    
    # agg    
    out_blocks        = np.concatenate(out_blocks)
    out_cultivars     = np.concatenate(out_cultivars)
    out_x             = np.concatenate(out_x)
    out_y             = np.concatenate(out_y)
    out_ytrue         = np.concatenate(out_ytrue)
    out_ypred_w1         = np.concatenate(out_ypred_w1)
    out_ypred_w2         = np.concatenate(out_ypred_w2)
    out_ypred_w3         = np.concatenate(out_ypred_w3)
    out_ypred_w4         = np.concatenate(out_ypred_w4)
    out_ypred_w5         = np.concatenate(out_ypred_w5)
    out_ypred_w6         = np.concatenate(out_ypred_w6)
    out_ypred_w7         = np.concatenate(out_ypred_w7)
    out_ypred_w8         = np.concatenate(out_ypred_w8)
    out_ypred_w9         = np.concatenate(out_ypred_w9)
    out_ypred_w10         = np.concatenate(out_ypred_w10)
    out_ypred_w11         = np.concatenate(out_ypred_w11)
    out_ypred_w12         = np.concatenate(out_ypred_w12)
    out_ypred_w13         = np.concatenate(out_ypred_w13)
    out_ypred_w14         = np.concatenate(out_ypred_w14)
    out_ypred_w15         = np.concatenate(out_ypred_w15)
    
    OutDF['block']    = out_blocks
    OutDF['cultivar'] = out_cultivars
    OutDF['x']        = out_x
    OutDF['y']        = out_y
    OutDF['ytrue']    = out_ytrue
    OutDF['ypred_w1']    = out_ypred_w1
    OutDF['ypred_w2']    = out_ypred_w2
    OutDF['ypred_w3']    = out_ypred_w3
    OutDF['ypred_w4']    = out_ypred_w4
    OutDF['ypred_w5']    = out_ypred_w5
    OutDF['ypred_w6']    = out_ypred_w6
    OutDF['ypred_w7']    = out_ypred_w7
    OutDF['ypred_w8']    = out_ypred_w8
    OutDF['ypred_w9']    = out_ypred_w9
    OutDF['ypred_w10']    = out_ypred_w10
    OutDF['ypred_w11']    = out_ypred_w11
    OutDF['ypred_w12']    = out_ypred_w12
    OutDF['ypred_w13']    = out_ypred_w13
    OutDF['ypred_w14']    = out_ypred_w14
    OutDF['ypred_w15']    = out_ypred_w15
    
    NewOUtDF = agg_pixelovelapp_df_2d(OutDF)
    
    return OutDF, NewOUtDF
    #return OutDF





def block_cultivar_level_csv(exp_name = None, spatial_resolution = None, week = None, save_csv_name = None):

    if spatial_resolution == 1: 
        patch_size = 80
    elif spatial_resolution == 10:
        patch_size = 16


    #input_data           = np.load(os.path.join('/data2/hkaman/Livingston/EXPs/', 'EXP_' + exp_name, exp_name + '_test_bc.npy'), allow_pickle=True)
    #block_names          = npy_block_names(input_data)
    #BC_df1, BC_df2       = time_series_eval_csv(input_data, block_names, patch_size)

    BC_df2     = pd.read_csv(os.path.join('/data2/hkaman/Livingston/EXPs/1m', 'EXP_' + exp_name, exp_name + '_test_bc.csv'))
    blocks     = BC_df2.groupby(by = ['block'])

    block_df    = pd.DataFrame()
    b, c, test_r2_b, test_mae_b, test_rmse_b, test_mape_b = [], [], [], [], [], [] 
    
    cultivar_df = pd.DataFrame()
    cu, test_r2_c, test_mae_c, test_rmse_c, test_mape_c   = [], [], [], [], []
        
    for block, tedf in blocks:
        b.append(block)
        cultivar_id     = tedf.iloc[0]['cultivar']
        cultivar_id     = int(cultivar_id)
        cultivar_id     = str(cultivar_id)
        key_ext         = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d          = key_ext.get(cultivar_id)
        cultivar_type   = list_d[0]
        
        c.append(cultivar_type)
        b_r_square3, b_mae3, b_rmse3, b_mape3, mean_ytrue, mean_ypred =  regression_metrics(tedf['ytrue'], tedf[week])
        test_r2_b.append(b_r_square3)
        test_mae_b.append(b_mae3)
        test_rmse_b.append(b_rmse3)
        test_mape_b.append(b_mape3)
        
        
    block_df['block']    = b
    block_df['cultivar'] = c
    block_df['R2']  = test_r2_b
    block_df['MAE']  = test_mae_b
    block_df['RMSE'] = test_rmse_b
    block_df['MAPE'] = test_mape_b
    block_df = block_df.round(decimals = 2)
    #block_df = block_df.sort_values(by = ['cultivar', 'block']).reset_index(drop=True)
    block_df.to_csv(save_csv_name + '_blocks.csv')
    
    ### Cultvars 
    test_cultivars  = BC_df2.groupby(by = ['cultivar'])
    for cul, tedf1 in test_cultivars:
        cultivar_id   = int(cul)
        cultivar_id   = str(cultivar_id)
        key_ext       = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d        = key_ext.get(cultivar_id)
        cultivar_type = list_d[0]
        
        cu.append(cultivar_type)
        b_r_square33, b_mae33, b_rmse33, b_mape33, mean_ytrue, mean_ypred =  regression_metrics(tedf1['ytrue'], tedf1[week])
        test_r2_c.append(b_r_square33)
        test_mae_c.append(b_mae33)
        test_rmse_c.append(b_rmse33)
        test_mape_c.append(b_mape33)

    cultivar_df['cultivar'] = cu
    cultivar_df['R2']  = test_r2_c
    cultivar_df['MAE']  = test_mae_c
    cultivar_df['RMSE'] = test_rmse_c
    cultivar_df['MAPE'] = test_mape_c
    cultivar_df = cultivar_df.round(decimals = 2)
    #cultivar_df = cultivar_df.sort_values(by = ['cultivar', 'block']).reset_index(drop=True)
    cultivar_df.to_csv(save_csv_name + '_cultivars.csv')
    
    return block_df, cultivar_df













def block_cultivar_extract(cultivar_list, pix_n):
    #outDF = pd.DataFrame(columns = ['CID', 'ytrue', 'ypred'])
    out_cultivars = []
    Patch_size = len(cultivar_list)

    for l in range(Patch_size):
            
        thisP_cultivar= cultivar_list[l]
        name_split    = os.path.split(thisP_cultivar)[-1]
        block_name    = name_split.replace(name_split[7:], '')
        res           = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d        = res.get(block_name)
        block_id      = list_d[1]
        cultivars     = np.array(pix_n*[block_id], dtype=np.int8)
       
        out_cultivars.append(cultivars)
    out_cultivars = np.concatenate(out_cultivars).astype(np.uint8)
    return out_cultivars

    
def resize_byindex(src, scale):
    out = None 
    this_image = src[0,:,:,0]
    this_image_inter = cv2.resize(this_image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
    this_image_inter = np.expand_dims(this_image_inter, axis  = 0)
    this_image_inter = np.expand_dims(this_image_inter, axis  = -1)

    return this_image_inter 

def get_blocks_from_patches(pred_npy):
    out_list = []
    for l in range(len(pred_npy)):
        list_to_array = np.array(pred_npy[l]['block'])
        list_uniqes = np.unique(list_to_array)
        out_list.append(list_uniqes)
    out_list = np.concatenate(out_list)
    final_out = np.unique(out_list)
    return final_out


def blocks_mean_std_info(df):
    
    block_groups = df.groupby(by = 'block')
    new_info = pd.DataFrame()
    blocks, blocks_mean, cul, blocks_std, years, new_mean = [], [], [], [], [], []

    for block_id, block_df in block_groups:
        blocks.append(block_id)
        years.append(block_df['year'].iloc[0])
        cul.append(block_df['cultivar'].iloc[0])
        blocks_mean.append(block_df['block_mean'].iloc[0])
        blocks_std.append(block_df['block_std'].iloc[0])
        #new_mean.append(block_df['patch_mean'] -  block_df['block_mean']) / (block_df['block_std'])

    new_info['block'] = blocks
    new_info['cultivar'] = cul
    new_info['year'] = years
    new_info['mean'] =blocks_mean
    new_info['std'] =blocks_std
        
    return new_info


def patch_meanstd_extract(cultivar_name, pix_n = None, norm_type = None):

    means, stds, mins, maxs = [], [], [], []

    mean = Cultivars_info[cultivar_name][0]
    std  = Cultivars_info[cultivar_name][1]
    min_ = Cultivars_info[cultivar_name][2]
    max_ = Cultivars_info[cultivar_name][3]

    means  = np.array(pix_n*[mean])
    stds   = np.array(pix_n*[std])
    mins   = np.array(pix_n*[min_])
    maxs   = np.array(pix_n*[max_])
    
    
    if norm_type == 'N': 
        return mins, maxs
    elif norm_type == 'S': 
        return means, stds
    
def block_meanstd_extract(df, block_name, pix_n = None):

    means, stds, mins, maxs = [], [], [], []

    #all_blocks_mean_info = block_level_normalization()
    this_block_info = df.loc[df['block'] == block_name] 
    mean            = this_block_info['mean']
    std             = this_block_info['std']
    means           = np.array(pix_n*[mean]).flatten()
    stds            = np.array(pix_n*[std]).flatten()

    return means, stds
    
def xy_vector_generator(x0, y0):
    x_vector, y_vector = [], []
    
    for i in range(x0, x0+80):
        for j in range(y0, y0+80):
            x_vector.append(i)
            y_vector.append(j)

    return x_vector, y_vector 


def inverse_normalization(mean, std, block_names, cultivar_names, ytrue, ypred, scale):
    
 
    flatten_ytrue = ytrue.flatten()
    mean_vector   = np.array(len(flatten_ytrue)*[mean])
    std_vector    = np.array(len(flatten_ytrue)*[std])

    flatten_ytrue = (flatten_ytrue * std_vector) + mean_vector
    reshape_ytrue = np.reshape(flatten_ytrue, (scale, scale))
    reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = 0)
    reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = -1)
    
    flatten_ypred = ypred.flatten()
    flatten_ypred = (flatten_ypred * std_vector) + mean_vector
    reshape_ypred = np.reshape(flatten_ypred, (scale, scale))
    reshape_ypred = np.expand_dims(reshape_ypred, axis  = 0)
    reshape_ypred = np.expand_dims(reshape_ypred, axis  = -1)
    
    return reshape_ytrue, reshape_ypred 


def ms_scenario_evaluation(pred_npy, blocks_list, block_normalization = None, scale = None):
    
    OutDF = pd.DataFrame()
    out_ytrue, out_ypred, out_blocks, out_cultivars, out_x, out_y = [], [], [], [], [], []
    
    for block in blocks_list:  
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name  = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id   = root_name
        
        res           = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d        = res.get(block_name)
        cultivar_id   = list_d[1]

        
        for l in range(len(pred_npy)):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            if len(tb_pred_indices) !=0:   
                for index in tb_pred_indices:
                    
                    block_key = next(iter(pred_npy[l]))
                    tb_block_name     = pred_npy[l][block_key][index]
                    b_name_split      = os.path.split(tb_block_name)[-1]
                    tb_block_root_name= b_name_split.replace(b_name_split[12:], '')
                    
                    tb_block_cultivar = pred_npy[l]['cultivar'][index]

                    x0                = pred_npy[l]['X'][index]
                    y0                = pred_npy[l]['Y'][index]
                    
                    x_vector, y_vector = xy_vector_generator(x0, y0)
       
                    tb_ytrue80        = pred_npy[l]['ytrue'][index]
                    tb_ypred80        = pred_npy[l]['ypred80'][index]
                    
                    tb_ytrue40        = resize_byindex(tb_ytrue80, 0.5)
                    tb_ypred40        = pred_npy[l]['ypred40'][index]

                    tb_ytrue20        = resize_byindex(tb_ytrue80, 0.25)
                    tb_ypred20        = pred_npy[l]['ypred20'][index]
                    
                    
                    if block_normalization is True:
                        
                        tb_win_block_mean = pred_npy[l]['win_block_mean'][index]
                        tb_win_block_std  = pred_npy[l]['win_block_std'][index]
                        tb_ytrue80_inv, th_ypred80_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_block_cultivar, 
                                                                               tb_ytrue80, tb_ypred80, 80)
                        tb_ytrue40_inv, th_ypred40_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_block_cultivar, 
                                                                               tb_ytrue40, tb_ypred40, 40)
                        tb_ytrue20_inv, th_ypred20_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_block_cultivar,
                                                                               tb_ytrue20, tb_ypred20, 20)
                        
                    elif block_normalization is False: 
                        tb_ytrue80_inv, th_ypred80_inv = tb_ytrue80, tb_ypred80
                        tb_ytrue40_inv, th_ypred40_inv = tb_ytrue40, tb_ypred40
                        tb_ytrue20_inv, th_ypred20_inv = tb_ytrue20, tb_ypred20
                    
                    if scale == 20:
                        flat_tb_ytrue_inv = tb_ytrue20_inv.flatten()
                        out_ytrue.append(flat_tb_ytrue_inv)

                        flat_th_ypre_inv = th_ypred20_inv.flatten()
                        out_ypred.append(flat_th_ypred_inv)

                        tb_block_id = np.array(len(flat_th_ypred_inv)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id)

                        tb_cultivar_id = np.array(len(flat_th_ypred_inv)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id)
                        
                        out_x.append(np.array(len(flat_th_ypred_inv)*[1], dtype=np.int8))
                        out_y.append(np.array(len(flat_th_ypred_inv)*[1], dtype=np.int8))
                        
                    elif scale == 40:
                        flat_tb_ytrue_inv = tb_ytrue40_inv.flatten()
                        out_ytrue.append(flat_tb_ytrue_inv)

                        flat_th_ypre_inv = th_ypred40_inv.flatten()
                        out_ypred.append(flat_th_ypred_inv)

                        tb_block_id = np.array(len(flat_th_ypred_inv)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id)

                        tb_cultivar_id = np.array(len(flat_th_ypred_inv)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id)
                        
                        out_x.append(np.array(len(flat_th_ypred_inv)*[1], dtype=np.int8))
                        out_y.append(np.array(len(flat_th_ypred_inv)*[1], dtype=np.int8))
                        
                    elif scale == 80:
                        
                        flat_tb_ytrue_inv = tb_ytrue80_inv.flatten()
                        out_ytrue.append(flat_tb_ytrue_inv)

                        flat_th_ypre_inv = th_ypred80_inv.flatten()
                        out_ypred.append(flat_th_ypred_inv)

                        tb_block_id = np.array(len(flat_th_ypred_inv)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id)

                        tb_cultivar_id = np.array(len(flat_th_ypred_inv)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id)
                        
                        out_x.append(x_vector)
                        out_y.append(y_vector)
                        
                    elif scale == 'agg':
                        
                        tb_ypred40to80_inv = resize_byindex(th_ypred40_inv, 2)
                        tb_ypred20to80_inv = resize_byindex(th_ypred20_inv, 4)
                        tb_ypredagg_inv    = np.mean(np.array([th_ypred80_inv, tb_ypred40to80_inv, tb_ypred20to80_inv]), axis=0)
                        
                        tb_flatten_ytrue_agg = tb_ytrue80_inv.flatten()
                        out_ytrue.append(tb_flatten_ytrue_agg)
                    
                        tb_flatten_ypred_agg = tb_ypredagg_inv.flatten()
                        out_ypred.append(tb_flatten_ypred_agg)
                    
                        tb_block_id_agg = np.array(len(tb_flatten_ypred_agg)*[block_id], dtype=np.int32)
                        out_blocks.append(tb_block_id_agg)
                    
                        tb_cultivar_id_agg = np.array(len(tb_flatten_ypred_agg)*[cultivar_id], dtype=np.int8)
                        out_cultivars.append(tb_cultivar_id_agg)
                        
                        out_x.append(x_vector)
                        out_y.append(y_vector)

                    
    # agg    
    out_blocks        = np.concatenate(out_blocks)
    out_cultivars     = np.concatenate(out_cultivars)
    out_x             = np.concatenate(out_x)
    out_y             = np.concatenate(out_y)
    out_ytrue         = np.concatenate(out_ytrue)
    out_ypred         = np.concatenate(out_ypred)
    
    
    OutDF['block']    = out_blocks
    OutDF['cultivar'] = out_cultivars
    OutDF['x']        = out_x
    OutDF['y']        = out_y
    OutDF['ytrue']    = out_ytrue
    OutDF['ypred']    = out_ypred
    
    #NewOUtDF = agg_pixelovelapp_df(OutDF)
    
    return OutDF#, NewOUtDF

'''def agg_pixelovelapp_df(df):
    
    newdf = df.groupby(["block", "cultivar", "x", "y"]).agg(
        ytrue = pd.NamedAgg(column="ytrue", aggfunc=np.mean),
        ypred = pd.NamedAgg(column="ypred", aggfunc=np.mean),
    )
    return newdf'''

def agg_pixelovelapp_df(df):
    
    newdf = pd.DataFrame()
    block, cultivar, x, y, ytrue, ypred = [],[],[],[],[], []
    g = df.groupby(["block", "cultivar", "x", "y"])
    
    for l, gdf in g:
        block.append(gdf['block'].iloc[0])
        cultivar.append(gdf['cultivar'].iloc[0])
        x.append(gdf['x'].iloc[0])
        y.append(gdf['y'].iloc[0])
        ytrue.append(gdf['ytrue'].mean())
        ypred.append(gdf['ypred'].mean())
    
    newdf['block'] = block
    newdf['cultivar'] = cultivar
    newdf['x'] = x
    newdf['y'] = y
    newdf['ytrue'] = ytrue
    newdf['ypred'] =ypred

    return newdf




def block_cultivar_ms_eval(pred_npy, YS = None, norm_type = None):

    blocks_eval_dict = []
    
    OutDFAgg = pd.DataFrame(columns = ['block', 'cultivar', 'ytrue', 'ypred'])
    out_ytrue_agg, out_ypred_agg, out_blocks_agg, out_cultivars_agg = [], [], [], []
    
    OutDF80 = pd.DataFrame(columns = ['block', 'cultivar', 'ytrue', 'ypred'])
    out_ytrue_80, out_ypred_80, out_blocks_80, out_cultivars_80 = [], [], [], []
    
    OutDF40 = pd.DataFrame(columns = ['block', 'cultivar', 'ytrue', 'ypred'])
    out_ytrue_40, out_ypred_40, out_blocks_40, out_cultivars_40 = [], [], [], []
    
    OutDF20 = pd.DataFrame(columns = ['block', 'cultivar', 'ytrue', 'ypred'])
    out_ytrue_20, out_ypred_20, out_blocks_20, out_cultivars_20 = [], [], [], []

    
    for block in blocks:  
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name  = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id   = root_name
        
        res           = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d        = res.get(block_name)
        block_id      = list_d[1]
        cultivars     = np.array(pix_n*[block_id], dtype=np.int8)
        cultivar_id   = list_d[1]
        

        tb_pred_out = np.full((blocks_size[block][0], blocks_size[block][1]), -1) 
        tb_true_out = np.full((blocks_size[block][0], blocks_size[block][1]), -1) 
 
        
        for l in range(len(pred_npy)):
            #for b in range(len(data[l]['ID'])):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            
            if len(tb_pred_indices) !=0:   
                #print(f"{block}: {tb_pred_indices}")
                for index in tb_pred_indices:
                    
                    block_key = next(iter(pred_npy[l]))
                    tb_block_name = pred_npy[l][block_key][index]
                    b_name_split = os.path.split(tb_block_name)[-1]
                    tb_block_root_name = b_name_split.replace(b_name_split[12:], '')
                    
                    
                    x0          = pred_npy[l]['X'][index]
                    y0          = pred_npy[l]['Y'][index]
       
                    tb_ytrue80        = pred_npy[l]['ytrue'][index]
                    tb_ypred80        = pred_npy[l]['ypred80'][index]
                    
                    tb_ytrue40        = resize_byindex(tb_ytrue80, 0.5)
                    tb_ypred40        = pred_npy[l]['ypred40'][index]

                    tb_ytrue20        = resize_byindex(tb_ytrue80, 0.25)
                    tb_ypred20        = pred_npy[l]['ypred20'][index]
                    
                    tb_cultivar_name  = pred_npy[l]['cultivar'][index] 
                    
                    if YS is True: 
                        tb_win_block_mean = pred_npy[l]['win_block_mean'][index]
                        tb_win_block_std  = pred_npy[l]['win_block_std'][index]
                    
                    # 80
                    #print("stop 1!")
                    if YS is True: 
                        tb_ytrue80_inv, th_ypred80_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name, 
                                                                               tb_ytrue80, tb_ypred80, 80)
                    elif YS is False: 
                        tb_ytrue80_inv = tb_ytrue80
                        th_ypred80_inv = tb_ypred80
                    
                    flat_tb_ytrue80_inv = tb_ytrue80_inv.flatten()
                    out_ytrue_80.append(flat_tb_ytrue80_inv)
                    
                    flat_th_ypred80_inv = th_ypred80_inv.flatten()
                    out_ypred_80.append(flat_th_ypred80_inv)
                    
                    tb_block_id_80 = np.array(len(flat_th_ypred80_inv)*[block_id], dtype=np.int32)
                    out_blocks_80.append(tb_block_id_80)
                    
                    tb_cultivar_id_80 = np.array(len(flat_th_ypred80_inv)*[cultivar_id], dtype=np.int8)
                    out_cultivars_80.append(tb_cultivar_id_80)

                    # 40
                    if YS is True: 
                        tb_ytrue40_inv, th_ypred40_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name, 
                                                                               tb_ytrue40, tb_ypred40, 40)
                    elif YS is False: 
                        tb_ytrue40_inv, th_ypred40_inv = tb_ytrue40, tb_ypred40
                    
                    
                    flat_tb_ytrue40_inv = tb_ytrue40_inv.flatten()
                    out_ytrue_40.append(flat_tb_ytrue40_inv)
                    
                    flat_th_ypred40_inv = th_ypred40_inv.flatten()
                    out_ypred_40.append(flat_th_ypred40_inv)
                    
                    tb_block_id_40 = np.array(len(flat_th_ypred40_inv)*[block_id], dtype=np.int32)
                    out_blocks_40.append(tb_block_id_40)
                    
                    tb_cultivar_id_40 = np.array(len(flat_th_ypred40_inv)*[cultivar_id], dtype=np.int8)
                    out_cultivars_40.append(tb_cultivar_id_40)
                    
                    # 20
                    if YS is True: 
                        tb_ytrue20_inv, th_ypred20_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name,
                                                                               tb_ytrue20, tb_ypred20, 20)
                    elif YS is False:   
                        tb_ytrue20_inv, th_ypred20_inv = tb_ytrue20, tb_ypred20
                    
                    
                    flat_tb_ytrue20_inv = tb_ytrue20_inv.flatten()
                    out_ytrue_20.append(flat_tb_ytrue20_inv)
                    
                    flat_th_ypred20_inv = th_ypred20_inv.flatten()
                    out_ypred_20.append(flat_th_ypred20_inv)
                    
                    tb_block_id_20 = np.array(len(flat_th_ypred20_inv)*[block_id], dtype=np.int32)
                    out_blocks_20.append(tb_block_id_20)
                    
                    tb_cultivar_id_20 = np.array(len(flat_th_ypred20_inv)*[cultivar_id], dtype=np.int8)
                    out_cultivars_20.append(tb_cultivar_id_20)
                    
                    # aggregate
                    tb_ypred40to80_inv = resize_byindex(th_ypred40_inv, 2)
                    tb_ypred20to80_inv = resize_byindex(th_ypred20_inv, 4)
                    tb_ypredagg_inv    = np.mean(np.array([th_ypred80_inv, tb_ypred40to80_inv, tb_ypred20to80_inv]), axis=0)
                    
                    tb_flatten_ytrue_agg = tb_ytrue80_inv.flatten()
                    out_ytrue_agg.append(tb_flatten_ytrue_agg)
                    
                    tb_flatten_ypred_agg = tb_ypredagg_inv.flatten()
                    out_ypred_agg.append(tb_flatten_ypred_agg)
                    
                    tb_block_id_agg = np.array(len(tb_flatten_ypred_agg)*[block_id], dtype=np.int32)
                    out_blocks_agg.append(tb_block_id_agg)
                    
                    tb_cultivar_id_agg = np.array(len(tb_flatten_ypred_agg)*[cultivar_id], dtype=np.int8)
                    out_cultivars_agg.append(tb_cultivar_id_agg)
                    

                    
                    
                    if np.all((tb_pred_out[x0:x0+80, y0:y0+80] == -1)):
                        tb_pred_out[x0:x0+80, y0:y0+80] = tb_ypredagg_inv[0,:,:,0]
                    else: 
                        tb_pred_out[x0:x0+80, y0:y0+80] = (tb_pred_out[x0:x0+80, y0:y0+80] + tb_ypredagg_inv[0,:,:,0])/2

                    if np.all((tb_true_out[x0:x0+80, y0:y0+80] == -1)):
                        tb_true_out[x0:x0+80, y0:y0+80] = tb_ytrue80_inv[0,:,:,0]
                    else: 
                        tb_true_out[x0:x0+80, y0:y0+80] = (tb_true_out[x0:x0+80, y0:y0+80] + tb_ytrue80_inv[0,:,:,0])/2
                        #print("loop is done!")
        #print("here!")
        # creating a dictionary for all the block for future visualzation: 
        this_block_dict  = {"block": block, "true_mtx": tb_true_out, "pred_mtx": tb_pred_out}
        blocks_eval_dict.append(this_block_dict)
        
        # aggregation dataframe: 
        #tb_flatten_ytrue_agg = tb_true_out.flatten()
        #tb_flatten_ytrue_agg = tb_flatten_ytrue_agg[tb_flatten_ytrue_agg != -1]
        #out_ytrue_agg.append(tb_flatten_ytrue_agg)
        
        #tb_flatten_ypred_agg = tb_pred_out.flatten()
        #tb_flatten_ypred_agg = tb_flatten_ypred_agg[tb_flatten_ypred_agg != -1]            
        #out_ypred_agg.append(tb_flatten_ypred_agg)
        
        #tb_block_id_agg    = np.array(len(tb_flatten_ypred_agg)*[block_id], dtype=np.int32)
        #out_blocks_agg.append(tb_block_id_agg)
        
        #tb_cultivar_id_agg = np.array(len(tb_flatten_ypred_agg)*[cultivar_id], dtype=np.int8)
        #out_cultivars_agg.append(tb_cultivar_id_agg)
        

    # agg    
    out_blocks_agg       = np.concatenate(out_blocks_agg)
    out_cultivars_agg    = np.concatenate(out_cultivars_agg)
    out_ytrue_agg        = np.concatenate(out_ytrue_agg)#.astype(np.float16)
    out_ypred_agg        = np.concatenate(out_ypred_agg)#.astype(np.float16) 
    #print(f"{out_blocks_agg.shape}|{out_cultivars_agg.shape}|{out_ytrue_agg.shape}|{out_ypred_agg.shape}")
    
    OutDFAgg['block']    = out_blocks_agg
    OutDFAgg['cultivar'] = out_cultivars_agg
    OutDFAgg['ytrue']    = out_ytrue_agg
    OutDFAgg['ypred']    = out_ypred_agg  
    
    # 80
    out_blocks_80       = np.concatenate(out_blocks_80)
    out_cultivars_80    = np.concatenate(out_cultivars_80)
    out_ytrue_80        = np.concatenate(out_ytrue_80)#.astype(np.float16)
    out_ypred_80        = np.concatenate(out_ypred_80)#.astype(np.float16) 
    OutDF80['block']    = out_blocks_80
    OutDF80['cultivar'] = out_cultivars_80
    OutDF80['ytrue']    = out_ytrue_80
    OutDF80['ypred']    = out_ypred_80 
    
    # 40
    out_blocks_40       = np.concatenate(out_blocks_40)
    out_cultivars_40    = np.concatenate(out_cultivars_40)
    out_ytrue_40        = np.concatenate(out_ytrue_40)#.astype(np.float16)
    out_ypred_40        = np.concatenate(out_ypred_40)#.astype(np.float16) 
    OutDF40['block']    = out_blocks_40
    OutDF40['cultivar'] = out_cultivars_40
    OutDF40['ytrue']    = out_ytrue_40
    OutDF40['ypred']    = out_ypred_40 
    # 20
    out_blocks_20       = np.concatenate(out_blocks_20)
    out_cultivars_20    = np.concatenate(out_cultivars_20)
    out_ytrue_20        = np.concatenate(out_ytrue_20)#.astype(np.float16)
    out_ypred_20        = np.concatenate(out_ypred_20)#.astype(np.float16) 
    OutDF20['block']    = out_blocks_20
    OutDF20['cultivar'] = out_cultivars_20
    OutDF20['ytrue']    = out_ytrue_20
    OutDF20['ypred']    = out_ypred_20 
    

    return OutDFAgg, OutDF80, OutDF40, OutDF20, blocks_eval_dict





def block_cultivar_test_csv_results(test_df, save_dir, save_csv_name):

    test_blocks     = test_df.groupby(by = ['block'])

    block_df    = pd.DataFrame(columns = ['block', 'cultivar', 'TestR^2', 'TestMAE',  'TestRMSE', 'TestMAPE'])
    b, c, test_r2_b, test_mae_b, test_rmse_b, test_mape_b = [], [], [], [], [], [] 
    
    cultivar_df = pd.DataFrame(columns = ['cultivar', 'TestR^2',  'TestMAE', 'TestRMSE', 'TestMAPE'])
    cu, test_r2_c, test_mae_c, test_rmse_c, test_mape_c   = [], [], [], [], []
        
    for block, tedf in test_blocks:
        b.append(block)
        cultivar_id     = tedf.iloc[0]['cultivar']
        cultivar_id     = int(cultivar_id)
        cultivar_id     = str(cultivar_id)
        key_ext         = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d          = key_ext.get(cultivar_id)
        cultivar_type   = list_d[0]
        
        c.append(cultivar_type)
        b_r_square3, b_mae3, b_rmse3, b_mape3, mean_ytrue, mean_ypred =  regression_metrics(tedf['ytrue'], tedf['ypred'])
        test_r2_b.append(b_r_square3)
        test_mae_b.append(b_mae3)
        test_rmse_b.append(b_rmse3)
        test_mape_b.append(b_mape3)
        
        
    block_df['block']    = b
    block_df['cultivar'] = c
    block_df['TestR^2']  = test_r2_b
    block_df['TestMAE']  = test_mae_b
    block_df['TestRMSE'] = test_rmse_b
    block_df['TestMAPE'] = test_mape_b
    block_df = block_df.round(decimals = 2)
    block_df.to_csv(os.path.join(save_dir, save_csv_name + '_blocks.csv'))
    
    ### Cultvars 
    test_cultivars  = test_df.groupby(by = ['cultivar'])
    for cul, tedf1 in test_cultivars:
        cultivar_id   = int(cul)
        cultivar_id   = str(cultivar_id)
        key_ext       = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d        = key_ext.get(cultivar_id)
        cultivar_type = list_d[0]
        
        cu.append(cultivar_type)
        b_r_square33, b_mae33, b_rmse33, b_mape33, mean_ytrue, mean_ypred =  regression_metrics(tedf1['ytrue'], tedf1['ypred'])
        test_r2_c.append(b_r_square33)
        test_mae_c.append(b_mae33)
        test_rmse_c.append(b_rmse33)
        test_mape_c.append(b_mape33)

    cultivar_df['cultivar'] = cu
    cultivar_df['TestR^2']  = test_r2_c
    cultivar_df['TestMAE']  = test_mae_c
    cultivar_df['TestRMSE'] = test_rmse_c
    cultivar_df['TestMAPE'] = test_mape_c
    cultivar_df = cultivar_df.round(decimals = 2)
    cultivar_df.to_csv(os.path.join(save_dir, save_csv_name + '_cultivars.csv'))


def cultivar_ms_eval(pred_npy, YS = None, norm_type = None):

    blocks_eval_dict = []
    
    OutDFAgg = pd.DataFrame(columns = ['block', 'cultivar', 'ytrue', 'ypred'])
    out_ytrue_agg, out_ypred_agg, out_blocks_agg, out_cultivars_agg = [], [], [], []
    
    for block in blocks_size:  
        name_split    = os.path.split(block)[-1]
        block_name    = name_split.replace(name_split[7:], '')
        root_name     = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id      = root_name
        
        res           = {key: Categ_[key] for key in Categ_.keys() & {block_name}}
        list_d        = res.get(block_name)
        cultivar_id   = list_d[1]


def get_blocks_from_patches(pred_npy):
    out_list = []
    for l in range(len(pred_npy)):
        list_to_array = np.array(pred_npy[l]['block'])
        list_uniqes = np.unique(list_to_array)
        out_list.append(list_uniqes)
    out_list = np.concatenate(out_list)
    final_out = np.unique(out_list)
    return final_out


def blocks_mean_std_info(df):
    
    block_groups = df.groupby(by = 'block')
    new_info = pd.DataFrame()
    blocks, blocks_mean, cul, blocks_std, years, new_mean = [], [], [], [], [], []

    for block_id, block_df in block_groups:
        blocks.append(block_id)
        years.append(block_df['year'].iloc[0])
        cul.append(block_df['cultivar'].iloc[0])
        blocks_mean.append(block_df['block_mean'].iloc[0])
        blocks_std.append(block_df['block_std'].iloc[0])
        #new_mean.append(block_df['patch_mean'] -  block_df['block_mean']) / (block_df['block_std'])

    new_info['block'] = blocks
    new_info['cultivar'] = cul
    new_info['year'] = years
    new_info['mean'] =blocks_mean
    new_info['std'] =blocks_std
        
    return new_info


def patch_meanstd_extract(cultivar_name, pix_n = None, norm_type = None):

    means, stds, mins, maxs = [], [], [], []

    mean = Cultivars_info[cultivar_name][0]
    std  = Cultivars_info[cultivar_name][1]
    min_ = Cultivars_info[cultivar_name][2]
    max_ = Cultivars_info[cultivar_name][3]

    means  = np.array(pix_n*[mean])
    stds   = np.array(pix_n*[std])
    mins   = np.array(pix_n*[min_])
    maxs   = np.array(pix_n*[max_])
    
    
    if norm_type == 'N': 
        return mins, maxs
    elif norm_type == 'S': 
        return means, stds
    
def block_meanstd_extract(df, block_name, pix_n = None):

    means, stds, mins, maxs = [], [], [], []

    #all_blocks_mean_info = block_level_normalization()
    this_block_info = df.loc[df['block'] == block_name] 
    mean            = this_block_info['mean']
    std             = this_block_info['std']
    means           = np.array(pix_n*[mean]).flatten()
    stds            = np.array(pix_n*[std]).flatten()

    return means, stds
    
def xy_vector_generator(x0, y0, wsize):
    x_vector, y_vector = [], []
    
    for i in range(x0, x0+wsize):
        for j in range(y0, y0+wsize):
            x_vector.append(i)
            y_vector.append(j)

    return x_vector, y_vector 

    


    
def scale_extraction(block_names, ytrue, ypred, scale):
    
    cultivars     = block_cultivar_extract(block_names, int(scale**2))
    flatten_ytrue = ytrue.flatten()
    flatten_ypred = ypred.flatten()
    
    return cultivars, flatten_ytrue, flatten_ypred 


def many_agg(src, scale): 
    out = []
    for l in range(src.shape[0]):
        this_image = src[l, 0,:,:,0]
        agg_img = aggregate(this_image, scale)
        flat_agg_img = agg_img.flatten()
        out.append(flat_agg_img)
    mtx = np.concatenate(out)
    return mtx
def aggregate(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.zeros((w, h))
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx



def resize_by_patch(src, scale):
    
    out = None
    for l in range(src.shape[0]): 
        this_image = src[l, 0,:,:,0]
        #this_image_inter = aggregate(this_image, scale)

        this_image_inter = cv2.resize(this_image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
        this_image_inter = np.expand_dims(this_image_inter, axis  = 0)
        this_image_inter = np.expand_dims(this_image_inter, axis  = -1)
        this_image_inter = np.expand_dims(this_image_inter, axis  = 0)
        #print(this_image_inter.shape)
        if out is None: 
            out = this_image_inter
        else:
            out = np.concatenate([out, this_image_inter], axis =0)
    return out 

def resize_by_index(src, scale):
    out = None 
    this_image = src[0,:,:,0]
    this_image_inter = cv2.resize(this_image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR)
    this_image_inter = np.expand_dims(this_image_inter, axis  = 0)
    this_image_inter = np.expand_dims(this_image_inter, axis  = -1)

    return this_image_inter 

def mean_std_vector_by_patch(means, stds, scale):
    
    mean_vector, std_vector = [], [] 
    
    for l in range(len(means)):
        
        th_patch_means = int(scale**2) * [means[l]]
        mean_vector.append(th_patch_means)
        th_patch_std   = int(scale**2) * [stds[l]]
        std_vector.append(th_patch_std)
    
    mean_vector = np.concatentae(mean_vector)
    std_vector  = np.concatentae(std_vector)
    
    return mean_vector, std_vector


def inverse_normalization_by_index(mean, std, ytrue, ypred, scale = None):
    

    flatten_ytrue = ytrue.flatten()
    mean_vector   = np.array(len(flatten_ytrue)*[mean])
    std_vector    = np.array(len(flatten_ytrue)*[std])

    flatten_ytrue = (flatten_ytrue * std_vector) + mean_vector
    reshape_ytrue = np.reshape(flatten_ytrue, (scale, scale))
    reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = 0)
    reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = -1)
    
    flatten_ypred = ypred.flatten()
    flatten_ypred = (flatten_ypred * std_vector) + mean_vector
    reshape_ypred = np.reshape(flatten_ypred, (scale, scale))
    reshape_ypred = np.expand_dims(reshape_ypred, axis  = 0)
    reshape_ypred = np.expand_dims(reshape_ypred, axis  = -1)
    tb_pred_out = np.full((blocks_size[block][0], blocks_size[block][1]), -1) 
    tb_true_out = np.full((blocks_size[block][0], blocks_size[block][1]), -1)  
 
        
    for l in range(len(pred_npy)):
        #for b in range(len(data[l]['ID'])):
        tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
        
        if len(tb_pred_indices) !=0:   
            #print(f"{block}: {tb_pred_indices}")
            for index in tb_pred_indices:
                
                block_key     = next(iter(pred_npy[l]))
                tb_block_name = pred_npy[l][block_key][index]
                b_name_split  = os.path.split(tb_block_name)[-1]
                tb_block_root_name = b_name_split.replace(b_name_split[12:], '')
                
                x0            = pred_npy[l]['X'][index]
                y0            = pred_npy[l]['Y'][index]
    
                tb_ytrue80        = pred_npy[l]['ytrue'][index]
                tb_ypred80        = pred_npy[l]['ypred80'][index]
                
                tb_ytrue40        = resize_byindex(tb_ytrue80, 0.5)
                tb_ypred40        = pred_npy[l]['ypred40'][index]

                tb_ytrue20        = resize_byindex(tb_ytrue80, 0.25)
                tb_ypred20        = pred_npy[l]['ypred20'][index]
                
                tb_cultivar_name  = pred_npy[l]['cultivar'][index] 
                
                tb_win_block_mean = pred_npy[l]['win_block_mean'][index]
                tb_win_block_std  = pred_npy[l]['win_block_std'][index]
                

                tb_ytrue80_inv, th_ypred80_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name, 
                                                                        tb_ytrue80, tb_ypred80, 80)
                tb_ytrue40_inv, th_ypred40_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name, 
                                                                        tb_ytrue40, tb_ypred40, 40)
                tb_ytrue20_inv, th_ypred20_inv = inverse_normalization(tb_win_block_mean, tb_win_block_std, tb_block_root_name, tb_cultivar_name, 
                                                                        tb_ytrue20, tb_ypred20, 20)
                
                # aggregate
                tb_ypred40to80_inv = resize_byindex(th_ypred40_inv, 2)
                tb_ypred20to80_inv = resize_byindex(th_ypred20_inv, 4)
                tb_ypredagg_inv    = np.mean(np.array([th_ypred80_inv, tb_ypred40to80_inv, tb_ypred20to80_inv]), axis=0 )
                
                if np.all((tb_pred_out[x0:x0+80, y0:y0+80] == -1)):
                    tb_pred_out[x0:x0+80, y0:y0+80] = tb_ypredagg_inv[0,:,:,0]
                else: 
                    tb_pred_out[x0:x0+80, y0:y0+80] = (tb_pred_out[x0:x0+80, y0:y0+80] + tb_ypredagg_inv[0,:,:,0])/2

                if np.all((tb_true_out[x0:x0+80, y0:y0+80] == -1)):
                    tb_true_out[x0:x0+80, y0:y0+80] = tb_ytrue80_inv[0,:,:,0]
                else: 
                    tb_true_out[x0:x0+80, y0:y0+80] = (tb_true_out[x0:x0+80, y0:y0+80] + tb_ytrue80_inv[0,:,:,0])/2
                    #print("loop is done!")
    #print("here!")               
    # creating a dictionary for all the block for future visualzation: 
    this_block_dict  = {"block": block, "true_mtx": tb_true_out, "pred_mtx": tb_pred_out}
    blocks_eval_dict.append(this_block_dict)
    
    # aggregation dataframe: 
    tb_flatten_ytrue_agg = tb_true_out.flatten()
    tb_flatten_ytrue_agg = tb_flatten_ytrue_agg[tb_flatten_ytrue_agg != -1]
    out_ytrue_agg.append(tb_flatten_ytrue_agg)
    
    tb_flatten_ypred_agg = tb_pred_out.flatten()
    tb_flatten_ypred_agg = tb_flatten_ypred_agg[tb_flatten_ypred_agg != -1]            
    out_ypred_agg.append(tb_flatten_ypred_agg)
    
    tb_block_id_agg = np.array(len(tb_flatten_ypred_agg)*[block_id], dtype=np.int32)
    out_blocks_agg.append(tb_block_id_agg)
    
    tb_cultivar_id_agg = np.array(len(tb_flatten_ypred_agg)*[cultivar_id], dtype=np.int8)
    out_cultivars_agg.append(tb_cultivar_id_agg)
        

    # agg    
    out_blocks_agg       = np.concatenate(out_blocks_agg)
    out_cultivars_agg    = np.concatenate(out_cultivars_agg)
    out_ytrue_agg        = np.concatenate(out_ytrue_agg)#.astype(np.float16)
    out_ypred_agg        = np.concatenate(out_ypred_agg)#.astype(np.float16) 
    OutDFAgg['block']    = out_blocks_agg
    OutDFAgg['cultivar'] = out_cultivars_agg
    OutDFAgg['ytrue']    = out_ytrue_agg
    OutDFAgg['ypred']    = out_ypred_agg  


    return OutDFAgg, blocks_eval_dict



def EvalScenarioMSOld(pred_npy, YS = None, norm_type = None):
    
    
    OutDFAgg = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivarsAgg, out_ytrueAgg, out_ypredAgg = [], [], []
    
    OutDF80 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars80, out_ytrue80, out_ypred80 = [], [], []
    
    OutDF40 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars40, out_ytrue40, out_ypred40 = [], [], []
    
    OutDF20 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars20, out_ytrue20, out_ypred20 = [], [], []
    
  

    
    return reshape_ytrue, reshape_ypred 

def inverse_normalization_by_patch(means, stds, ytrue, ypred, scale = None):
    

    out_ytrue, out_ypred = None,None
    for l in range(len(means)): 
        th_patch_mean = int(scale**2) * [means[l]]
        th_patch_std  = int(scale**2) * [stds[l]]
        
        
        th_ytrue      = ytrue[l,0, :,:, 0]
        flatten_ytrue = th_ytrue.flatten()
        flatten_ytrue = (flatten_ytrue * th_patch_std) + th_patch_mean 
        reshape_ytrue = np.reshape(flatten_ytrue, (scale, scale))
        reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = 0)
        reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = -1)
        reshape_ytrue = np.expand_dims(reshape_ytrue, axis  = 0)
        
        if out_ytrue is None: 
            out_ytrue = reshape_ytrue
        else: 
            out_ytrue = np.concatenate([out_ytrue, reshape_ytrue], axis = 0)
        
        th_ypred      = ypred[l,0, :,:, 0]
        flatten_ypred = th_ypred.flatten()
        flatten_ypred = (flatten_ypred * th_patch_std) + th_patch_mean
        reshape_ypred = np.reshape(flatten_ypred, (scale, scale))
        reshape_ypred = np.expand_dims(reshape_ypred, axis  = 0)
        reshape_ypred = np.expand_dims(reshape_ypred, axis  = -1)
        reshape_ypred = np.expand_dims(reshape_ypred, axis  = 0)
        
        if out_ypred is None: 
            out_ypred = reshape_ypred
        else: 
            out_ypred = np.concatenate([out_ypred, reshape_ypred], axis = 0)
    
    return out_ytrue, out_ypred 


def agg_pixelovelapp_df_2d(df):
    out = pd.DataFrame()
    
    newdf = df.groupby(["block", "cultivar", "x", "y"]).agg(
        block    = pd.NamedAgg(column="block", aggfunc=np.unique),
        cultivar = pd.NamedAgg(column="cultivar", aggfunc=np.unique),
        x        = pd.NamedAgg(column="x", aggfunc=np.unique),
        y        = pd.NamedAgg(column="y", aggfunc=np.unique),
        ytrue    = pd.NamedAgg(column="ytrue", aggfunc=np.mean),
        ypred_w1    = pd.NamedAgg(column="ypred_w1", aggfunc=np.mean),
        ypred_w2    = pd.NamedAgg(column="ypred_w2", aggfunc=np.mean),
        ypred_w3    = pd.NamedAgg(column="ypred_w3", aggfunc=np.mean),
        ypred_w4    = pd.NamedAgg(column="ypred_w4", aggfunc=np.mean),
        ypred_w5    = pd.NamedAgg(column="ypred_w5", aggfunc=np.mean),
        ypred_w6    = pd.NamedAgg(column="ypred_w6", aggfunc=np.mean),
        ypred_w7    = pd.NamedAgg(column="ypred_w7", aggfunc=np.mean),
        ypred_w8    = pd.NamedAgg(column="ypred_w8", aggfunc=np.mean),
        ypred_w9    = pd.NamedAgg(column="ypred_w9", aggfunc=np.mean),
        ypred_w10    = pd.NamedAgg(column="ypred_w10", aggfunc=np.mean),
        ypred_w11    = pd.NamedAgg(column="ypred_w11", aggfunc=np.mean),
        ypred_w12    = pd.NamedAgg(column="ypred_w12", aggfunc=np.mean),
        ypred_w13    = pd.NamedAgg(column="ypred_w13", aggfunc=np.mean),
        ypred_w14    = pd.NamedAgg(column="ypred_w14", aggfunc=np.mean),
        ypred_w15    = pd.NamedAgg(column="ypred_w15", aggfunc=np.mean),
        
    )
    
    
    out['block'] = newdf['block'].values
    out['cultivar'] = newdf['cultivar'].values
    out['x'] = newdf['x'].values
    out['y'] = newdf['y'].values
    out['ytrue'] = newdf['ytrue'].values
    out['ypred_w1'] = newdf['ypred_w1'].values
    out['ypred_w2'] = newdf['ypred_w2'].values
    out['ypred_w3'] = newdf['ypred_w3'].values
    out['ypred_w4'] = newdf['ypred_w4'].values
    out['ypred_w5'] = newdf['ypred_w5'].values
    out['ypred_w6'] = newdf['ypred_w6'].values
    out['ypred_w7'] = newdf['ypred_w7'].values
    out['ypred_w8'] = newdf['ypred_w8'].values
    out['ypred_w9'] = newdf['ypred_w9'].values
    out['ypred_w10'] = newdf['ypred_w10'].values
    out['ypred_w11'] = newdf['ypred_w11'].values
    out['ypred_w12'] = newdf['ypred_w12'].values
    out['ypred_w13'] = newdf['ypred_w13'].values
    out['ypred_w14'] = newdf['ypred_w14'].values
    out['ypred_w15'] = newdf['ypred_w15'].values


    return out




def agg_pixelovelapp_df(df):
    out = pd.DataFrame()
    
    newdf = df.groupby(["block", "cultivar", "x", "y"]).agg(
        
        block    = pd.NamedAgg(column="block", aggfunc=np.unique),
        cultivar = pd.NamedAgg(column="cultivar", aggfunc=np.unique),
        x        = pd.NamedAgg(column="x", aggfunc=np.unique),
        y        = pd.NamedAgg(column="y", aggfunc=np.unique),
        ytrue    = pd.NamedAgg(column="ytrue", aggfunc=np.mean),
        ypred    = pd.NamedAgg(column="ypred", aggfunc=np.mean),
    )
    
    
    out['block'] = newdf['block'].values
    out['cultivar'] = newdf['cultivar'].values
    out['x'] = newdf['x'].values
    out['y'] = newdf['y'].values
    out['ytrue'] = newdf['ytrue'].values
    out['ypred'] = newdf['ypred'].values


    return out

def block_pixel_overlap_agg(df):
    
    newdf = pd.DataFrame()
    block, cultivar, x, y, ytrue, ypred = [],[],[],[],[], []
    g = df.groupby(["block", "cultivar", "x", "y"])
    
    for l, gdf in g:
        block.append(gdf['block'].iloc[0])
        cultivar.append(gdf['cultivar'].iloc[0])
        x.append(gdf['x'].iloc[0])
        y.append(gdf['y'].iloc[0])
        ytrue.append(gdf['ytrue'].mean())
        ypred.append(gdf['ypred'].mean())
    
    newdf['block'] = block
    newdf['cultivar'] = cultivar
    newdf['x'] = x
    newdf['y'] = y
    newdf['ytrue'] = ytrue
    newdf['ypred'] =ypred

    return newdf















def ScenarioEvaluation(pred_npy, block_normalization = None, scale = None):

    OutDF = pd.DataFrame()
    out_ytrue, out_ypred = [], []
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue80 = pred_npy[l]['ytrue']
        ypred80 = pred_npy[l]['ypred80']
        
        ytrue40 = ms_interpol(ytrue80, 0.5)
        ypred40 = pred_npy[l]['ypred40']
        ypred40to80 = ms_interpol(ypred40, 2)
        
        ytrue20 = ms_interpol(ytrue80, 0.25)
        ypred20 = pred_npy[l]['ypred20']
        ypred20to80 = ms_interpol(ypred20, 4)

        ypred_agg = np.mean( np.array([ ypred80, ypred40to80, ypred20to80]), axis=0 )
        
        ytrue40     = resize_by_patch(ytrue80, 0.5)
        ypred40     = pred_npy[l]['ypred40']
        
        if YS == True:
            cultivar_allpatch = pred_npy[l]['cultivar'] 
            
            cultivarsAgg, fytrueAgg, fypredAgg = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred_agg, norm_type, 80)
            out_cultivarsAgg.append(cultivarsAgg)
            out_ytrueAgg.append(fytrueAgg)
            out_ypredAgg.append(fypredAgg)
            
            cultivars80, fytrue80, fypred80 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred80, norm_type, 80)
            out_cultivars80.append(cultivars80)
            out_ytrue80.append(fytrue80)
            out_ypred80.append(fypred80)
            
            cultivars40, fytrue40, fypred40 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue40, ypred40, norm_type, 40)
            out_cultivars40.append(cultivars40)
            out_ytrue40.append(fytrue40)
            out_ypred40.append(fypred40)
            
            cultivars20, fytrue20, fypred20 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue20, ypred20, norm_type, 20)
            out_cultivars20.append(cultivars20)
            out_ytrue20.append(fytrue20)
            out_ypred20.append(fypred20)
            
            flat_inv_ypred_40 = inv_ypred_40.flatten()
            out_ypred.append(flat_inv_ypred_40)

        elif scale == 80:
            flat_inv_yture_80 = inv_yture_80.flatten()
            out_ytrue.append(flat_inv_yture_80)
            
            
        else:
            cultivarsAgg, fytrueAgg, fypredAgg = scale_extraction(block_allpatch, ytrue80, ypred_agg, 80)
            out_cultivarsAgg.append(cultivarsAgg)
            out_ytrueAgg.append(fytrueAgg)
            out_ypredAgg.append(fypredAgg)
            
            cultivars80, fytrue80, fypred80 = scale_extraction(block_allpatch, ytrue80, ypred80, 80)
            out_cultivars80.append(cultivars80)
            out_ytrue80.append(fytrue80)
            out_ypred80.append(fypred80)
            
            cultivars40, fytrue40, fypred40 = scale_extraction(block_allpatch, ytrue40, ypred40, 40)
            out_cultivars40.append(cultivars40)
            out_ytrue40.append(fytrue40)
            out_ypred40.append(fypred40)
            
            cultivars20, fytrue20, fypred20 = scale_extraction(block_allpatch, ytrue20, ypred20, 20)
            out_cultivars20.append(cultivars20)
            out_ytrue20.append(fytrue20)
            out_ypred20.append(fypred20)
            

            
    out_cultivarsAgg = np.concatenate(out_cultivarsAgg)#.astype(np.float16)
    out_ytrueAgg = np.concatenate(out_ytrueAgg)#.astype(np.float16)
    out_ypredAgg = np.concatenate(out_ypredAgg)#.astype(np.float16)    
    OutDFAgg['cultivar'] = out_cultivarsAgg
    OutDFAgg['ytrue'] = out_ytrueAgg
    OutDFAgg['ypred'] = out_ypredAgg          

    out_cultivars80 = np.concatenate(out_cultivars80)#.astype(np.float16)
    out_ytrue80 = np.concatenate(out_ytrue80)#.astype(np.float16)
    out_ypred80 = np.concatenate(out_ypred80)#.astype(np.float16)    
    OutDF80['cultivar'] = out_cultivars80
    OutDF80['ytrue'] = out_ytrue80
    OutDF80['ypred'] = out_ypred80  
    
    out_cultivars40 = np.concatenate(out_cultivars40)#.astype(np.float16)
    out_ytrue40 = np.concatenate(out_ytrue40)#.astype(np.float16)
    out_ypred40 = np.concatenate(out_ypred40)#.astype(np.float16)    
    OutDF40['cultivar'] = out_cultivars40
    OutDF40['ytrue'] = out_ytrue40
    OutDF40['ypred'] = out_ypred40 
    
    out_cultivars20 = np.concatenate(out_cultivars20)#.astype(np.float16)
    out_ytrue20 = np.concatenate(out_ytrue20)#.astype(np.float16)
    out_ypred20 = np.concatenate(out_ypred20)#.astype(np.float16)    
    OutDF20['cultivar'] = out_cultivars20
    OutDF20['ytrue'] = out_ytrue20
    OutDF20['ypred'] = out_ypred20 
    
    return OutDFAgg, OutDF80, OutDF40, OutDF20
    
def EvalScenarioMS(pred_npy, YS = None, norm_type = None):
    OutDFAgg = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivarsAgg, out_ytrueAgg, out_ypredAgg = [], [], []
    
    OutDF80 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars80, out_ytrue80, out_ypred80 = [], [], []
    
    OutDF40 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars40, out_ytrue40, out_ypred40 = [], [], []
    
    OutDF20 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars20, out_ytrue20, out_ypred20 = [], [], []
    
    OutDF10 = pd.DataFrame(columns = ['cultivar', 'ytrue', 'ypred'])
    out_cultivars10, out_ytrue10, out_ypred10 = [], [], []
    

    
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue80 = pred_npy[l]['ytrue']
        ypred80 = pred_npy[l]['ypred80']
        
        ytrue40 = resize_by_index(ytrue80, 0.5)
        ypred40 = pred_npy[l]['ypred40']
        ypred40to80 = resize_by_index(ypred40, 2)
        
        ytrue20 = resize_by_index(ytrue80, 0.25)
        ypred20 = pred_npy[l]['ypred20']
        ypred20to80 = resize_by_index(ypred20, 4)

        ypred_agg = np.mean( np.array([ ypred80, ypred40to80, ypred20to80]), axis=0 )
        
        
        if YS == True:
            cultivar_allpatch = pred_npy[l]['cultivar'] 
            
            cultivarsAgg, fytrueAgg, fypredAgg = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred_agg, norm_type, 80)
            out_cultivarsAgg.append(cultivarsAgg)
            out_ytrueAgg.append(fytrueAgg)
            out_ypredAgg.append(fypredAgg)
            
            cultivars80, fytrue80, fypred80 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred80, norm_type, 80)
            out_cultivars80.append(cultivars80)
            out_ytrue80.append(fytrue80)
            out_ypred80.append(fypred80)
            
            cultivars40, fytrue40, fypred40 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue40, ypred40, norm_type, 40)
            out_cultivars40.append(cultivars40)
            out_ytrue40.append(fytrue40)
            out_ypred40.append(fypred40)
            
            cultivars20, fytrue20, fypred20 = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue20, ypred20, norm_type, 20)
            out_cultivars20.append(cultivars20)
            out_ytrue20.append(fytrue20)
            out_ypred20.append(fypred20)
            
            
            
        else:
            cultivarsAgg, fytrueAgg, fypredAgg = scale_extraction(block_allpatch, ytrue80, ypred_agg, 80)
            out_cultivarsAgg.append(cultivarsAgg)
            out_ytrueAgg.append(fytrueAgg)
            out_ypredAgg.append(fypredAgg)
            
            cultivars80, fytrue80, fypred80 = scale_extraction(block_allpatch, ytrue80, ypred80, 80)
            out_cultivars80.append(cultivars80)
            out_ytrue80.append(fytrue80)
            out_ypred80.append(fypred80)
            
            cultivars40, fytrue40, fypred40 = scale_extraction(block_allpatch, ytrue40, ypred40, 40)
            out_cultivars40.append(cultivars40)
            out_ytrue40.append(fytrue40)
            out_ypred40.append(fypred40)
            
            cultivars20, fytrue20, fypred20 = scale_extraction(block_allpatch, ytrue20, ypred20, 20)
            out_cultivars20.append(cultivars20)
            out_ytrue20.append(fytrue20)
            out_ypred20.append(fypred20)
            

            
    out_cultivarsAgg = np.concatenate(out_cultivarsAgg)#.astype(np.float16)
    out_ytrueAgg = np.concatenate(out_ytrueAgg)#.astype(np.float16)
    out_ypredAgg = np.concatenate(out_ypredAgg)#.astype(np.float16)    
    OutDFAgg['cultivar'] = out_cultivarsAgg
    OutDFAgg['ytrue'] = out_ytrueAgg
    OutDFAgg['ypred'] = out_ypredAgg          

    out_cultivars80 = np.concatenate(out_cultivars80)#.astype(np.float16)
    out_ytrue80 = np.concatenate(out_ytrue80)#.astype(np.float16)
    out_ypred80 = np.concatenate(out_ypred80)#.astype(np.float16)    
    OutDF80['cultivar'] = out_cultivars80
    OutDF80['ytrue'] = out_ytrue80
    OutDF80['ypred'] = out_ypred80  
    
    out_cultivars40 = np.concatenate(out_cultivars40)#.astype(np.float16)
    out_ytrue40 = np.concatenate(out_ytrue40)#.astype(np.float16)
    out_ypred40 = np.concatenate(out_ypred40)#.astype(np.float16)    
    OutDF40['cultivar'] = out_cultivars40
    OutDF40['ytrue'] = out_ytrue40
    OutDF40['ypred'] = out_ypred40 
    
    out_cultivars20 = np.concatenate(out_cultivars20)#.astype(np.float16)
    out_ytrue20 = np.concatenate(out_ytrue20)#.astype(np.float16)
    out_ypred20 = np.concatenate(out_ypred20)#.astype(np.float16)    
    OutDF20['cultivar'] = out_cultivars20
    OutDF20['ytrue'] = out_ytrue20
    OutDF20['ypred'] = out_ypred20 
    
    out_cultivars10 = np.concatenate(out_cultivars10)#.astype(np.float16)
    out_ytrue10 = np.concatenate(out_ytrue10)#.astype(np.float16)
    out_ypred10 = np.concatenate(out_ypred10)#.astype(np.float16)    
    OutDF10['cultivar'] = out_cultivars10
    OutDF10['ytrue'] = out_ytrue10
    OutDF10['ypred'] = out_ypred10 
      
    
    return OutDFAgg, OutDF80, OutDF40, OutDF20, OutDF10
        



def EvalScenario(pred_npy, YS = None, norm_type = None, AggYS = None, Agg = None, Agg_Scale = None):
    
    outDF = pd.DataFrame(columns = ['Cultivar', 'ytrue', 'ypred'])
    out_ytrue, out_ypred = [], []
    out_cultivars = []
    
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue = pred_npy[l]['ytrue']
        ypred = pred_npy[l]['ypred']
        
        
        if YS == True:
            cultivars = block_cultivar_extract(block_allpatch, int(6400))
            out_cultivars.append(cultivars)
            
            cultivar_allpatch = pred_npy[l]['cultivar']
            means, stds = block_denorm_extract(cultivar_allpatch, pix_n = int(6400), norm_type = norm_type)
            flatten_ytrue = ytrue.flatten()
            flatten_ytrue = (flatten_ytrue * stds) + means 
            out_ytrue.append(flatten_ytrue)
            
            flatten_ypred = ypred.flatten()
            flatten_ypred = (flatten_ypred * stds) + means 
            out_ypred.append(flatten_ypred)
            
        elif Agg == True: 
            number_pixels = int(6400/(Agg_Scale**2))
            cultivars = block_cultivar_extract(block_allpatch, number_pixels)
            out_cultivars.append(cultivars)
            
            
            ytrue_agg = many_agg(ytrue, Agg_Scale)
            ypred_agg = many_agg(ypred, Agg_Scale)
            out_ytrue.append(ytrue_agg)
            out_ypred.append(ypred_agg)
            
            
        elif AggYS == True:
            number_pixels = int(6400/(Agg_Scale**2))
            cultivars = block_cultivar_extract(block_allpatch, number_pixels)
            out_cultivars.append(cultivars)
            
            
            cultivar_allpatch = pred_npy[l]['cultivar']
            means, stds = block_denorm_extract(cultivar_allpatch, pix_n = number_pixels, norm_type = norm_type)

            ytrue_agg = many_agg(ytrue, Agg_Scale)
            ytrue_agg = (ytrue_agg *stds) + means


            ypred_agg = many_agg(ypred, Agg_Scale)
            ypred_agg = (ypred_agg *stds) + means

            #print(ytrue_agg.shape)
            out_ytrue.append(ytrue_agg)
            out_ypred.append(ypred_agg)
            
        else:
            cultivars = block_cultivar_extract(block_allpatch, int(6400))
            out_cultivars.append(cultivars)
            flatten_ytrue = ytrue.flatten()
            out_ytrue.append(flatten_ytrue)
            
            flatten_ypred = ypred.flatten()
            out_ypred.append(flatten_ypred)
            
            

    out_cultivars = np.concatenate(out_cultivars)#.astype(np.float16)
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred = np.concatenate(out_ypred)#.astype(np.float16)    
    

    outDF['Cultivar'] = out_cultivars
    outDF['ytrue'] = out_ytrue
    outDF['ypred'] = out_ypred            
    
    return outDF

def EvalCultivars(pred_npy):
    
    OutDFAgg = pd.DataFrame(columns = ['Cultivar', 'ytrue', 'ypred'])
    out_cultivarsAgg, out_ytrueAgg, out_ypredAgg = [], [], []

    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
        #print(block_allpatch) 
        
        ytrue80 = pred_npy[l]['ytrue']
        ypred80 = pred_npy[l]['ypred80']
        
        ypred40 = pred_npy[l]['ypred40']
        ypred40to80 = ms_interpol(ypred40, 2)
    
        ypred20 = pred_npy[l]['ypred20']
        ypred20to80 = ms_interpol(ypred20, 4)

        ypred_agg = np.mean( np.array([ ypred80, ypred40to80, ypred20to80]), axis=0 )
    
        cultivar_allpatch = pred_npy[l]['Cultivar'] 
        
        cultivarsAgg, fytrueAgg, fypredAgg = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred_agg, 'S', 80)
        out_cultivarsAgg.append(cultivarsAgg)
        out_ytrueAgg.append(fytrueAgg)
        out_ypredAgg.append(fypredAgg)
    
    out_cultivarsAgg = np.concatenate(out_cultivarsAgg)#.astype(np.float16)
    out_ytrueAgg = np.concatenate(out_ytrueAgg)#.astype(np.float16)
    out_ypredAgg = np.concatenate(out_ypredAgg)#.astype(np.float16)    
    OutDFAgg['cultivar'] = out_cultivarsAgg
    OutDFAgg['ytrue'] = out_ytrueAgg
    OutDFAgg['ypred'] = out_ypredAgg 
    
    return OutDFAgg

def EvalCultivars80(pred_npy):
    
    OutDFAgg = pd.DataFrame(columns = ['Cultivar', 'ytrue', 'ypred'])
    out_cultivarsAgg, out_ytrueAgg, out_ypredAgg = [], [], []

    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
        #print(block_allpatch) 
        
        ytrue80 = pred_npy[l]['ytrue']
        ypred80 = pred_npy[l]['ypred80']
        

        cultivar_allpatch = pred_npy[l]['Cultivar'] 
        
        cultivarsAgg, fytrueAgg, fypredAgg = scale_ys_extraction(block_allpatch, cultivar_allpatch, ytrue80, ypred80, 'S', 80)
        out_cultivarsAgg.append(cultivarsAgg)
        out_ytrueAgg.append(fytrueAgg)
        out_ypredAgg.append(fypredAgg)
    
    out_cultivarsAgg = np.concatenate(out_cultivarsAgg)#.astype(np.float16)
    out_ytrueAgg = np.concatenate(out_ytrueAgg)#.astype(np.float16)
    out_ypredAgg = np.concatenate(out_ypredAgg)#.astype(np.float16)    
    OutDFAgg['Cultivar'] = out_cultivarsAgg
    OutDFAgg['ytrue'] = out_ytrueAgg
    OutDFAgg['ypred'] = out_ypredAgg 
    
    return OutDFAgg    

def many_agg(src, scale):
    
    out = []
    for l in range(src.shape[0]):
        this_image = src[l, 0,:,:,0]
        agg_img = aggregate(this_image, scale)
        flat_agg_img = agg_img.flatten()
        out.append(flat_agg_img)
    mtx = np.concatenate(out)


    return mtx    

def scale_ys_extraction(block_names, cultivar_names, ytrue, ypred, norm_type, scale):
    
    cultivars     = block_cultivar_extract(block_names, int(scale**2))
    means, stds   = block_denorm_extract(cultivar_names, pix_n = int(scale**2), norm_type = norm_type)
    
    flatten_ytrue = ytrue.flatten()
    flatten_ytrue = (flatten_ytrue * stds) + means 

    flatten_ypred = ypred.flatten()
    flatten_ypred = (flatten_ypred * stds) + means



def block_eval_mtx(pred_dic, true_df, block_id):
    
    df_indices = [i for i, x in enumerate(true_df['ID']) if x == block_id]
    mask_block = np.load(true_df.loc[df_indices[0]]['LABEL_PATH'], allow_pickle=True)
    mask_block = mask_block[0,:,:,0]
    pred_out = np.full_like(mask_block, -1) 
    true_out = np.full_like(mask_block, -1)  
    pixel_index_iter = np.zeros_like(mask_block) 
    
    for l in range(len(pred_dict)):
        #for b in range(len(data[l]['ID'])):
        pred_indices = [i for i, x in enumerate(pred_dict[l]['ID']) if x == block_id]
        if len(pred_indices) !=0:    
            for index in pred_indices:
                x0 = pred_dict[l]['X'][index]
                y0 = pred_dict[l]['Y'][index]
                ypred = pred_dict[l]['ypred'][index]
                
                #print(f"X:{x0}; Y:{y0}")
                #print(f"True: {ytrue.shape}; Pred:{ypred.shape}")
                
                '''patch_fill_ones = np.ones((80,80)) 
                if np.all((pixel_index_iter[x0:x0+80, y0:y0+80] == 0)):
                    pixel_index_iter[x0:x0+80, y0:y0+80] = patch_fill_ones
                else: 
                    pixel_index_iter[x0:x0+80, y0:y0+80] = (pixel_index_iter[x0:x0+80, y0:y0+80] + patch_fill_ones)'''
                
                
                
                if np.all((pred_out[x0:x0+80, y0:y0+80] == -1)):
                    pred_out[x0:x0+80, y0:y0+80] = ypred[0,:,:,0]
                else: 
                    pred_out[x0:x0+80, y0:y0+80] = (pred_out[x0:x0+80, y0:y0+80] + ypred[0,:,:,0])/2

                ytrue = mask_block[x0:x0+80, y0:y0+80]
                true_out[x0:x0+80, y0:y0+80] = ytrue

    #pixel_index_iter[pixel_index_iter == 0 ] = 1 
    #true_out  = pred_out / pixel_index_iter   
                
    flatten_ytrue = true_out.flatten()
    flatten_ytrue = flatten_ytrue[flatten_ytrue != -1]
    flatten_ypred = pred_out.flatten()
    flatten_ypred = flatten_ypred[flatten_ypred != -1]            
    reg_metrics = regression_metrics(flatten_ytrue, flatten_ypred) 
    
    return reg_metrics, true_out, flatten_ytrue, pred_out, flatten_ypred



def patch_eval_mtx(pred_dic, true_df, block_id):
    
    df_indices = [i for i, x in enumerate(true_df['ID']) if x == block_id]
    mask_block = np.load(true_df.loc[df_indices[0]]['LABEL_PATH'], allow_pickle=True)
    mask_block = mask_block[0,:,:,0]
    pred_out = np.full_like(mask_block, -1) 
    true_out = np.full_like(mask_block, -1)  
    pixel_index_iter = np.zeros_like(mask_block) 
    
    for l in range(len(pred_dict)):
        #for b in range(len(data[l]['ID'])):
        pred_indices = [i for i, x in enumerate(pred_dict[l]['ID']) if x == block_id]
        if len(pred_indices) !=0:    
            for index in pred_indices:
                #x0 = pred_dict[l]['X'][index]
                #y0 = pred_dict[l]['Y'][index]
                ypred = pred_dict[l]['ypred'][index]
                ypred = ypred[0,:,:,0]
                ytrue = pred_dict[l]['ytrue'][index]
                ytrue = ytrue[0,:,:,0]
                #print(f"X:{x0}; Y:{y0}")
                #print(f"True: {ytrue.shape}; Pred:{ypred.shape}")
                
                '''patch_fill_ones = np.ones((80,80)) 
                if np.all((pixel_index_iter[x0:x0+80, y0:y0+80] == 0)):
                    pixel_index_iter[x0:x0+80, y0:y0+80] = patch_fill_ones
                else: 
                    pixel_index_iter[x0:x0+80, y0:y0+80] = (pixel_index_iter[x0:x0+80, y0:y0+80] + patch_fill_ones)'''
                
                '''if np.all((pred_out[x0:x0+80, y0:y0+80] == -1)):
                    pred_out[x0:x0+80, y0:y0+80] = ypred[0,:,:,0]
                else: 
                    pred_out[x0:x0+80, y0:y0+80] = (pred_out[x0:x0+80, y0:y0+80] + ypred[0,:,:,0])/2'''

                #ytrue = mask_block[x0:x0+80, y0:y0+80]
                #true_out[x0:x0+80, y0:y0+80] = ytrue
                reg_metrics = regression_metrics(ytrue, ypred) 
                print(f"RMSE: {reg_metrics[2]}, R^2: {reg_metrics[0]}, MAE: {reg_metrics[1]}.")

    #pixel_index_iter[pixel_index_iter == 0 ] = 1 
    #true_out  = pred_out / pixel_index_iter   
                
    #flatten_ytrue = true_out.flatten()
    #flatten_ytrue = flatten_ytrue[flatten_ytrue != -1]
    #flatten_ypred = pred_out.flatten()
    #flatten_ypred = flatten_ypred[flatten_ypred != -1]            
    #reg_metrics = regression_metrics(flatten_ytrue, flatten_ypred) 
    
    #return reg_metrics, true_out, flatten_ytrue, pred_out, flatten_ypred    


def ScenarioEvaluation2D(pred_npy):

    OutDF = pd.DataFrame()
    out_ytrue= []
    out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5, out_ypred_w6, out_ypred_w7, out_ypred_w8 = [],[],[],[],[],[],[],[]
    out_ypred_w9, out_ypred_w10, out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [],[],[],[],[],[],[]
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue     = pred_npy[l]['ytrue']
        ytrue_flat = ytrue.flatten()
        out_ytrue.append(ytrue_flat)
        
        ypred_w1  = pred_npy[l]['ypred_w1']
        ypred_w1_flat = ypred_w1.flatten()
        out_ypred_w1.append(ypred_w1_flat)
        
        ypred_w2  = pred_npy[l]['ypred_w2']
        ypred_w2_flat = ypred_w2.flatten()
        out_ypred_w2.append(ypred_w2_flat)
        
        ypred_w3  = pred_npy[l]['ypred_w3']
        ypred_w3_flat = ypred_w3.flatten()
        out_ypred_w3.append(ypred_w3_flat)
        
        ypred_w4  = pred_npy[l]['ypred_w4']
        ypred_w4_flat = ypred_w4.flatten()
        out_ypred_w4.append(ypred_w4_flat)
        
        ypred_w5  = pred_npy[l]['ypred_w5']
        ypred_w5_flat = ypred_w5.flatten()
        out_ypred_w5.append(ypred_w5_flat)
        
        ypred_w6  = pred_npy[l]['ypred_w6']
        ypred_w6_flat = ypred_w6.flatten()
        out_ypred_w6.append(ypred_w6_flat)
        
        ypred_w7  = pred_npy[l]['ypred_w7']
        ypred_w7_flat = ypred_w7.flatten()
        out_ypred_w7.append(ypred_w7_flat)
        
        ypred_w8  = pred_npy[l]['ypred_w8']
        ypred_w8_flat = ypred_w8.flatten()
        out_ypred_w8.append(ypred_w8_flat)
        
        ypred_w9  = pred_npy[l]['ypred_w9']
        ypred_w9_flat = ypred_w9.flatten()
        out_ypred_w9.append(ypred_w9_flat)
        
        ypred_w10  = pred_npy[l]['ypred_w10']
        ypred_w10_flat = ypred_w10.flatten()
        out_ypred_w10.append(ypred_w10_flat)
        
        ypred_w11  = pred_npy[l]['ypred_w11']
        ypred_w11_flat = ypred_w11.flatten()
        out_ypred_w11.append(ypred_w11_flat)
        
        ypred_w12  = pred_npy[l]['ypred_w12']
        ypred_w12_flat = ypred_w12.flatten()
        out_ypred_w12.append(ypred_w12_flat)
        
        ypred_w13  = pred_npy[l]['ypred_w13']
        ypred_w13_flat = ypred_w13.flatten()
        out_ypred_w13.append(ypred_w13_flat)
        
        ypred_w14  = pred_npy[l]['ypred_w14']
        ypred_w14_flat = ypred_w14.flatten()
        out_ypred_w14.append(ypred_w14_flat)
        
        ypred_w15  = pred_npy[l]['ypred_w15']
        ypred_w15_flat = ypred_w15.flatten()
        out_ypred_w15.append(ypred_w15_flat)
      


            
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred_w1 = np.concatenate(out_ypred_w1) 
    out_ypred_w2 = np.concatenate(out_ypred_w2) 
    out_ypred_w3 = np.concatenate(out_ypred_w3) 
    out_ypred_w4 = np.concatenate(out_ypred_w4) 
    out_ypred_w5 = np.concatenate(out_ypred_w5) 
    out_ypred_w6 = np.concatenate(out_ypred_w6) 
    out_ypred_w7 = np.concatenate(out_ypred_w7) 
    out_ypred_w8 = np.concatenate(out_ypred_w8) 
    out_ypred_w9 = np.concatenate(out_ypred_w9) 
    out_ypred_w10 = np.concatenate(out_ypred_w10) 
    out_ypred_w11 = np.concatenate(out_ypred_w11) 
    out_ypred_w12 = np.concatenate(out_ypred_w12) 
    out_ypred_w13 = np.concatenate(out_ypred_w13) 
    out_ypred_w14 = np.concatenate(out_ypred_w14) 
    out_ypred_w15 = np.concatenate(out_ypred_w15) 

    
    OutDF['ytrue'] = out_ytrue
    OutDF['ypred_w1'] = out_ypred_w1
    OutDF['ypred_w2'] = out_ypred_w2 
    OutDF['ypred_w3'] = out_ypred_w3 
    OutDF['ypred_w4'] = out_ypred_w4 
    OutDF['ypred_w5'] = out_ypred_w5 
    OutDF['ypred_w6'] = out_ypred_w6 
    OutDF['ypred_w7'] = out_ypred_w7 
    OutDF['ypred_w8'] = out_ypred_w8 
    OutDF['ypred_w9'] = out_ypred_w9 
    OutDF['ypred_w10'] = out_ypred_w10 
    OutDF['ypred_w11'] = out_ypred_w11 
    OutDF['ypred_w12'] = out_ypred_w12 
    OutDF['ypred_w13'] = out_ypred_w13 
    OutDF['ypred_w14'] = out_ypred_w14 
    OutDF['ypred_w15'] = out_ypred_w15


    return OutDF
    

def time_series_2d_eval(train, val, test, fig_save_name, save = None): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    train_r2_list, train_mae_list, train_rmse_list, train_mape_list = [], [], [], [] 
    val_r2_list, val_mae_list, val_rmse_list, val_mape_list         = [], [], [], []
    test_r2_list, test_mae_list, test_rmse_list, test_mape_list     = [], [], [], []
    
    for i in range(len(Weeks)): 
        train_r2, train_mae, train_rmse, train_mape, _, _ = regression_metrics(train['ytrue'], train.iloc[:, i+1])
        train_r2_list.append(train_r2)
        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)
        train_mape_list.append(train_mape)
        
        val_r2, val_mae, val_rmse, val_mape, _, _         = regression_metrics(val['ytrue'], val.iloc[:, i+1])
        val_r2_list.append(val_r2)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        
        test_r2, test_mae, test_rmse, test_mape, _, _     = regression_metrics(test['ytrue'], test.iloc[:, i+1])
        test_r2_list.append(test_r2)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_mape_list.append(test_mape)
    
    results['weeks']      = Weeks
    results['Train_R2']   = train_r2_list
    results['Valid_R2']   = val_r2_list
    results['Test_R2']    = test_r2_list
    results['Train_MAE']  = train_mae_list
    results['Valid_MAE']  = val_mae_list
    results['Test_MAE']   = test_mae_list
    results['Train_RMSE'] = train_rmse_list
    results['Valid_RMSE'] = val_rmse_list
    results['Test_RMSE']  = test_rmse_list
    results['Train_MAPE'] = train_mape_list
    results['Valid_MAPE'] = val_mape_list
    results['Test_MAPE']  = test_mape_list
 

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(2, 2, figsize = (20, 10), sharex=True)


    axs[0, 0].plot(results["weeks"], results["Train_R2"], "-o")
    axs[0, 0].plot(results["weeks"], results["Valid_R2"], "-*")
    axs[0, 0].plot(results["weeks"], results["Test_R2"],  "-d")
    axs[0, 0].set_ylabel('R2')
    axs[0, 0].set_facecolor('white')
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(results["weeks"], results["Train_RMSE"], "-o")
    axs[0, 1].plot(results["weeks"], results["Valid_RMSE"], "-*")
    axs[0, 1].plot(results["weeks"], results["Test_RMSE"],  "-d")
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_facecolor('white')
    plt.setp(axs[0, 1].spines.values(), color='k')
    
    axs[1, 0].plot(results["weeks"], results["Train_MAE"], "-o")
    axs[1, 0].plot(results["weeks"], results["Valid_MAE"], "-*")
    axs[1, 0].plot(results["weeks"], results["Test_MAE"],  "-d")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].set_ylabel('MAE (ton/ac)')
    axs[1, 0].set_facecolor('white')
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(results["weeks"], results["Train_MAPE"], "-o", label = 'train')
    axs[1, 1].plot(results["weeks"], results["Valid_MAPE"], "-*", label = 'valid')
    axs[1, 1].plot(results["weeks"], results["Test_MAPE"],  "-d", label = 'test')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    plt.setp(axs[1, 1].spines.values(), color='k')
    axs[1, 1].legend(loc="upper right")
    
    
    
    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)
    
    return results 

def concat_year_s2(df1, df2, df3):
    
    df = pd.DataFrame()
    tr_mae_m, v_mae_m, te_mae_m = [], [], []
    tr_mape_m, v_mape_m, te_mape_m = [], [], []
    tr_mae_s, v_mae_s, te_mae_s = [], [], []
    tr_mape_s, v_mape_s, te_mape_s = [], [], []
    weeks=[] 
    for i in range(15): 
        weeks.append(df1.iloc[i]['weeks'])
        tr_mae_m.append(np.mean([df1.iloc[i]['Train_MAE'], df2.iloc[i]['Train_MAE'], df3.iloc[i]['Train_MAE']]))
        tr_mae_s.append(np.std([df1.iloc[i]['Train_MAE'], df2.iloc[i]['Train_MAE'], df3.iloc[i]['Train_MAE']]))

        v_mae_m.append(np.mean([df1.iloc[i]['Valid_MAE'], df2.iloc[i]['Valid_MAE'], df3.iloc[i]['Valid_MAE']]))
        v_mae_s.append(np.std([df1.iloc[i]['Valid_MAE'], df2.iloc[i]['Valid_MAE'], df3.iloc[i]['Valid_MAE']]))

        te_mae_m.append(np.mean([df1.iloc[i]['Test_MAE'], df2.iloc[i]['Test_MAE'], df3.iloc[i]['Test_MAE']]))
        te_mae_s.append(np.std([df1.iloc[i]['Test_MAE'], df2.iloc[i]['Test_MAE'], df3.iloc[i]['Test_MAE']]))

        tr_mape_m.append(np.mean([df1.iloc[i]['Train_MAPE'], df2.iloc[i]['Train_MAPE'], df3.iloc[i]['Train_MAPE']]))
        tr_mape_s.append(np.std([df1.iloc[i]['Train_MAPE'], df2.iloc[i]['Train_MAPE'], df3.iloc[i]['Train_MAPE']]))

        v_mape_m.append(np.mean([df1.iloc[i]['Valid_MAPE'], df2.iloc[i]['Valid_MAPE'], df3.iloc[i]['Valid_MAPE']]))
        v_mape_s.append(np.std([df1.iloc[i]['Valid_MAPE'], df2.iloc[i]['Valid_MAPE'], df3.iloc[i]['Valid_MAPE']]))

        te_mape_m.append(np.mean([df1.iloc[i]['Test_MAPE'], df2.iloc[i]['Test_MAPE'], df3.iloc[i]['Test_MAPE']]))
        te_mape_s.append(np.std([df1.iloc[i]['Test_MAPE'], df2.iloc[i]['Test_MAPE'], df3.iloc[i]['Test_MAPE']]))

    df['weeks'] = weeks
    df['Train_MAE_M']  = tr_mae_m
    df['Train_MAE_S']  = tr_mae_s
    df['Valid_MAE_M']  = v_mae_m
    df['Valid_MAE_S']  = v_mae_s
    df['Test_MAE_M']  = te_mae_m
    df['Test_MAE_S']  = te_mae_s
    df['Train_MAPE_M'] = tr_mape_m
    df['Train_MAPE_S'] = tr_mape_s
    df['Valid_MAPE_M'] = v_mape_m
    df['Valid_MAPE_S'] = v_mape_s
    df['Test_MAPE_M']  = te_mape_m
    df['Test_MAPE_S']  = te_mape_s
    return df

def S2_mean_eval(df2016, df2017, df2018, df2019):
    
    df = pd.DataFrame()
    tr_mae_m, v_mae_m, te_mae_m = [], [], []
    tr_mape_m, v_mape_m, te_mape_m = [], [], []
    tr_mae_s, v_mae_s, te_mae_s = [], [], []
    tr_mape_s, v_mape_s, te_mape_s = [], [], []
    weeks=[] 
    for i in range(15): 
        weeks.append(df2016.iloc[i]['weeks'])
        tr_mae_m.append(np.mean([df2016.iloc[i]['Train_MAE_M'], df2017.iloc[i]['Train_MAE_M'], df2018.iloc[i]['Train_MAE_M'], df2019.iloc[i]['Train_MAE_M']]))
        tr_mae_s.append(np.std([df2016.iloc[i]['Train_MAE_M'], df2017.iloc[i]['Train_MAE_M'], df2018.iloc[i]['Train_MAE_M'], df2019.iloc[i]['Train_MAE_M']]))

        v_mae_m.append(np.mean([df2016.iloc[i]['Valid_MAE_M'], df2017.iloc[i]['Valid_MAE_M'], df2018.iloc[i]['Valid_MAE_M'], df2019.iloc[i]['Valid_MAE_M']]))
        v_mae_s.append(np.std([df2016.iloc[i]['Valid_MAE_M'], df2017.iloc[i]['Valid_MAE_M'], df2018.iloc[i]['Valid_MAE_M'], df2019.iloc[i]['Valid_MAE_M']]))

        te_mae_m.append(np.mean([df2016.iloc[i]['Test_MAE_M'], df2017.iloc[i]['Test_MAE_M'], df2018.iloc[i]['Test_MAE_M'], df2019.iloc[i]['Test_MAE_M']]))
        te_mae_s.append(np.std([df2016.iloc[i]['Test_MAE_M'], df2017.iloc[i]['Test_MAE_M'], df2018.iloc[i]['Test_MAE_M'], df2019.iloc[i]['Test_MAE_M']]))

        tr_mape_m.append(np.mean([df2016.iloc[i]['Train_MAPE_M'], df2017.iloc[i]['Train_MAPE_M'], df2018.iloc[i]['Train_MAPE_M'], df2019.iloc[i]['Train_MAPE_M']]))
        tr_mape_s.append(np.std([df2016.iloc[i]['Train_MAPE_M'], df2017.iloc[i]['Train_MAPE_M'], df2018.iloc[i]['Train_MAPE_M'], df2019.iloc[i]['Train_MAPE_M']]))

        v_mape_m.append(np.mean([df2016.iloc[i]['Valid_MAPE_M'], df2017.iloc[i]['Valid_MAPE_M'], df2018.iloc[i]['Valid_MAPE_M'], df2019.iloc[i]['Valid_MAPE_M']]))
        v_mape_s.append(np.std([df2016.iloc[i]['Valid_MAPE_M'], df2017.iloc[i]['Valid_MAPE_M'], df2018.iloc[i]['Valid_MAPE_M'], df2019.iloc[i]['Valid_MAPE_M']]))

        te_mape_m.append(np.mean([df2016.iloc[i]['Test_MAPE_M'], df2017.iloc[i]['Test_MAPE_M'], df2018.iloc[i]['Test_MAPE_M'], df2019.iloc[i]['Test_MAPE_M']]))
        te_mape_s.append(np.std([df2016.iloc[i]['Test_MAPE_M'], df2017.iloc[i]['Test_MAPE_M'], df2018.iloc[i]['Test_MAPE_M'], df2019.iloc[i]['Test_MAPE_M']]))

    df['weeks'] = weeks
    df['Train_MAE_M']  = tr_mae_m
    df['Train_MAE_S']  = tr_mae_s
    df['Valid_MAE_M']  = v_mae_m
    df['Valid_MAE_S']  = v_mae_s
    df['Test_MAE_M']  = te_mae_m
    df['Test_MAE_S']  = te_mae_s
    df['Train_MAPE_M'] = tr_mape_m
    df['Train_MAPE_S'] = tr_mape_s
    df['Valid_MAPE_M'] = v_mape_m
    df['Valid_MAPE_S'] = v_mape_s
    df['Test_MAPE_M']  = te_mape_m
    df['Test_MAPE_S']  = te_mape_s
    
    return df

def timeseries_10m_all(S1, S2, S3): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(3, 2, figsize = (20, 15), sharex=True)


    axs[0, 0].plot(S1["weeks"], S1["Train_MAE"], "-o")
    axs[0, 0].plot(S1["weeks"], S1["Valid_MAE"], "-*")
    axs[0, 0].plot(S1["weeks"], S1["Test_MAE"],  "-d")
    axs[0, 0].set_ylabel('MAE (ton/ac), S1')
    axs[0, 0].set_facecolor('white')
    axs[0, 0].set_ylim([0.7, 2.85])
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(S1["weeks"], S1["Train_MAPE"], "-o", label = 'train')
    axs[0, 1].plot(S1["weeks"], S1["Valid_MAPE"], "-*", label = 'valid')
    axs[0, 1].plot(S1["weeks"], S1["Test_MAPE"],  "-d", label = 'test')
    axs[0, 1].set_ylabel('MAPE (%)')
    axs[0, 1].set_facecolor('white')
    axs[0, 1].set_ylim([0.05, 0.23])
    plt.setp(axs[0, 1].spines.values(), color='k')
    axs[0, 1].legend(loc="upper right")
    
    axs[1, 0].plot(S2["weeks"], S2["Train_MAE_M"], "-o")
    axs[1, 0].fill_between(S2["weeks"], S2["Train_MAE_M"] - S2['Train_MAE_S'], S2["Train_MAE_M"]+ S2['Train_MAE_S'], alpha=.2)
    axs[1, 0].plot(S2["weeks"], S2["Valid_MAE_M"], "-*")
    axs[1, 0].fill_between(S2["weeks"], S2["Valid_MAE_M"] - S2['Valid_MAE_S'], S2["Valid_MAE_M"]+ S2['Valid_MAE_S'], alpha=.2)
    axs[1, 0].plot(S2["weeks"], S2["Test_MAE_M"],  "-d")
    axs[1, 0].fill_between(S2["weeks"], S2["Test_MAE_M"] - S2['Test_MAE_S'], S2["Test_MAE_M"]+ S2['Test_MAE_S'], alpha=.2)
    axs[1, 0].set_ylabel('MAE (ton/ac), S2')
    axs[1, 0].set_facecolor('white')
    axs[1, 0].set_ylim([0.7, 2.85])
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(S2["weeks"], S2["Train_MAPE_M"], "-o")
    axs[1, 1].fill_between(S2["weeks"], S2["Train_MAPE_M"] - S2['Train_MAPE_S'], S2["Train_MAPE_M"]+ S2['Train_MAPE_S'], alpha=.2)
    axs[1, 1].plot(S2["weeks"], S2["Valid_MAPE_M"], "-*")
    axs[1, 1].fill_between(S2["weeks"], S2["Valid_MAPE_M"] - S2['Valid_MAPE_S'], S2["Valid_MAPE_M"]+ S2['Valid_MAPE_S'], alpha=.2)
    axs[1, 1].plot(S2["weeks"], S2["Test_MAPE_M"],  "-d")
    axs[1, 1].fill_between(S2["weeks"], S2["Test_MAPE_M"] - S2['Test_MAPE_S'], S2["Test_MAPE_M"]+ S2['Test_MAPE_S'], alpha=.2)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_ylim([0.05, 0.23])
    axs[1, 1].set_facecolor('white')
    plt.setp(axs[1, 1].spines.values(), color='k')

    axs[2, 0].plot(S3["weeks"], S3["Train_MAE"], "-o")
    axs[2, 0].plot(S3["weeks"], S3["Valid_MAE"], "-*")
    axs[2, 0].plot(S3["weeks"], S3["Test_MAE"],  "-d")
    axs[2, 0].set_ylabel('MAE (ton/ac), S3')
    axs[2, 0].set_facecolor('white')
    axs[2, 0].tick_params(axis='x', rotation=45)
    axs[2, 0].set_ylim([0.7, 2.85])
    plt.setp(axs[2, 0].spines.values(), color='k')
    
    axs[2, 1].plot(S3["weeks"], S3["Train_MAPE"], "-o")
    axs[2, 1].plot(S3["weeks"], S3["Valid_MAPE"], "-*")
    axs[2, 1].plot(S3["weeks"], S3["Test_MAPE"],  "-d")
    axs[2, 1].set_ylabel('MAPE (%)')
    axs[2, 1].set_facecolor('white')
    axs[2, 1].tick_params(axis='x', rotation=45)
    axs[2, 1].set_ylim([0.05, 0.23])
    plt.setp(axs[2, 1].spines.values(), color='k')

def time_series_2d_eval_S2(df2016, df2017, df2018, df2019): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(4, 2, figsize = (20, 20), sharex=True)


    axs[0, 0].plot(df2016["weeks"], df2016["Train_MAE_M"], "-o")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Train_MAE_M"] - df2016['Train_MAE_S'], df2016["Train_MAE_M"]+ df2016['Train_MAE_S'], alpha=.2)
    axs[0, 0].plot(df2016["weeks"], df2016["Valid_MAE_M"], "-*")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Valid_MAE_M"] - df2016['Valid_MAE_S'], df2016["Valid_MAE_M"]+ df2016['Valid_MAE_S'], alpha=.2)
    axs[0, 0].plot(df2016["weeks"], df2016["Test_MAE_M"],  "-d")
    axs[0, 0].fill_between(df2016["weeks"], df2016["Test_MAE_M"] - df2016['Test_MAE_S'], df2016["Test_MAE_M"]+ df2016['Test_MAE_S'], alpha=.2)
    axs[0, 0].set_ylabel('MAE (ton/ac), 2016')
    axs[0, 0].set_facecolor('white')
    axs[0, 0].set_ylim([1, 3])
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(df2016["weeks"], df2016["Train_MAPE_M"], "-o", label = 'train')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Train_MAPE_M"] - df2016['Train_MAPE_S'], df2016["Train_MAPE_M"]+ df2016['Train_MAPE_S'], alpha=.2)
    axs[0, 1].plot(df2016["weeks"], df2016["Valid_MAPE_M"], "-*", label = 'valid')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Valid_MAPE_M"] - df2016['Valid_MAPE_S'], df2016["Valid_MAPE_M"]+ df2016['Valid_MAPE_S'], alpha=.2)
    axs[0, 1].plot(df2016["weeks"], df2016["Test_MAPE_M"],  "-d", label = 'test')
    axs[0, 1].fill_between(df2016["weeks"], df2016["Test_MAPE_M"] - df2016['Test_MAPE_S'], df2016["Test_MAPE_M"]+ df2016['Test_MAPE_S'], alpha=.2)
    axs[0, 1].set_ylabel('MAPE (%)')
    axs[0, 1].set_facecolor('white')
    axs[0, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[0, 1].spines.values(), color='k')
    axs[0, 1].legend(loc="upper right")
    
    axs[1, 0].plot(df2017["weeks"], df2017["Train_MAE_M"], "-o")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Train_MAE_M"] - df2017['Train_MAE_S'], df2017["Train_MAE_M"]+ df2017['Train_MAE_S'], alpha=.2)
    axs[1, 0].plot(df2017["weeks"], df2017["Valid_MAE_M"], "-*")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Valid_MAE_M"] - df2017['Valid_MAE_S'], df2017["Valid_MAE_M"]+ df2017['Valid_MAE_S'], alpha=.2)
    axs[1, 0].plot(df2017["weeks"], df2017["Test_MAE_M"],  "-d")
    axs[1, 0].fill_between(df2017["weeks"], df2017["Test_MAE_M"] - df2017['Test_MAE_S'], df2017["Test_MAE_M"]+ df2017['Test_MAE_S'], alpha=.2)
    axs[1, 0].set_ylabel('MAE (ton/ac), 2017')
    axs[1, 0].set_facecolor('white')
    axs[1, 0].set_ylim([1, 3])
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(df2017["weeks"], df2017["Train_MAPE_M"], "-o")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Train_MAPE_M"] - df2017['Train_MAPE_S'], df2017["Train_MAPE_M"]+ df2017['Train_MAPE_S'], alpha=.2)
    axs[1, 1].plot(df2017["weeks"], df2017["Valid_MAPE_M"], "-*")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Valid_MAPE_M"] - df2017['Valid_MAPE_S'], df2017["Valid_MAPE_M"]+ df2017['Valid_MAPE_S'], alpha=.2)
    axs[1, 1].plot(df2017["weeks"], df2017["Test_MAPE_M"],  "-d")
    axs[1, 1].fill_between(df2017["weeks"], df2017["Test_MAPE_M"] - df2017['Test_MAPE_S'], df2017["Test_MAPE_M"]+ df2017['Test_MAPE_S'], alpha=.2)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    axs[1, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[1, 1].spines.values(), color='k')

    axs[2, 0].plot(df2018["weeks"], df2018["Train_MAE_M"], "-o")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Train_MAE_M"] - df2018['Train_MAE_S'], df2018["Train_MAE_M"]+ df2018['Train_MAE_S'], alpha=.2)
    axs[2, 0].plot(df2018["weeks"], df2018["Valid_MAE_M"], "-*")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Valid_MAE_M"] - df2018['Valid_MAE_S'], df2018["Valid_MAE_M"]+ df2018['Valid_MAE_S'], alpha=.2)
    axs[2, 0].plot(df2018["weeks"], df2018["Test_MAE_M"],  "-d")
    axs[2, 0].fill_between(df2018["weeks"], df2018["Test_MAE_M"] - df2018['Test_MAE_S'], df2018["Test_MAE_M"]+ df2018['Test_MAE_S'], alpha=.2)
    axs[2, 0].set_ylabel('MAE (ton/ac), 2018')
    axs[2, 0].set_facecolor('white')
    axs[2, 0].set_ylim([1, 3])
    plt.setp(axs[2, 0].spines.values(), color='k')
    
    axs[2, 1].plot(df2018["weeks"], df2018["Train_MAPE_M"], "-o")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Train_MAPE_M"] - df2018['Train_MAPE_S'], df2018["Train_MAPE_M"]+ df2018['Train_MAPE_S'], alpha=.2)
    axs[2, 1].plot(df2018["weeks"], df2018["Valid_MAPE_M"], "-*")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Valid_MAPE_M"] - df2018['Valid_MAPE_S'], df2018["Valid_MAPE_M"]+ df2018['Valid_MAPE_S'], alpha=.2)
    axs[2, 1].plot(df2018["weeks"], df2018["Test_MAPE_M"],  "-d")
    axs[2, 1].fill_between(df2018["weeks"], df2018["Test_MAPE_M"]  - df2018['Test_MAPE_S'], df2018["Test_MAPE_M"] + df2018['Test_MAPE_S'], alpha=.2)
    axs[2, 1].set_ylabel('MAPE (%)')
    axs[2, 1].set_facecolor('white')
    axs[2, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[2, 1].spines.values(), color='k')

    axs[3, 0].plot(df2019["weeks"], df2019["Train_MAE_M"], "-o")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Train_MAE_M"] - df2019['Train_MAE_S'], df2019["Train_MAE_M"]+ df2019['Train_MAE_S'], alpha=.2)
    axs[3, 0].plot(df2019["weeks"], df2019["Valid_MAE_M"], "-*")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Valid_MAE_M"] - df2019['Train_MAE_S'], df2019["Valid_MAE_M"]+ df2019['Train_MAE_S'], alpha=.2)
    axs[3, 0].plot(df2019["weeks"], df2019["Test_MAE_M"],  "-d")
    axs[3, 0].fill_between(df2019["weeks"], df2019["Test_MAE_M"] - df2019['Test_MAE_S'], df2019["Test_MAE_M"]+ df2019['Test_MAE_S'], alpha=.2)
    axs[3, 0].set_ylabel('MAE (ton/ac), 2019')
    axs[3, 0].set_facecolor('white')
    axs[3, 0].tick_params(axis='x', rotation=45)
    axs[3, 0].set_ylim([1, 3])
    plt.setp(axs[3, 0].spines.values(), color='k')
    
    axs[3, 1].plot(df2019["weeks"], df2019["Train_MAPE_M"], "-o")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Train_MAPE_M"] - df2019['Train_MAPE_S'], df2019["Train_MAPE_M"]+ df2019['Train_MAPE_S'], alpha=.2)
    axs[3, 1].plot(df2019["weeks"], df2019["Valid_MAPE_M"], "-*")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Valid_MAPE_M"] - df2019['Valid_MAPE_S'], df2019["Valid_MAPE_M"]+ df2019['Valid_MAPE_S'], alpha=.2)
    axs[3, 1].plot(df2019["weeks"], df2019["Test_MAPE_M"],  "-d")
    axs[3, 1].fill_between(df2019["weeks"], df2019["Test_MAPE_M"] - df2019['Test_MAPE_S'], df2019["Test_MAPE_M"]+ df2019['Test_MAPE_S'], alpha=.2)
    axs[3, 1].set_ylabel('MAPE (%)')
    axs[3, 1].set_facecolor('white')
    axs[3, 1].tick_params(axis='x', rotation=45)
    axs[3, 1].set_ylim([0.06, 0.28])
    plt.setp(axs[3, 1].spines.values(), color='k')






#==============================================================================================#
#====================================       Plot           ====================================#
#==============================================================================================#
def regression_metrics(ytrue, ypred):
    """Calculating the evaluation metric based on regression results:
    input:
            ytrue: 
            ypred:
            
    output:
            root mean square error(rmse)
            root square(r^2)
            mean absolute error(mae)
            mean absolute percent error(mape)
            mean of true yield
            mean of prediction yield
    """
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)
    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def triple_scatter_plot(TrainDF, ValDF, TestDF, scatter_mode = None, gridsize = None, fig_save_name = None, save = None): 
    
    
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 3, figsize = (21, 6))
    
    Train_True = TrainDF['ytrue']
    Train_Pred = TrainDF['ypred']
    train_r2, train_mae, train_rmse, train_mape, _,_ = regression_metrics(Train_True, Train_Pred)
        
    axs[0].set_xlim([0, 30])
    axs[0].set_ylim([0, 30])
    axs[0].plot([0, 30], [0, 30],
               '--r', linewidth=2)
    if scatter_mode == 'h':
        axs[0].hexbin(Train_True, Train_Pred, gridsize=(gridsize, gridsize), extent=[0, 30, 0, 30])
    elif scatter_mode == 'c':
        axs[0].scatter(Train_True, Train_Pred, alpha=0.2)
        #sns.lmplot(x = 'ytrue', y = 'ypred', data=TrainDF, ax=axs[0])
    
    axs[0].set_xlabel('Measured')
    axs[0].set_ylabel('Predicted')
    axs[0].grid(False)
    axs[0].set_facecolor('white')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+ '\n' + r'$MAPE={:.2f}$').format(train_r2, train_mae, train_rmse, train_mape)
    axs[0].legend([extra], [scores], loc='upper left')
    axs[0].set_title('Train Data')
    #===============================================================================
    Val_True = ValDF['ytrue']
    Val_Pred = ValDF['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(Val_True, Val_Pred)
    axs[1].set_xlim([0, 30])
    axs[1].set_ylim([0, 30])
    axs[1].plot([0, 30], [0, 30],
               '--r', linewidth=2)
    if scatter_mode == 'h':
        axs[1].hexbin(Val_True, Val_Pred, gridsize=(gridsize,gridsize), extent=[0, 30, 0, 30])
        plt.plot([0, 30], [0, 30],'--r', linewidth=2)
    elif scatter_mode == 'c':
        axs[1].scatter(Val_True, Val_Pred, alpha=0.2)

    
    axs[1].set_xlabel('Measured')
    #axs[1].set_ylabel('Predicted')
    axs[1].grid(False)
    axs[1].set_facecolor('white')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2,val_mae,  val_rmse, val_mape)
    axs[1].legend([extra], [scores], loc='upper left')
    axs[1].set_title('Validation Data')
    #============================================================================
    Test_True = TestDF['ytrue']
    Test_Pred = TestDF['ypred'] 
    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(Test_True, Test_Pred)
    axs[2].set_xlim([0, 30])
    axs[2].set_ylim([0, 30])
    axs[2].plot([0, 30], [0, 30],
               '--r', linewidth=2)
    #
    if scatter_mode == 'h':
        axs[2].hexbin(Test_True, Test_Pred, gridsize=(gridsize,gridsize), extent=[0, 30, 0, 30])
    elif scatter_mode == 'c':
        axs[2].scatter(Test_True, Test_Pred, alpha=0.2)
        #sns.lmplot('ytrue', 'ypred', data=TestDF,  ax=axs[2])
    axs[2].set_xlabel('Measured')
    #axs[2].set_ylabel('Predicted')
    axs[2].grid(False)
    axs[2].set_facecolor('white')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+ '\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    axs[2].legend([extra], [scores], loc='upper left')
    axs[2].set_title('Test Data')
    
    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)



def block_eval_barplot_S4(blocks_csv, cultivar_list = None, block_list = None):

    
    blocks_csv = blocks_csv[blocks_csv['cultivar'].isin(cultivar_list)]
    blocks_csv = blocks_csv[blocks_csv['block'].isin(block_list)]
    blocks_csv = blocks_csv.sort_values(by = ['cultivar', 'block'])
    
    colors = ['#1f77b4', '#1f77b4',
                '#ff7f0e', '#ff7f0e',
                '#2ca02c', '#2ca02c','#2ca02c','#d62728', '#d62728',
                '#9467bd',  '#9467bd', '#9467bd', '#8c564b', '#8c564b','#8c564b',
                '#e377c2', '#e377c2','#e377c2', '#7f7f7f']
    
    colours_labels = {"CS": "#1f77b4", "Ch": "#ff7f0e", "MB": "#2ca02c", "Me": "#d62728","MoA": "#9467bd", "Res": "#8c564b","Sym": "#e377c2", "Syr": "#7f7f7f"}


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (24, 8))


    plt.rcParams["axes.grid"] = False
    mae = blocks_csv.plot.bar(x = 'block', y = 'MAE', color = colors, width = 0.4, ax = ax1)
    ax1.legend([Patch(facecolor=colours_labels['CS']), Patch(facecolor=colours_labels['Ch']),
    Patch(facecolor=colours_labels['MB']),Patch(facecolor=colours_labels['Me']),Patch(facecolor=colours_labels['MoA']),Patch(facecolor=colours_labels['Res']),
    Patch(facecolor=colours_labels['Sym']), Patch(facecolor=colours_labels['Syr'])], ["CS", "Ch", "MB", "Me", "MoA", "Res", "Sym", "Syr"], loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.2))

    for i, v in enumerate(blocks_csv["MAE"].iteritems()):        
        ax1.text(i ,v[1], "{:.2f}".format(v[1]), color='k', position = (i-0.15, v[1] + 0.3), fontsize = 'x-small', rotation=90)
    ax1.set_ylim([0, blocks_csv['MAE'].max()+2])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set(ylabel='MAE')
    ax1.set_facecolor('white')
    ax1.axvline(x = 1.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 3.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 8.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 11.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 14.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 17.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    #ax1.axvline(x = 27.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    #ax1.axvline(x = 31.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.setp(ax1.spines.values(), color='k')


    mape = blocks_csv.plot.bar(x = 'block', y = 'MAPE', color = colors, width = 0.4, ax = ax2)
    for j, m in enumerate(blocks_csv["MAPE"].iteritems()):        
        ax2.text(j ,m[1], "{:.2f}".format(m[1]), color='k', position = (j - 0.15, m[1] + 0.01), fontsize = 'x-small', rotation=90)
    ax2.set_ylim([0, blocks_csv['MAPE'].max()+0.1])
    ax2.set(xlabel = 'block', ylabel='MAPE')
    ax2.set_facecolor('white')
    plt.setp(ax2.spines.values(), color='k')
    ax2.legend().set_visible(False)
    plt.xticks(rotation=90, fontsize = 14)
    ax2.axvline(x = 1.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 3.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 8.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 11.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 14.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 17.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    #ax2.axvline(x = 27.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    #ax2.axvline(x = 31.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.show()
    None    


def block_eval_barplot(blocks_csv, cultivar_list = None, block_list = None):

    
    #blocks_csv = blocks_csv[blocks_csv['cultivar'].isin(cultivar_list)]
    #blocks_csv = blocks_csv[blocks_csv['block'].isin(block_list)]
    blocks_csv = blocks_csv.sort_values(by = ['cultivar', 'block'])
    
    colors = ['#1f77b4', '#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4', 
                '#ff7f0e', '#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e',
                '#2ca02c', '#2ca02c','#2ca02c','#2ca02c', '#d62728', '#d62728','#d62728', 
                '#9467bd',  '#9467bd', '#9467bd', '#8c564b', '#8c564b','#8c564b','#8c564b',
                '#e377c2', '#e377c2','#e377c2','#e377c2', '#7f7f7f', '#7f7f7f', '#7f7f7f']
    
    colours_labels = {"CS": "#1f77b4", "Ch": "#ff7f0e", "MB": "#2ca02c", "Me": "#d62728","MoA": "#9467bd", "Res": "#8c564b","Sym": "#e377c2", "Syr": "#7f7f7f"}


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (24, 8))


    plt.rcParams["axes.grid"] = False
    mae = blocks_csv.plot.bar(x = 'block', y = 'MAE', color = colors, width = 0.4, ax = ax1)
    ax1.legend([Patch(facecolor=colours_labels['CS']), Patch(facecolor=colours_labels['Ch']),
    Patch(facecolor=colours_labels['MB']),Patch(facecolor=colours_labels['Me']),Patch(facecolor=colours_labels['MoA']),Patch(facecolor=colours_labels['Res']),
    Patch(facecolor=colours_labels['Sym']), Patch(facecolor=colours_labels['Syr'])], ["CS", "Ch", "MB", "Me", "MoA", "Res", "Sym", "Syr"], loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.2))

    for i, v in enumerate(blocks_csv["MAE"].iteritems()):        
        ax1.text(i ,v[1], "{:.2f}".format(v[1]), color='k', position = (i-0.15, v[1] + 0.3), fontsize = 'x-small', rotation=90)
    ax1.set_ylim([0, blocks_csv['MAE'].max()+2])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set(ylabel='MAE')
    ax1.set_facecolor('white')
    ax1.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 9.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 13.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 17.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 20.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 23.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 27.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 31.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.setp(ax1.spines.values(), color='k')


    mape = blocks_csv.plot.bar(x = 'block', y = 'MAPE', color = colors, width = 0.4, ax = ax2)
    for j, m in enumerate(blocks_csv["MAPE"].iteritems()):        
        ax2.text(j ,m[1], "{:.2f}".format(m[1]), color='k', position = (j - 0.15, m[1] + 0.01), fontsize = 'x-small', rotation=90)
    ax2.set_ylim([0, blocks_csv['MAPE'].max()+0.1])
    ax2.set(xlabel = 'block', ylabel='MAPE')
    ax2.set_facecolor('white')
    plt.setp(ax2.spines.values(), color='k')
    ax2.legend().set_visible(False)
    plt.xticks(rotation=90, fontsize = 14)
    ax2.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 9.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 13.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 17.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 20.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 23.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 27.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 31.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.show()
    None    

def time_series_block_eval(df, block_id = None): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    test_r2_list, test_mae_list, test_rmse_list, test_mape_list     = [], [], [], []
    
    for i in range(len(Weeks)): 
        test_r2, test_mae, test_rmse, test_mape, _, _     = regression_metrics(df['ytrue'], df.iloc[:, i+5])
        test_r2_list.append(test_r2)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_mape_list.append(test_mape)
    
    results['weeks']      = Weeks
    results['Test_R2']    = test_r2_list
    results['Test_MAE']   = test_mae_list
    results['Test_RMSE']  = test_rmse_list
    results['Test_MAPE']  = test_mape_list
 

    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 2, figsize = (20, 5), sharex=True)

    axs[0].plot(results["weeks"], results["Test_MAE"],  "-d", label = 'test')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel('MAE')
    
    axs[1].plot(results["weeks"], results["Test_MAPE"],  "-d")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].set_ylabel('MAPE')
    
    plt.suptitle(block_id)

def thriple_time_series_block_eval(df1, df2, df3): 
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    test_mae_list1, test_mape_list1     = [], []
    test_mae_list2, test_mape_list2     = [], []
    test_mae_list3, test_mape_list3     = [], []
    
    for i in range(len(Weeks)): 
        test_r2_1, test_mae_1, test_rmse_1, test_mape_1, _, _     = regression_metrics(df1['ytrue'], df1.iloc[:, i+5])
        test_mae_list1.append(test_mae_1)
        test_mape_list1.append(test_mape_1)

        test_r2_2, test_mae_2, test_rmse_2, test_mape_2, _, _     = regression_metrics(df2['ytrue'], df2.iloc[:, i+5])
        test_mae_list2.append(test_mae_2)
        test_mape_list2.append(test_mape_2)

        test_r2_3, test_mae_3, test_rmse_3, test_mape_3, _, _     = regression_metrics(df3['ytrue'], df3.iloc[:, i+5])
        test_mae_list3.append(test_mae_3)
        test_mape_list3.append(test_mape_3)
    
    results['weeks']      = Weeks
    results['Test_MAE_1']   = test_mae_list1
    results['Test_MAPE_1']  = test_mape_list1
    results['Test_MAE_2']   = test_mae_list2
    results['Test_MAPE_2']  = test_mape_list2
    results['Test_MAE_3']   = test_mae_list3
    results['Test_MAPE_3']  = test_mape_list3
 

    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 2, figsize = (20, 5), sharex=True)

    axs[0].plot(results["weeks"], results["Test_MAE_1"],  "-*", color = 'black')
    axs[0].plot(results["weeks"], results["Test_MAE_2"],  "-o", color = 'blue')
    axs[0].plot(results["weeks"], results["Test_MAE_3"],  "-d", color = 'red')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].set_ylabel('MAE (ton/ac)')
    
    axs[1].plot(results["weeks"], results["Test_MAPE_1"],  "-*", color = 'black', label = 'B1-2017')
    axs[1].plot(results["weeks"], results["Test_MAPE_2"],  "-o", color = 'blue', label = 'B2-2019')
    axs[1].plot(results["weeks"], results["Test_MAPE_3"],  "-d", color = 'red', label = 'B3-2018')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].legend(loc='upper right')
    axs[1].set_ylabel('MAPE (%)')

    

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

sns.set_theme(style='white')


def train_val_test_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None):

    
    if week == None: 
        week_pred = 'ypred'
    else: 
        week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4,  
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'})#,,  marginal_kws = dict(bins = np.arange(0, 50000))

    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')

    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac) - Train')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = mincnt) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac) - Validation')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================
    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']
    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac)')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21,7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)

    plt.show()



def yield_true_pred_plot(ytrue, ypred, min_v = None, max_v= None):

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(1, 4, figsize = (24, 8))

    img1 = axs[0].imshow(ytrue)
    axs[0].set_title('Yield Observation', fontsize = 14)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1,  cax=cax)
    img1.set_clim(min_v, max_v)

    img2 = axs[1].imshow(ypred)
    axs[1].set_title('Yield Prediction', fontsize = 14)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img2, cax=cax)
    img2.set_clim(min_v, max_v)
    axs[1].get_yaxis().set_visible(False)

    #mae_map = image_subtract(ytrue, ypred) 
    #mape_map = image_subtract(ytrue, ypred) 
    mae_map, mape_map = image_mae_mape_map(ytrue, ypred)
    img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[2].set_title('MAE Map (ton/ac)', fontsize = 14)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img3, cax=cax)
    img3.set_clim(-1, np.max(mae_map))
    axs[2].get_yaxis().set_visible(False)
    #axs[2].get_xaxis().set_visible(False)
    #axs[2].set_xlabel(xlabel_text)

    img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[3].set_title('MAPE Map (%)', fontsize = 14)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar3 =fig.colorbar(img4, cax=cax)
    img4.set_clim(-5, 20)
    axs[3].get_yaxis().set_visible(False)

    #return mape_map
    #plt.savefig('./imgs/B186_1m.png', dpi = 300)


def image_subtract(img1, img2): 
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    output1 = np.empty_like(img1_flat)
    output2 = np.empty_like(img1_flat)
    for i in range(len(img1_flat)):
        if (img1_flat[i] == -1) and (img2_flat[i] == -1):
            output1[i] = -1
            output2[i] = -100
        else: 
            output1[i] = img1_flat[i] - img2_flat[i]

    out = output1.reshape(img1.shape[0], img1.shape[1])
    return out

def image_mae_mape_map(ytrue, ypred): 

    w = ytrue.shape[0]
    h = ytrue.shape[1] 

    ytrue_flat = ytrue.ravel()
    ypred_flat = ypred.ravel()
    out_mae    = np.empty_like(ytrue_flat)
    out_mape   = np.empty_like(ytrue_flat)

    for i in range(len(ytrue_flat)):
        if (ytrue_flat[i] == -1) and (ypred_flat[i] == -1):
            out_mae[i]  = -10
            out_mape[i] = -10
        elif abs(ytrue_flat[i] - ypred_flat[i]) > 5:
            out_mae[i]  = -10
            out_mape[i] = -10
        else: 
            out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
            out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

    out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
    out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

    return out1, out2





def PSNR_SSIM_Calc(ytrue, ypred): 
    img1_flat = ytrue.ravel()
    img2_flat = ypred.ravel()

    INDX = np.where(ytrue[3:-3, 3:-3] == -1)
    


    img1_flat_non = img1_flat[img1_flat != -1]
    img2_flat_non = img2_flat[img2_flat != -1]

    #MSE  = mean_squared_error(img1_flat_non, img2_flat_non)
    MSE, PSNR = PSNR_Func(img1_flat_non, img2_flat_non)
    ssim_value, ssim_map = calculate_ssim(ytrue, ypred)

    ssim_map_modified = np.copy(ssim_map)
    ssim_map_modified[INDX] = 0

    ssim_value2 = ssim_map_modified.mean()
    #print(f"{ytrue.shape}|{ypred.shape}|{ssim_map.shape}")
    return MSE, PSNR, ssim_value2, ssim_map

from math import log10, sqrt

def PSNR_Func(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 30.0
    #psnr = 10 * log10(max_pixel**2 / sqrt(mse))
    psnr = 10 * log10(max_pixel**2 / mse)
    return mse, psnr



def ssim(img1, img2):

    C1 = (0.01 * 30)**2
    C2 = (0.03 * 30)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    f = 3
    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REPLICATE)[f:-f , f:-f]  # valid
    mu1_ = convolve2d(img1, window, mode='same', boundary = 'symm')
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REPLICATE)[f:-f , f:-f]
    mu2_ = convolve2d(img2, window, mode='same', boundary = 'symm')
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2


    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=cv2.BORDER_REPLICATE)[f:-f , f:-f]   - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=cv2.BORDER_REPLICATE)[f:-f , f:-f]   - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=cv2.BORDER_REPLICATE)[f:-f , f:-f] - mu1_mu2

    #sigma1_sq = convolve2d(img1**2,  window, mode='same', boundary = 'fill')  - mu1_sq
    #sigma2_sq = convolve2d(img2**2, window, mode='same', boundary = 'fill')  - mu2_sq
    #sigma12   = convolve2d(img1 * img2, window, mode='same', boundary = 'fill') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))



    return ssim_map.mean(), ssim_map


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def scenarios_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None):

    week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'}, marginal_kws = {"ymax": 50000}) # , , 
    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    #for patch in TR.ax_marg_x.patches:
     #   patch.set_clim(0, 50000)

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')



    #TR.ax_marg_y.patches.ylim(0, 50000)
    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac), S1')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = 4000) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac), S2')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================
    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']
    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac), S3')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21,7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)

    plt.show()


def block_cultivar_test_csv_results_2d(test_df, week = None, save_dir= None, save_csv_name = None):

    test_blocks     = test_df.groupby(by = ['block'])

    block_df    = pd.DataFrame()
    b, c, test_r2_b, test_mae_b, test_rmse_b, test_mape_b = [], [], [], [], [], [] 
    
    cultivar_df = pd.DataFrame()
    cu, test_r2_c, test_mae_c, test_rmse_c, test_mape_c   = [], [], [], [], []
        
    for block, tedf in test_blocks:
        b.append(block)
        cultivar_id     = tedf.iloc[0]['cultivar']
        cultivar_id     = int(cultivar_id)
        cultivar_id     = str(cultivar_id)
        key_ext         = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d          = key_ext.get(cultivar_id)
        cultivar_type   = list_d[0]
        
        c.append(cultivar_type)
        b_r_square3, b_mae3, b_rmse3, b_mape3, mean_ytrue, mean_ypred =  regression_metrics(tedf['ytrue'], tedf[week])
        test_r2_b.append(b_r_square3)
        test_mae_b.append(b_mae3)
        test_rmse_b.append(b_rmse3)
        test_mape_b.append(b_mape3)
        
        
    block_df['block']    = b
    block_df['cultivar'] = c
    block_df['R2']  = test_r2_b
    block_df['MAE']  = test_mae_b
    block_df['RMSE'] = test_rmse_b
    block_df['MAPE'] = test_mape_b
    block_df = block_df.round(decimals = 2)
    block_df.to_csv(os.path.join(save_dir, save_csv_name + '_blocks.csv'))
    
    ### Cultvars 
    test_cultivars  = test_df.groupby(by = ['cultivar'])
    for cul, tedf1 in test_cultivars:
        cultivar_id   = int(cul)
        cultivar_id   = str(cultivar_id)
        key_ext       = {key: cultivars_[key] for key in cultivars_.keys() & {cultivar_id}}
        #print(key_ext)
        list_d        = key_ext.get(cultivar_id)
        cultivar_type = list_d[0]
        
        cu.append(cultivar_type)
        b_r_square33, b_mae33, b_rmse33, b_mape33, mean_ytrue, mean_ypred =  regression_metrics(tedf1['ytrue'], tedf1[week])
        test_r2_c.append(b_r_square33)
        test_mae_c.append(b_mae33)
        test_rmse_c.append(b_rmse33)
        test_mape_c.append(b_mape33)

    cultivar_df['cultivar'] = cu
    cultivar_df['R2']  = test_r2_c
    cultivar_df['MAE']  = test_mae_c
    cultivar_df['RMSE'] = test_rmse_c
    cultivar_df['MAPE'] = test_mape_c
    cultivar_df = cultivar_df.round(decimals = 2)
    cultivar_df.to_csv(os.path.join(save_dir, save_csv_name + '_cultivars.csv'))
    
    return block_df, cultivar_df


def block_eval_barplot_S3(blocks_csv, cultivar_list = None, block_list = None):

    #blocks_csv = blocks_csv[(blocks_csv.block != 262016) & (blocks_csv.block != 262017)& (blocks_csv.block != 1932017)]
    #blocks_csv = blocks_csv[blocks_csv['cultivar'].isin(cultivar_list)]
    #blocks_csv = blocks_csv[blocks_csv['block'].isin(block_list)]
    blocks_csv = blocks_csv.sort_values(by = ['cultivar', 'block'])
    
    colors = ['#1f77b4', '#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4','#1f77b4', 
                '#ff7f0e', '#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e','#ff7f0e',
                '#2ca02c', '#2ca02c','#2ca02c','#2ca02c', 
                '#d62728', '#d62728','#d62728', 
                '#9467bd',  '#9467bd', '#9467bd', '#9467bd', 
                '#8c564b', '#8c564b','#8c564b','#8c564b','#8c564b', '#8c564b','#8c564b','#8c564b',
                '#e377c2', '#e377c2','#e377c2','#e377c2', 
                '#7f7f7f', '#7f7f7f', '#7f7f7f']
    
    colours_labels = {"CS": "#1f77b4", "Ch": "#ff7f0e", "MB": "#2ca02c", "Me": "#d62728","MoA": "#9467bd", "Res": "#8c564b","Sym": "#e377c2", "Syr": "#7f7f7f"}


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (24, 8))


    plt.rcParams["axes.grid"] = False
    mae = blocks_csv.plot.bar(x = 'block', y = 'MAE', color = colors, width = 0.4, ax = ax1)
    ax1.legend([Patch(facecolor=colours_labels['CS']), Patch(facecolor=colours_labels['Ch']),
    Patch(facecolor=colours_labels['MB']),Patch(facecolor=colours_labels['Me']),Patch(facecolor=colours_labels['MoA']),Patch(facecolor=colours_labels['Res']),
    Patch(facecolor=colours_labels['Sym']), Patch(facecolor=colours_labels['Syr'])], ["CS", "Ch", "MB", "Me", "MoA", "Res", "Sym", "Syr"], loc='upper center', ncol=8, bbox_to_anchor=(0.5, 1.2))

    for i, v in enumerate(blocks_csv["MAE"].iteritems()):        
        ax1.text(i ,v[1], "{:.2f}".format(v[1]), color='k', position = (i-0.15, v[1] + 0.3), fontsize = 'x-small', rotation=90)
    ax1.set_ylim([0, blocks_csv['MAE'].max()+2])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set(ylabel='MAE (ton/ac)')
    ax1.set_facecolor('white')
    ax1.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 10.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 14.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 18.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 22.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 25.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 29.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 33.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax1.axvline(x = 37.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax1.axvline(x = 41.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.setp(ax1.spines.values(), color='k')


    mape = blocks_csv.plot.bar(x = 'block', y = 'MAPE', color = colors, width = 0.4, ax = ax2)
    for j, m in enumerate(blocks_csv["MAPE"].iteritems()):        
        ax2.text(j ,m[1], "{:.2f}".format(m[1]), color='k', position = (j - 0.15, m[1] + 0.01), fontsize = 'x-small', rotation=90)
    ax2.set_ylim([0, blocks_csv['MAPE'].max()+0.1])
    ax2.set(xlabel = 'block', ylabel='MAPE (%)')
    ax2.set_facecolor('white')
    plt.setp(ax2.spines.values(), color='k')
    ax2.legend().set_visible(False)
    plt.xticks(rotation=90, fontsize = 14)
    ax2.axvline(x = 2.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 6.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 10.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 14.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 18.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 22.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 25.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 29.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 33.5, linestyle = '--', color = '#C5C9C7', label = 'axvline - full height')
    ax2.axvline(x = 37.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    ax2.axvline(x = 41.5, linestyle = '--', color = '#929591', label = 'axvline - full height')
    plt.show()
    None    



def agg_regression_metrics(ytrue, ypred, scale):
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    
    ytrue_agg = aggregate(ytrue, scale)
    ypred_agg = aggregate(ypred, scale)
    
    rmse       = np.sqrt(mean_squared_error(ytrue_agg, ypred_agg))
    r_square   = r2_score(ytrue_agg, ypred_agg)
    mae        = mean_absolute_error(ytrue_agg, ypred_agg)
    
    return [rmse, r_square, mae, mean_ytrue, mean_ypred] 

  

def block_true_pred_mtx_timeseries(df, block_id, aggregation = None, spatial_resolution  = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)
    
    print(this_block_df.shape)
    true_out = np.full((block_x_size, block_y_size), -1)
    pred_out1 = np.full((block_x_size, block_y_size), -1)   
    pred_out2 = np.full((block_x_size, block_y_size), -1) 
    pred_out3 = np.full((block_x_size, block_y_size), -1) 
    pred_out4 = np.full((block_x_size, block_y_size), -1) 
    pred_out5 = np.full((block_x_size, block_y_size), -1) 
    pred_out6 = np.full((block_x_size, block_y_size), -1) 
    pred_out7 = np.full((block_x_size, block_y_size), -1)
    pred_out8 = np.full((block_x_size, block_y_size), -1) 
    pred_out9 = np.full((block_x_size, block_y_size), -1) 
    pred_out10 = np.full((block_x_size, block_y_size), -1) 
    pred_out11 = np.full((block_x_size, block_y_size), -1) 
    pred_out12 = np.full((block_x_size, block_y_size), -1) 
    pred_out13 = np.full((block_x_size, block_y_size), -1) 
    pred_out14 = np.full((block_x_size, block_y_size), -1) 
    pred_out15 = np.full((block_x_size, block_y_size), -1) 


    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 

            if len(new) > 0:
                true_out[x, y] = new['ytrue'].mean()

                pred_out1[x, y] = new['ypred_w1'].max()
                pred_out2[x, y] = new['ypred_w2'].max()
                pred_out3[x, y] = new['ypred_w3'].max()
                pred_out4[x, y] = new['ypred_w4'].max()
                pred_out5[x, y] = new['ypred_w5'].max()
                pred_out6[x, y] = new['ypred_w6'].max()
                pred_out7[x, y] = new['ypred_w7'].max()
                pred_out8[x, y] = new['ypred_w8'].max()
                pred_out9[x, y] = new['ypred_w9'].max()
                pred_out10[x, y] = new['ypred_w10'].max()
                pred_out11[x, y] = new['ypred_w11'].max()
                pred_out12[x, y] = new['ypred_w12'].max()
                pred_out13[x, y] = new['ypred_w13'].max()
                pred_out14[x, y] = new['ypred_w14'].max()
                pred_out15[x, y] = new['ypred_w15'].max()



        #from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(4, 4, figsize = (24, 24))

    img1 = axs[0, 0].imshow(true_out)
    axs[0, 0].set_title('Yield Observation', fontsize = 16)
    divider = make_axes_locatable(axs[0, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar1 = fig.colorbar(img1,  cax=cax)
    img1.set_clim(5, 20)

    img2 = axs[0, 1].imshow(pred_out15)
    tr_r2_1, tr_mae_1, tr_rmse_1, tr_mape_1, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w1'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_1, tr_mape_1)
    axs[0, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 1].set_title('Yield Prediction Week 1', fontsize = 16)
    divider = make_axes_locatable(axs[0, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar2 =fig.colorbar(img2, cax=cax)
    img2.set_clim(5, 20)
    axs[0, 1].get_yaxis().set_visible(False)


    img3 = axs[0, 2].imshow(pred_out2)
    tr_r2_2, tr_mae_2, tr_rmse_2, tr_mape_2, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w2'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_2, tr_mape_2)
    axs[0, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 2].set_title('Yield Prediction Week 2', fontsize = 16)
    divider = make_axes_locatable(axs[0, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar3 =fig.colorbar(img3, cax=cax)
    img3.set_clim(5, 20)
    axs[0, 2].get_yaxis().set_visible(False)


    img4 = axs[0, 3].imshow(pred_out3)
    tr_r2_3, tr_mae_3, tr_rmse_3, tr_mape_3, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w3'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_3, tr_mape_3)
    axs[0, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[0, 3].set_title('Yield Prediction Week 3', fontsize = 16)
    divider = make_axes_locatable(axs[0, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar4 =fig.colorbar(img4, cax=cax)
    img4.set_clim(5, 20)
    axs[0, 3].get_yaxis().set_visible(False)


    img5 = axs[1, 0].imshow(pred_out4)
    tr_r2_4, tr_mae_4, tr_rmse_4, tr_mape_4, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w4'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_4, tr_mape_4)
    axs[1, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 0].set_title('Yield Prediction Week 4', fontsize = 16)
    divider = make_axes_locatable(axs[1, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar5 =fig.colorbar(img5, cax=cax)
    img5.set_clim(5, 20)


    img6 = axs[1, 1].imshow(pred_out5)
    tr_r2_5, tr_mae_5, tr_rmse_5, tr_mape_5, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w5'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_5, tr_mape_5)
    axs[1, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 1].set_title('Yield Prediction Week 5', fontsize = 16)
    divider = make_axes_locatable(axs[1, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar6 =fig.colorbar(img6, cax=cax)
    img6.set_clim(5, 20)
    axs[1, 1].get_yaxis().set_visible(False)



    img7 = axs[1, 2].imshow(pred_out6)
    tr_r2_6, tr_mae_6, tr_rmse_6, tr_mape_6, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w6'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_6, tr_mape_6)
    axs[1, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[1, 2].set_title('Yield Prediction Week 6', fontsize = 16)
    divider = make_axes_locatable(axs[1, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar7 =fig.colorbar(img7, cax=cax)
    img7.set_clim(5, 20)
    axs[1, 2].get_yaxis().set_visible(False)



    img8 = axs[1, 3].imshow(pred_out7)
    tr_r2_7, tr_mae_7, tr_rmse_7, tr_mape_7, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w7'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_7, tr_mape_7)
    axs[1, 3].legend([extra], [scores], loc='upper left', fontsize = 16)
    axs[1, 3].set_title('Yield Prediction Week 7', fontsize = 16)
    divider = make_axes_locatable(axs[1, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar8 =fig.colorbar(img8, cax=cax)
    img8.set_clim(5, 20)
    axs[1, 3].get_yaxis().set_visible(False)


    img9 = axs[2, 0].imshow(pred_out8)
    tr_r2_8, tr_mae_8, tr_rmse_8, tr_mape_8, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w8'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_8, tr_mape_8)
    axs[2, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 0].set_title('Yield Prediction Week 8', fontsize = 14)
    #divider = make_axes_locatable(axs[2, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar9 =fig.colorbar(img9, cax=cax)
    img9.set_clim(5, 20)




    img10 = axs[2, 1].imshow(pred_out9)
    tr_r2_9, tr_mae_9, tr_rmse_9, tr_mape_9, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w9'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_9, tr_mape_9)
    axs[2, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 1].set_title('Yield Prediction Week 9', fontsize = 16)
    #divider = make_axes_locatable(axs[2, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar10 =fig.colorbar(img10, cax=cax)
    img10.set_clim(5, 20)
    axs[2, 1].get_yaxis().set_visible(False)



    img11 = axs[2, 2].imshow(pred_out10)
    tr_r2_10, tr_mae_10, tr_rmse_10, tr_mape_10, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w10'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}'+ '\n' + r'$MAPE={:.2f}$').format(tr_mae_10, tr_mape_10)
    axs[2, 2].legend([extra], [scores], loc='upper left', fontsize = 16)
    axs[2, 2].set_title('Yield Prediction Week 10', fontsize = 16)
    #divider = make_axes_locatable(axs[2, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar11 =fig.colorbar(img11, cax=cax)
    img11.set_clim(5, 20)
    axs[2, 2].get_yaxis().set_visible(False)



    img12 = axs[2, 3].imshow(pred_out11)
    tr_r2_11, tr_mae_11, tr_rmse_11, tr_mape_11, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w11'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_11, tr_mape_11)
    axs[2, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[2, 3].set_title('Yield Prediction Week 11', fontsize = 16)
    divider = make_axes_locatable(axs[2, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar12 =fig.colorbar(img12, cax=cax)
    img12.set_clim(5, 20)
    axs[2, 3].get_yaxis().set_visible(False)



    img13 = axs[3, 0].imshow(pred_out12)
    tr_r2_12, tr_mae_12, tr_rmse_12, tr_mape_12, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w12'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_12, tr_mape_12)
    axs[3, 0].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 0].set_title('Yield Prediction Week 12', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar13 =fig.colorbar(img13, cax=cax)
    img13.set_clim(5, 20)



    img14 = axs[3, 1].imshow(pred_out13)
    tr_r2_13, tr_mae_13, tr_rmse_13, tr_mape_13, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w13'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_13, tr_mape_13)
    axs[3, 1].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 1].set_title('Yield Prediction Week 13', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 1])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar14 =fig.colorbar(img14, cax=cax)
    img14.set_clim(5, 20)
    axs[3, 1].get_yaxis().set_visible(False)


    img15 = axs[3, 2].imshow(pred_out14)
    tr_r2_14, tr_mae_14, tr_rmse_14, tr_mape_14, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w14'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_14, tr_mape_14)
    axs[3, 2].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 2].set_title('Yield Prediction Week 14', fontsize = 16)
    #divider = make_axes_locatable(axs[3, 2])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar15 =fig.colorbar(img15, cax=cax)
    img15.set_clim(5, 20)
    axs[3, 2].get_yaxis().set_visible(False)



    img16 = axs[3, 3].imshow(pred_out15)
    tr_r2_15, tr_mae_15, tr_rmse_15, tr_mape_15, _,_ = regression_metrics(this_block_df['ytrue'], this_block_df['ypred_w15'])
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'MAE={:.2f}' + '\n' + r'$MAPE={:.2f}$').format(tr_mae_15, tr_mape_15)
    axs[3, 3].legend([extra], [scores], loc='upper left', fontsize = 15)
    axs[3, 3].set_title('Yield Prediction Week 15', fontsize = 16)
    divider = make_axes_locatable(axs[3, 3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar16 =fig.colorbar(img16, cax=cax)
    img16.set_clim(5, 20)
    axs[3, 3].get_yaxis().set_visible(False)

    fig.tight_layout()
    return this_block_df 


        
def block_true_pred_mtx(df, block_id, aggregation = None, spatial_resolution  = None, scale = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)
    
    print(this_block_df.shape)
    pred_out = np.full((block_x_size, block_y_size), -1) 
    true_out = np.full((block_x_size, block_y_size), -1)  

    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 
            if len(new) > 0:
                pred_out[x, y] = new['ypred_w15'].min()
                true_out[x, y] = new['ytrue'].mean()



    if aggregation is True: 
        
        print(f"{pred_out.shape}|{true_out.shape}")
        pred_agg = aggregate2(pred_out, scale)
        true_agg = aggregate2(true_out, scale)
        print(f"Agg: {pred_agg.shape}|{true_agg.shape}")
        
        df_agg = pd.DataFrame() 
        
        flat_pred_agg = pred_agg.flatten()
        flat_pred_agg = flat_pred_agg[flat_pred_agg != -1]
        
        flat_true_agg = true_agg.flatten()
        flat_true_agg = flat_true_agg[flat_true_agg != -1] 
              
        df_agg['ytrue'] = flat_true_agg
        df_agg['ypred'] = flat_pred_agg 
        
        return df_agg, true_agg, pred_agg
    else:
        return this_block_df, true_out, pred_out            


def scenario_aggregation_results(df, spatial_resolutin = None):
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')

    output = pd.DataFrame()
    blocks = []
    b_true = []
    b_out_w1, b_out_w2, b_out_w3, b_out_w4, b_out_w5 = [], [], [], [], [] 
    b_out_w6, b_out_w7, b_out_w8, b_out_w9, b_out_w10 = [], [], [], [], [] 
    b_out_w11, b_out_w12, b_out_w13, b_out_w14, b_out_w15 = [], [], [], [], [] 

    for block_id, block_df in blocks_df:
        name_split = os.path.split(str(block_id))[-1]
        root_name  = name_split.replace(name_split[-4:], '')
        year       = name_split[-4:]
    
        if len(root_name) == 1:
            this_block_name = 'LIV_00' + root_name + '_' + year
        elif len(root_name) == 2:
            this_block_name = 'LIV_0' + root_name + '_' + year
        elif len(root_name) == 3:
            this_block_name = 'LIV_' + root_name + '_' + year


        res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
        list_d        = res.get(this_block_name)
        block_x_size  = int(list_d[0]/10)
        block_y_size  = int(list_d[1]/10)
        
        
        true_out    = np.full((block_x_size, block_y_size), -1) 
        pred_out_w1 = np.full((block_x_size, block_y_size), -1)  
        pred_out_w2 = np.full((block_x_size, block_y_size), -1)
        pred_out_w3 = np.full((block_x_size, block_y_size), -1)
        pred_out_w4 = np.full((block_x_size, block_y_size), -1)
        pred_out_w5 = np.full((block_x_size, block_y_size), -1)
        pred_out_w6 = np.full((block_x_size, block_y_size), -1)
        pred_out_w7 = np.full((block_x_size, block_y_size), -1)
        pred_out_w8 = np.full((block_x_size, block_y_size), -1)
        pred_out_w9 = np.full((block_x_size, block_y_size), -1)
        pred_out_w10 = np.full((block_x_size, block_y_size), -1)
        pred_out_w11 = np.full((block_x_size, block_y_size), -1)
        pred_out_w12 = np.full((block_x_size, block_y_size), -1)
        pred_out_w13 = np.full((block_x_size, block_y_size), -1)
        pred_out_w14 = np.full((block_x_size, block_y_size), -1)
        pred_out_w15 = np.full((block_x_size, block_y_size), -1)

        if (block_x_size > 8) and (block_y_size > 8):
            for x in range(block_x_size):
                for y in range(block_y_size):
                    new            = block_df.loc[(block_df['x'] == x)&(block_df['y'] == y)] 
                    if len(new) > 0:
                        true_out[x, y] = new['ytrue']

                        pred_out_w1[x, y] = new['ypred_w1']
                        pred_out_w2[x, y] = new['ypred_w2']
                        pred_out_w3[x, y] = new['ypred_w3']
                        pred_out_w4[x, y] = new['ypred_w4']
                        pred_out_w5[x, y] = new['ypred_w5']
                        pred_out_w6[x, y] = new['ypred_w6']
                        pred_out_w7[x, y] = new['ypred_w7']
                        pred_out_w8[x, y] = new['ypred_w8']
                        pred_out_w9[x, y] = new['ypred_w8']
                        pred_out_w10[x, y] = new['ypred_w10']
                        pred_out_w11[x, y] = new['ypred_w11']
                        pred_out_w12[x, y] = new['ypred_w12']
                        pred_out_w13[x, y] = new['ypred_w13']
                        pred_out_w14[x, y] = new['ypred_w14']
                        pred_out_w15[x, y] = new['ypred_w15']
                        
        scale = int(spatial_resolutin/10)
        true_agg = aggregate2(true_out, scale)

        pred_agg_w1 = aggregate2(pred_out_w1, scale)
        pred_agg_w2 = aggregate2(pred_out_w2, scale)
        pred_agg_w3 = aggregate2(pred_out_w3, scale)
        pred_agg_w4 = aggregate2(pred_out_w4, scale)
        pred_agg_w5  = aggregate2(pred_out_w5, scale)
        pred_agg_w6  = aggregate2(pred_out_w6, scale)
        pred_agg_w7  = aggregate2(pred_out_w7, scale)
        pred_agg_w8  = aggregate2(pred_out_w8, scale)
        pred_agg_w9  = aggregate2(pred_out_w9, scale)
        pred_agg_w10 = aggregate2(pred_out_w10, scale)
        pred_agg_w11 = aggregate2(pred_out_w11, scale)
        pred_agg_w12 = aggregate2(pred_out_w12, scale)
        pred_agg_w13 = aggregate2(pred_out_w13, scale)
        pred_agg_w14 = aggregate2(pred_out_w14, scale)
        pred_agg_w15 = aggregate2(pred_out_w15, scale)
    
        flat_true_agg = true_agg.flatten()
        flat_true_agg = flat_true_agg[flat_true_agg != -1] 
        b_true.append(flat_true_agg)

        flat_pred_agg_w1 = pred_agg_w1.flatten()
        flat_pred_agg_w1 = flat_pred_agg_w1[flat_pred_agg_w1 != -1]
        b_out_w1.append(flat_pred_agg_w1) 

        flat_pred_agg_w2 = pred_agg_w2.flatten()
        flat_pred_agg_w2 = flat_pred_agg_w2[flat_pred_agg_w2 != -1]
        b_out_w2.append(flat_pred_agg_w2) 

        flat_pred_agg_w3 = pred_agg_w3.flatten()
        flat_pred_agg_w3 = flat_pred_agg_w3[flat_pred_agg_w3 != -1]
        b_out_w3.append(flat_pred_agg_w3) 

        flat_pred_agg_w4 = pred_agg_w4.flatten()
        flat_pred_agg_w4 = flat_pred_agg_w4[flat_pred_agg_w4 != -1]
        b_out_w4.append(flat_pred_agg_w4) 

        flat_pred_agg_w5 = pred_agg_w1.flatten()
        flat_pred_agg_w5 = flat_pred_agg_w5[flat_pred_agg_w5 != -1]
        b_out_w5.append(flat_pred_agg_w5) 

        flat_pred_agg_w6 = pred_agg_w6.flatten()
        flat_pred_agg_w6 = flat_pred_agg_w6[flat_pred_agg_w6 != -1]
        b_out_w6.append(flat_pred_agg_w6) 

        flat_pred_agg_w7 = pred_agg_w7.flatten()
        flat_pred_agg_w7 = flat_pred_agg_w7[flat_pred_agg_w7 != -1]
        b_out_w7.append(flat_pred_agg_w7) 

        flat_pred_agg_w8 = pred_agg_w8.flatten()
        flat_pred_agg_w8 = flat_pred_agg_w8[flat_pred_agg_w8 != -1]
        b_out_w8.append(flat_pred_agg_w8) 

        flat_pred_agg_w9 = pred_agg_w9.flatten()
        flat_pred_agg_w9 = flat_pred_agg_w9[flat_pred_agg_w9 != -1]
        b_out_w9.append(flat_pred_agg_w9) 

        flat_pred_agg_w10 = pred_agg_w10.flatten()
        flat_pred_agg_w10 = flat_pred_agg_w10[flat_pred_agg_w10 != -1]
        b_out_w10.append(flat_pred_agg_w10) 

        flat_pred_agg_w11 = pred_agg_w11.flatten()
        flat_pred_agg_w11 = flat_pred_agg_w11[flat_pred_agg_w11 != -1]
        b_out_w11.append(flat_pred_agg_w11) 

        flat_pred_agg_w12 = pred_agg_w12.flatten()
        flat_pred_agg_w12 = flat_pred_agg_w12[flat_pred_agg_w12 != -1]
        b_out_w12.append(flat_pred_agg_w12) 

        flat_pred_agg_w13 = pred_agg_w13.flatten()
        flat_pred_agg_w13 = flat_pred_agg_w13[flat_pred_agg_w13 != -1]
        b_out_w13.append(flat_pred_agg_w13) 

        flat_pred_agg_w14 = pred_agg_w14.flatten()
        flat_pred_agg_w14 = flat_pred_agg_w14[flat_pred_agg_w14 != -1]
        b_out_w14.append(flat_pred_agg_w14) 

        flat_pred_agg_w15 = pred_agg_w15.flatten()
        flat_pred_agg_w15 = flat_pred_agg_w15[flat_pred_agg_w15 != -1]
        b_out_w15.append(flat_pred_agg_w15) 

        block_id_vector = len(flat_pred_agg_w15) * [this_block_name]
        blocks.append(block_id_vector)


    output['block']    = np.concatenate(blocks) 
    output['ytrue']    = np.concatenate(b_true) 
    output['ypred_w1'] = np.concatenate(b_out_w1)  
    output['ypred_w2'] = np.concatenate(b_out_w2)  
    output['ypred_w3'] = np.concatenate(b_out_w3)  
    output['ypred_w4'] = np.concatenate(b_out_w4)  
    output['ypred_w5'] = np.concatenate(b_out_w5)  
    output['ypred_w6'] = np.concatenate(b_out_w6)  
    output['ypred_w7'] = np.concatenate(b_out_w7)  
    output['ypred_w8'] = np.concatenate(b_out_w8)  
    output['ypred_w9'] = np.concatenate(b_out_w9)  
    output['ypred_w10'] = np.concatenate(b_out_w10)  
    output['ypred_w11'] = np.concatenate(b_out_w11)  
    output['ypred_w12'] = np.concatenate(b_out_w12)  
    output['ypred_w13'] = np.concatenate(b_out_w13)  
    output['ypred_w14'] = np.concatenate(b_out_w14)  
    output['ypred_w15'] = np.concatenate(b_out_w15)  
            
    return output        
         

def aggregate2(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.full((w, h), -1, dtype=np.float32)
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx            
            
def block_perform_plot(df, block_id, aggregation = False, agg_scale = None, fig_save_name=None, save = None): 
    
    this_block_df, ytrue, ypred = block_true_pred_mtx(df, block_id, aggregation, agg_scale)
    
    r2 , mae, rmse, mape, mean_ytrue, mean_ypred = regression_metrics(this_block_df['ytrue'], this_block_df['ypred'])
    
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(2, 2, figsize = (20, 10))
    fig.suptitle(block_id) 
    
    img1 = axs[0, 0].imshow(ytrue)
    axs[0, 0].set_title('Yield Observation')
    #divider = make_axes_locatable(axs[0, 0])
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #cbar1 = fig.colorbar(img1,  cax=cax)
    img1.set_clim(0, 30)

    img2 = axs[0, 1].imshow(ypred)
    axs[0, 1].set_title('Yield Prediction')
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img2, cax=cax)
    img2.set_clim(0, 30)
    axs[0, 1].get_yaxis().set_visible(False)
    
    
    
    axs[1, 0].set_xlim([0, 30])
    axs[1, 0].set_ylim([0, 30])
    axs[1, 0].plot([0, 30], [0, 30],
               '--r', linewidth=2)
    #axs[1, 0].scatter(flatten_ytrue, flatten_ypred, alpha=0.2)
    axs[1, 0].hexbin(this_block_df['ytrue'], this_block_df['ypred'], gridsize=(100,100), extent=[0, 30, 0, 30])
    axs[1, 0].set_xlabel('Measured')
    axs[1, 0].set_ylabel('Predicted')
    axs[1, 0].grid(False)
    axs[1, 0].set_facecolor('white')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+ '\n' + r'$MAPE={:.2f}$').format(r2, mae, rmse, mape)
    axs[1, 0].legend([extra], [scores], loc='upper left')
    #axs[0, 0].set_title('Linear Regression Results on Test Data')
    
    
    box_df = df[['ytrue', 'ypred_w13']]
    s_df   = pd.DataFrame()
    s_df['ytrue'] = [mean_ytrue]
    s_df['ypred_w13'] = [mean_ypred]

    print(s_df)
    sns.boxplot(data=box_df,  orient='v' , ax=axs[1, 1])
    #axs[1, 1].set_xlabel('Yield(p/ha)')
    axs[1, 1].set_ylabel('Yield(p/ha)')
    sns.stripplot(data = s_df, color="r", size=14, ax=axs[1, 1])
    
    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)            
            
            
            
            
        
def triple_dist_plot(train_true, train_pred, valid_true, valid_pred, test_true, test_pred, fig_save_name = None, save = None): 
    
    tr_true_df = pd.DataFrame(train_true, columns = ['True_Lower','True_Upper'])
    tr_pred_df = pd.DataFrame(train_pred, columns = ['Pred_Lower','Pred_Upper'])
    
    plt.rcParams["axes.grid"] = False
    fig, axs = plt.subplots(2, 3, figsize = (21, 12))
    

    #sns.kdeplot(tr_true_df['Lower'], hist = False, kde = True,
    #                 kde_kws = {'shade': True, 'linewidth': 3}, 
    #                  label = 'tr_true Y01', ax = axs[0, 0])
    sns.kdeplot(tr_true_df['True_Lower'], shade = True, legend = True, ax = axs[0, 0], palette="crest")
        
    sns.kdeplot(tr_pred_df['Pred_Lower'], shade = True, legend = True, ax = axs[0, 0], palette="crest")
    axs[0, 0].set_title('Train Lower True and Pred Dist.')
    axs[0, 0].set(xlabel=None)
    

    sns.kdeplot(tr_true_df['True_Upper'], shade = True, legend = True, ax = axs[1, 0], palette="crest")
    
    sns.kdeplot(tr_pred_df['Pred_Upper'], shade = True, legend = True, ax = axs[1, 0], palette="crest")
    
    axs[1, 0].set_title('Train Upper True and Pred Dist.')
    axs[1, 0].set(xlabel=None)
    #===============================================================================
    v_true_df = pd.DataFrame(valid_true, columns = ['True_Lower','True_Upper'])
    v_pred_df = pd.DataFrame(valid_pred, columns = ['Pred_Lower','Pred_Upper'])
    
    sns.kdeplot(v_true_df['True_Lower'], shade = True, legend = True, ax = axs[0, 1], palette="crest")
    
    sns.kdeplot(v_pred_df['Pred_Lower'], shade = True, legend = True, ax = axs[0, 1], palette="crest")
    axs[0, 1].set_title('Valid Lower True and Pred Dist.')
    axs[0, 1].set(xlabel=None)
    axs[0, 1].set(ylabel=None)

    sns.kdeplot(v_true_df['True_Upper'], shade = True, legend = True, ax = axs[1,1], palette="crest")
    
    sns.kdeplot(v_pred_df['Pred_Upper'], shade = True, legend = True, ax = axs[1,1], palette="crest")
    
    axs[1,1].set_title('Valid Upper True and Pred Dist.')
    axs[1,1].set(xlabel=None)
    axs[1,1].set(ylabel=None)
    #============================================================================
    te_true_df = pd.DataFrame(test_true, columns = ['True_Lower','True_Upper'])
    te_pred_df = pd.DataFrame(test_pred, columns = ['Pred_Lower','Pred_Upper'])
    
    
    sns.kdeplot(te_true_df['True_Lower'], shade = True, legend = True, ax = axs[0, 2], palette="crest")
    
    sns.kdeplot(te_pred_df['Pred_Lower'], shade = True, legend = True, ax = axs[0, 2], palette="crest")
    axs[0, 2].set_title('Test Lower True and Pred Dist.')
    axs[0, 2].set(xlabel=None)
    axs[0, 2].set(ylabel=None)
    
    
    sns.kdeplot(te_true_df['True_Upper'], shade = True, legend = True, ax = axs[1,2], palette="crest")
    
    sns.kdeplot(te_pred_df['Pred_Upper'], shade = True, legend = True, ax = axs[1,2], palette="crest")
    
    axs[1,2].set_title('Test Upper True and Pred Dist.')
    axs[1,2].set(xlabel=None)
    axs[1,2].set(ylabel=None)
    
    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)           
        


        
def test_cultivars_plots(test_df, scatter_mode = None, gridsize = 0, fig_save_name = None, save = None): 
    
    
    plt.rcParams["axes.grid"] = False
    group_list = test_df.groupby(by = ['cultivar'])
    
    rowlength = int(group_list.ngroups/3)                  
    fig, axs = plt.subplots(figsize=(21, int(group_list.ngroups/3)*6), 
                        nrows=rowlength, ncols=3,     
                        gridspec_kw=dict(hspace=0.4)) 

    targets = zip(group_list.groups.keys(), axs.flatten())
    
    
    for i, (key, ax) in enumerate(targets):
        
        this_group = group_list.get_group(key)
        ns = this_group.shape[0]
        ytrue = this_group['ytrue']
        ypred = this_group['ypred']
        test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(ytrue, ypred)
        
        ax.set_xlim([0, 30])
        ax.set_ylim([0, 30])
        ax.plot([0, 30], [0, 30],
                   '--r', linewidth=2)
        
        if scatter_mode == 'h':
            ax.hexbin(ytrue, ypred, gridsize=(gridsize,gridsize), extent=[0, 30, 0, 30])
        elif scatter_mode == 'c':
            ax.scatter(ytrue, ypred, alpha=0.2)
        ax.grid(False)
        ax.set_facecolor('white')
        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                                edgecolor='none', linewidth=0)
        scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+ '\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
        ax.legend([extra], [scores], loc='upper left', fontsize=12)
        ax.set_title('Cultivar=%d'%key, size=14)

    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)
        
        
def cultivar_aggimg_plots(ytrue_list, ypred_list, name_list, fig_save_name, gridsize = 0, scatter_mode = None, save = None): 
    plt.rcParams["axes.grid"] = False
    
    
    fig, axs = plt.subplots(5, 3, figsize = (21, 30))
    t = 0
    for i in range(5):
        for j in range(3): 
            

            Test_True = ytrue_list[t]
            Test_Pred = ypred_list[t] 
            test_rmse, test_r2, test_mae, test_mape, _, _ = regression_metrics(Test_True, Test_Pred)


            axs[i,j].set_xlim([Test_True.min(), Test_True.max()])
            axs[i,j].set_ylim([Test_True.min(), Test_True.max()])
            axs[i,j].plot([Test_True.min(), Test_True.max()], [Test_True.min(), Test_True.max()],
                       '--r', linewidth=2)
            #
            if scatter_mode == 'h':
                axs[i,j].hexbin(Test_True, Test_Pred, gridsize=(gridsize,gridsize))
            elif scatter_mode == 'c':
                axs[i,j].scatter(Test_True, Test_Pred, alpha=0.2)
            #axs[i,j].set_xlabel('Measured')
            #axs[2].set_ylabel('Predicted')
            axs[i,j].grid(False)
            axs[i,j].set_facecolor('white')
            extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                                    edgecolor='none', linewidth=0)
            scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+ '\n' + r'$MAPE={:.2f}$').format(test_r2, test_rmse, test_mae, test_mape)
            axs[i,j].legend([extra], [scores], loc='upper left', fontsize=12)
            axs[i,j].set_title(name_list[t], size=14)
            
            
            t+=1
    
    if save is True: 
        plt.savefig(fig_save_name, dpi = 300)
        
def yield_plot(src, fig_save_name, title, save_fig = True):
    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    
    img = ax.imshow(src[0,:,:]) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(img, cax=cax)
    ax.set_title(title)
    if save_fig is True: 
        plt.savefig(fig_save_name, dpi = 300)
        
def cultivar_plots(test_df):
    plt.rcParams["axes.grid"] = False
    group_list = test_df.groupby(by = ['Cultivar'])
    
    n_smaples, r2 = [],[]
    for state, frame in group_list:  
        ns = frame.shape[0]
        n_smaples.append(ns)
        
        ytrue = frame['ytrue']
        ypred = frame['ypred']
        r_square   = r2_score(ytrue, ypred)
        r2.append(r_square) 
        print(f"{state}: Number of sampels = {ns} and R^2 = {r_square}")
 
    sns.lmplot('ytrue', 'ypred', data=test_df, hue='Cultivar', col='Cultivar', col_wrap=4, height=4, legend=False)   
    
def scenario_eplit_dist_plot(train, val, test, fig_save_name, save_fig = True):
    
    
    fig, axs = plt.subplots(1, 3, figsize = (21, 6), sharey=True)
    #plt.rcParams["figure.figsize"] = [16, 15]
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    ax1 = sns.kdeplot(train['patch_mean'], ax = axs[0], shade ='fill', label = 'train', legend = True)
    ax1.set(ylabel='Density')
    ax1.legend()

    ax2 = sns.kdeplot(val['patch_mean'],  ax = axs[1], shade ='fill', label = 'validation')
    ax2.legend()
    ax3 = sns.kdeplot(test['patch_mean'],  ax = axs[2], shade ='fill', label = 'test')
    ax3.legend()
    None
    
    if save_fig is True: 
        plt.savefig(fig_save_name, dpi = 300)



def yield_map_mosaic(yield_map_dir, year = None):

    if year == '2016':
        format = "*.asc"
    else:
        format = "*.tif"

    list_yield_tif = sorted(glob(yield_map_dir + format))

    # creating empty list 
    src_files_to_mosiac = []

    for map in list_yield_tif:
        src = rasterio.open(map)
        src_files_to_mosiac.append(src)
        mos, out_trans = merge(src_files_to_mosiac)

    mos[mos<0] = -1

    return mos, out_trans


def yield_plot(src, fig_save_name, title, save_fig = True):
    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    
    img = ax.imshow(src[0,:,:]) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(img, cax=cax)
    ax.set_title(title)
    if save_fig is True: 
        plt.savefig(fig_save_name, dpi = 300)


def double_imglable_plot(root_dir, block_name = None):
    
    image_dir  = os.path.join(root_dir, 'imgs/')
    label_dir  = os.path.join(root_dir, 'labels/')

    img_name   = block_name + '_img.npy'
    label_name = block_name + '_label.npy'
  

    img   = np.load(image_dir + img_name, allow_pickle = True)
    img   = np.swapaxes(img, -1, 0)
    img   = img[:, :, :, 0]
    label = np.load(label_dir + label_name, allow_pickle = True)
    label = label[0,:,:,0]
    print(f"{img.shape}|{label.shape}")

    plt.rcParams["axes.grid"] = False
    fig, ((axrgb, lrgb), (axhist, lhist)) = plt.subplots(2, 2, figsize=(2*8, 16))
    show(img, ax=axrgb)
    show_hist(img, bins=60, histtype='stepfilled',
            lw=0.0, stacked=False, alpha=0.3, ax=axhist)
    axhist.get_legend().remove()
    show(label, cmap='viridis',  ax=lrgb)
    show_hist(label, bins = 60, histtype='stepfilled',
            lw=0.0, stacked=False, alpha=0.3, ax=lhist)
    lhist.get_legend().remove()
    #plt.title(block_name)
    #plt.savefig('./' + block_name + '.png', dpi = 300)




def satellite_timeseries_plot(root_dir, block_name = None):

    image_dir  = os.path.join(root_dir, 'imgs/')
    img_name   = block_name + '_img.npy'

    img = np.load(image_dir + img_name, allow_pickle=True)


    plt.rcParams["axes.grid"] = False
    for i in range(img.shape[0]):
        
        image = np.swapaxes(img, -1, 0)
        image = image[:,:,:, i]
        fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(2*8, 8))
        show(image, ax=axrgb)
        show_hist(image, bins=50, histtype='stepfilled',
                lw=0.0, stacked=False, alpha=0.3, ax=axhist)

def block_results_barplot(blocks_csv):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (24, 16))
    #plt.rcParams["figure.figsize"] = [16, 15]
    #plt.rcParams["axes.grid"] = False
    #plt.rcParams["figure.autolayout"] = True
    sns.set_style("whitegrid", {'axes.grid' : False})
    colors = sns.color_palette('tab20', 15)
    #plt.subplots_adjust(hspace = 0.01)

    bar1 = sns.barplot(x="block",  y="R2", data=blocks_csv, hue = 'cultivar', palette=colors,  dodge=False, ax = ax1)
    #bar1.legend_.remove()
    plt.setp(bar1.get_legend().get_texts(), fontsize='10') 
    bar1.set_ylim([0, 1])
    ax1.axes.get_xaxis().set_visible(False)

    bar2 = sns.barplot(x="block",  y="MAE", data=blocks_csv, hue = 'cultivar', palette=colors, dodge=False, ax = ax2)
    bar2.legend_.remove()
    bar2.set_ylim([0, blocks_csv['MAE'].max()])
    ax2.axes.get_xaxis().set_visible(False)

    bar3 = sns.barplot(x="block",  y="RMSE", data=blocks_csv, hue = 'cultivar', palette=colors, dodge=False, ax = ax3)
    bar3.legend_.remove()
    bar3.set_ylim([0, blocks_csv['RMSE'].max()])
    ax3.axes.get_xaxis().set_visible(False)

    bar4 = sns.barplot(x="block",  y="MAPE", data=blocks_csv,  hue = 'cultivar', palette=colors, dodge=False, ax = ax4)
    bar4.legend_.remove()
    bar4.set_ylim([0, 0.6])

    plt.xticks(rotation=90, fontsize = 14)
    None    
    
def cultivar_results_barplot(cultivar_csv):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (24, 12))
    #plt.rcParams["axes.grid"] = False
    sns.set_style("whitegrid", {'axes.grid' : False})
    #plt.rcParams["figure.autolayout"] = True
    colors = sns.color_palette('tab20', 15)

    bar1 = sns.barplot(x="cultivar",  y="R2", data=cultivar_csv,  dodge=False, ax = ax1) 
    bar1.set_ylim([0, 1])
    ax1.axes.get_xaxis().set_visible(False)

    bar2 = sns.barplot(x="cultivar",  y="MAE", data=cultivar_csv,  dodge=False, ax = ax2)
    bar2.set_ylim([0, cultivar_csv['MAE'].max()])
    ax2.axes.get_xaxis().set_visible(False)

    bar3 = sns.barplot(x="cultivar",  y="RMSE", data=cultivar_csv, dodge=False, ax = ax3)
    bar3.set_ylim([0, cultivar_csv['RMSE'].max()])


    bar4 = sns.barplot(x="cultivar",  y="MAPE", data=cultivar_csv, dodge=False, ax = ax4)
    bar4.set_ylim([0, 0.6])

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    None
    


def save_loss_df(loss_stat, loss_df_name, loss_fig_name):

    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    df.to_csv(loss_df_name) 
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi = 300)

  
#==============================================================================================#
#====================================      Info   ==============================#
#==============================================================================================#    
cultivars_ = {'1' : ['ALICANTE_BOUSCHET'],
              '2' : ['CABERNET_SAUVIGNON'],
              '3' : ['CHARDONNAY'],
              '4' : ['DORNFELDER'],
              '5' : ['LAMBRUSCO'],
              '6' : ['MALBEC'],
              '7' : ['MALVASIA_BIANCA'],
              '8' : ['MERLOT'],
              '9' : ['MUSCAT_CANELLI'],
              '10' : ['MUSCAT_OF_ALEXANDRIA'],
              '11' : ['PINOT_GRIS'],
              '12' : ['PINOT_NOIR'],
              '13' : ['RIESLING'],
              '14' : ['SYMPHONY'],
              '15' : ['SYRAH']}   


Categ_ = {'LIV_003':['MALVASIA_BIANCA', '7', '12', '7', '1991', '4WIREWO', '1'], 
          'LIV_004':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_005':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_006':['MALVASIA_BIANCA', '7', '12', '7', '1993', '4WIREWO', '1'], 
          'LIV_007':['SYMPHONY', '14', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_008':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_009':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_010':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_011':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_012':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_013':['SYRAH', '15', '12', '7', '1995', 'LIVDC', '3'], 
          'LIV_014':['RIESLING', '13', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_015':['MALVASIA_BIANCA', '7', '12', '7', '1985', 'QUAD', '6'], 
          'LIV_016':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_017':['CABERNET_SAUVIGNON', '2', '11', '5', '1996', 'LIVDC', '3'], 
          'LIV_018':['CHARDONNAY', '3','10', '4', '1995', 'LIVDC', '3'], 
          'LIV_019':['RIESLING','13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_021':['PINOT_GRIS', '11', '10', '4', '2015', 'STACKEDT', '5'], 
          'LIV_022':['PINOT_NOIR', '12', '10', '5', '1997', 'SPLIT', '2'],
          'LIV_025':['CABERNET_SAUVIGNON', '2', '9', '9', '1996', '4WIREWO', '1'], 
          'LIV_026':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_027':['MERLOT', '8', '10', '4', '1994', 'LIVDC', '3'], 
          'LIV_028':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_032':['CABERNET_SAUVIGNON', '2', '10', '5', '1996', 'LIVDC', '3'], 
          'LIV_038':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_050':['SYMPHONY', '14', '10', '7', '1997', 'LIVDC', '3'], 
          'LIV_058':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_061':['PINOT_GRIS', '11', '10', '6', '2002', '4WIREWO', '1'], 
          'LIV_062':['SYRAH', '15', '10', '8', '1995', 'LIVDC', '3'], 
          'LIV_063':['PINOT_GRIS', '11', '10', '4', '2014', 'STACKEDT', '5'], 
          'LIV_064':['CABERNET_SAUVIGNON', '2', '8', '9', '1997', '4WIREWO', '1'], 
          'LIV_066':['PINOT_GRIS', '11', '10', '4', '2012', 'SPLIT', '2'], 
          'LIV_068':['MERLOT', '8', '10', '8', '1994', '4WIREWM', '4'], 
          'LIV_070':['CABERNET_SAUVIGNON', '2', '8', '9', '1996', '4WIREWO', '1'], 
          'LIV_073':['PINOT_NOIR', '12', '12', '7', '2003', 'SPLIT', '2'], 
          'LIV_076':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_077':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_089':['CHARDONNAY', '3', '9', '6', '2010', 'VERTICAL', '7'], 
          'LIV_090':['CHARDONNAY', '3', '11', '6', '1993', 'VERTICAL', '7'], 
          'LIV_094':['RIESLING', '13', '11', '5', '2014', 'STACKEDT', '5'], 
          'LIV_102':['MUSCAT_OF_ALEXANDRIA', '10', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_103':['CABERNET_SAUVIGNON', '2', '9', '6', '2012', 'HIGHWIRE', '8'], 
          'LIV_105':['MUSCAT_OF_ALEXANDRIA', '10', '10', '4', '2011', 'LIVDC', '3'], 
          'LIV_107':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_111':['CHARDONNAY', '3', '10', '4', '1995', 'LIVDC', '3'], 
          'LIV_114':['MALVASIA_BIANCA', '7', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_123':['MALBEC', '6', '11', '5', '2010', 'SPLIT', '2'], 
          'LIV_125':['DORNFELDER', '4', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_126':['CABERNET_SAUVIGNON', '2', '9', '6', '2010', 'TALLVERTICAL', '9'], 
          'LIV_128':['RIESLING', '13', '11', '5', '2013', 'SPLIT', '2'], 
          'LIV_135':['CHARDONNAY', '3', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_136':['ALICANTE_BOUSCHET', '1', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_163':['CHARDONNAY', '3', '12', '4', '1995', 'LIVDC', '3'], 
          'LIV_172':['MUSCAT_CANELLI', '9', '11', '5', '2011', 'SPLIT', '2'], 
          'LIV_175':['CHARDONNAY', '3', '11', '4', '1994', 'LIVDC', '3'], 
          'LIV_176':['CHARDONNAY', '3', '11', '7', '1994', '4WIREWO', '1'], 
          'LIV_177':['CHARDONNAY', '3', '10', '6', '1993', '4WIREWO', '1'], 
          'LIV_178':['RIESLING', '13', '11', '5', '2012', 'SPLIT', '2'], 
          'LIV_181':['SYMPHONY', '14', '10', '6', '1997', 'LIVDC', '3'], 
          'LIV_182':['LAMBRUSCO', '5', '11', '5', '2013', 'QUAD', '6'], 
          'LIV_186':['RIESLING', '13', '11', '7', '2012', 'SPLIT', '2'], 
          'LIV_193':['MERLOT', '8', '11', '5', '2010', 'SPLIT', '2']}


blocks_size = {'LIV_003_2016': [392, 1560],
 'LIV_003_2017': [394, 1560],
 'LIV_003_2018': [396, 1563],
 'LIV_003_2019': [392, 1563],
 'LIV_004_2016': [149, 779],
 'LIV_004_2017': [151, 778],
 'LIV_004_2018': [152, 782],
 'LIV_004_2019': [154, 780],
 'LIV_005_2016': [792, 998],
 'LIV_005_2017': [793, 998],
 'LIV_005_2018': [796, 1002],
 'LIV_006_2016': [436, 790],
 'LIV_006_2017': [437, 795],
 'LIV_006_2018': [442, 794],
 'LIV_006_2019': [439, 759],
 'LIV_007_2016': [788, 374],
 'LIV_007_2017': [789, 375],
 'LIV_007_2018': [792, 377],
 'LIV_007_2019': [791, 378],
 'LIV_008_2016': [672, 790],
 'LIV_008_2017': [672, 791],
 'LIV_008_2018': [676, 791],
 'LIV_009_2016': [825, 760],
 'LIV_009_2017': [827, 759],
 'LIV_009_2018': [826, 766],
 'LIV_009_2019': [827, 765],
 'LIV_010_2016': [326, 866],
 'LIV_010_2017': [325, 866],
 'LIV_010_2018': [330, 871],
 'LIV_011_2016': [505, 973],
 'LIV_011_2017': [506, 974],
 'LIV_011_2018': [509, 974],
 'LIV_011_2019': [509, 965],
 'LIV_012_2016': [704, 708],
 'LIV_012_2017': [702, 707],
 'LIV_012_2018': [703, 707],
 'LIV_012_2019': [706, 705],
 'LIV_013_2016': [323, 693],
 'LIV_013_2017': [323, 663],
 'LIV_013_2018': [325, 695],
 'LIV_013_2019': [324, 628],
 'LIV_014_2016': [758, 764],
 'LIV_014_2017': [759, 765],
 'LIV_014_2018': [759, 768],
 'LIV_014_2019': [759, 769],
 'LIV_016_2016': [842, 918],
 'LIV_016_2017': [827, 919],
 'LIV_016_2018': [839, 920],
 'LIV_016_2019': [843, 918],
 'LIV_017_2016': [377, 201],
 'LIV_017_2017': [378, 202],
 'LIV_017_2018': [376, 205],
 'LIV_018_2016': [666, 1582],
 'LIV_018_2017': [667, 1582],
 'LIV_018_2018': [671, 1585],
 'LIV_018_2019': [632, 1586],
 'LIV_019_2016': [355, 388],
 'LIV_019_2017': [355, 392],
 'LIV_019_2018': [322, 398],
 'LIV_019_2019': [312, 385],
 'LIV_021_2017': [1280, 1783],
 'LIV_021_2018': [1597, 1790],
 'LIV_021_2019': [1599, 1785],
 'LIV_022_2016': [388, 393],
 'LIV_022_2017': [389, 393],
 'LIV_022_2018': [393, 394],
 'LIV_025_2016': [892, 790],
 'LIV_025_2017': [820, 790],
 'LIV_025_2018': [863, 795],
 'LIV_025_2019': [895, 794],
 'LIV_026_2016': [584, 379],
 'LIV_026_2017': [590, 381],
 'LIV_027_2016': [127, 381],
 'LIV_027_2017': [130, 381],
 'LIV_028_2016': [373, 784],
 'LIV_028_2017': [383, 785],
 'LIV_028_2018': [384, 787],
 'LIV_028_2019': [333, 788],
 'LIV_032_2016': [350, 727],
 'LIV_032_2017': [393, 733],
 'LIV_032_2018': [399, 729],
 'LIV_038_2016': [405, 797],
 'LIV_038_2017': [405, 798],
 'LIV_038_2018': [409, 799],
 'LIV_050_2016': [423, 800],
 'LIV_050_2017': [424, 801],
 'LIV_050_2018': [427, 801],
 'LIV_050_2019': [427, 800],
 'LIV_058_2016': [394, 797],
 'LIV_058_2017': [395, 797],
 'LIV_058_2018': [399, 801],
 'LIV_061_2016': [677, 506],
 'LIV_061_2017': [678, 503],
 'LIV_061_2018': [684, 510],
 'LIV_061_2019': [646, 509],
 'LIV_062_2016': [733, 806],
 'LIV_062_2017': [733, 805],
 'LIV_062_2018': [734, 806],
 'LIV_063_2016': [1252, 801],
 'LIV_063_2017': [1252, 802],
 'LIV_063_2018': [1167, 818],
 'LIV_063_2019': [1265, 817],
 'LIV_066_2016': [366, 791],
 'LIV_066_2017': [367, 795],
 'LIV_066_2019': [373, 797],
 'LIV_068_2016': [366, 791],
 'LIV_068_2017': [189, 781],
 'LIV_070_2016': [189, 797],
 'LIV_070_2017': [594, 782],
 'LIV_070_2018': [598, 788],
 'LIV_073_2016': [783, 601],
 'LIV_073_2017': [786, 601],
 'LIV_073_2018': [791, 605],
 'LIV_076_2016': [393, 1026],
 'LIV_076_2017': [396, 1031],
 'LIV_076_2018': [397, 1032],
 'LIV_076_2019': [396, 1031],
 'LIV_077_2016': [907, 458],
 'LIV_077_2017': [610, 457],
 'LIV_077_2018': [914, 463],
 'LIV_077_2019': [911, 462],
 'LIV_089_2016': [234, 325],
 'LIV_089_2017': [235, 327],
 'LIV_089_2018': [236, 329],
 'LIV_089_2019': [236, 330],
 'LIV_090_2016': [271, 790],
 'LIV_090_2017': [272, 791],
 'LIV_090_2018': [276, 791],
 'LIV_090_2019': [274, 726],
 'LIV_094_2016': [146, 179],
 'LIV_094_2017': [379, 252],
 'LIV_094_2018': [381, 256],
 'LIV_094_2019': [382, 246],
 'LIV_102_2016': [793, 536],
 'LIV_102_2017': [739, 535],
 'LIV_102_2018': [794, 537],
 'LIV_102_2019': [787, 526],
 'LIV_103_2016': [402, 1164],
 'LIV_103_2017': [400, 1163],
 'LIV_103_2018': [408, 1168],
 'LIV_103_2019': [408, 1168],
 'LIV_105_2016': [666, 786],
 'LIV_105_2017': [644, 784],
 'LIV_105_2018': [671, 791],
 'LIV_105_2019': [668, 789],
 'LIV_107_2016': [757, 402],
 'LIV_107_2017': [790, 403],
 'LIV_107_2018': [794, 407],
 'LIV_107_2019': [727, 404],
 'LIV_111_2016': [317, 498],
 'LIV_111_2017': [366, 497],
 'LIV_111_2018': [331, 454],
 'LIV_111_2019': [319, 446],
 'LIV_114_2016': [333, 333],
 'LIV_114_2017': [333, 335],
 'LIV_114_2018': [339, 336],
 'LIV_114_2019': [338, 338],
 'LIV_123_2016': [391, 759],
 'LIV_123_2017': [392, 761],
 'LIV_123_2018': [395, 764],
 'LIV_123_2019': [394, 765],
 'LIV_125_2016': [425, 387],
 'LIV_125_2017': [426, 386],
 'LIV_125_2018': [361, 388],
 'LIV_125_2019': [430, 391],
 'LIV_128_2016': [378, 790],
 'LIV_128_2017': [379, 790],
 'LIV_128_2018': [384, 795],
 'LIV_128_2019': [350, 790],
 'LIV_135_2016': [814, 718],
 'LIV_135_2017': [747, 716],
 'LIV_135_2018': [821, 722],
 'LIV_135_2019': [821, 723],
 'LIV_136_2016': [740, 863],
 'LIV_136_2017': [815, 864],
 'LIV_136_2018': [818, 869],
 'LIV_136_2019': [816, 868],
 'LIV_172_2016': [393, 740],
 'LIV_172_2017': [394, 740],
 'LIV_172_2018': [400, 743],
 'LIV_172_2019': [398, 743],
 'LIV_176_2016': [375, 177],
 'LIV_176_2017': [375, 177],
 'LIV_176_2018': [379, 172],
 'LIV_176_2019': [379, 174],
 'LIV_177_2016': [832, 374],
 'LIV_177_2017': [822, 320],
 'LIV_177_2018': [839, 374],
 'LIV_177_2019': [839, 373],
 'LIV_178_2016': [399, 781],
 'LIV_178_2017': [403, 780],
 'LIV_178_2018': [405, 784],
 'LIV_178_2019': [338, 781],
 'LIV_181_2016': [384, 188],
 'LIV_181_2017': [352, 186],
 'LIV_181_2018': [391, 191],
 'LIV_181_2019': [390, 192],
 'LIV_182_2016': [384, 391],
 'LIV_182_2017': [385, 392],
 'LIV_182_2018': [385, 396],
 'LIV_182_2019': [383, 395],
 'LIV_186_2016': [504, 787],
 'LIV_186_2017': [542, 784],
 'LIV_186_2018': [548, 789],
 'LIV_186_2019': [547, 791],
 'LIV_193_2016': [223, 324],
 'LIV_193_2017': [227, 326],
 'LIV_193_2018': [159, 295],
 'LIV_193_2019': [107, 288], }

blocks = ['LIV_003_2016','LIV_003_2017','LIV_003_2018','LIV_003_2019',
          'LIV_004_2016','LIV_004_2017','LIV_004_2018','LIV_004_2019',
          'LIV_005_2016','LIV_005_2017','LIV_005_2018',
          'LIV_006_2016','LIV_006_2017','LIV_006_2018','LIV_006_2019',
          'LIV_007_2016','LIV_007_2017','LIV_007_2018','LIV_007_2019',
          'LIV_008_2016','LIV_008_2017','LIV_008_2018', 
          'LIV_009_2016','LIV_009_2017','LIV_009_2018','LIV_009_2019', 
          'LIV_010_2016','LIV_010_2017','LIV_010_2018',
          'LIV_011_2016','LIV_011_2017','LIV_011_2018','LIV_011_2019',
          'LIV_012_2016','LIV_012_2017','LIV_012_2018','LIV_012_2019',
          'LIV_013_2016','LIV_013_2017','LIV_013_2018','LIV_013_2019',
          'LIV_014_2016','LIV_014_2017','LIV_014_2018','LIV_014_2019',
          'LIV_016_2016','LIV_016_2017','LIV_016_2018','LIV_016_2019',
          'LIV_017_2016','LIV_017_2017','LIV_017_2018', 
          'LIV_018_2016','LIV_018_2017','LIV_018_2018','LIV_018_2019', 
          'LIV_019_2016','LIV_019_2017','LIV_019_2018','LIV_019_2019', 
          'LIV_021_2017','LIV_021_2018','LIV_021_2019',
          'LIV_022_2016','LIV_022_2017','LIV_022_2018',
          'LIV_025_2016','LIV_025_2017','LIV_025_2018','LIV_025_2019', 
          'LIV_026_2016','LIV_026_2017',
          'LIV_027_2016','LIV_027_2017',
          'LIV_028_2016','LIV_028_2017','LIV_028_2018','LIV_028_2019', 
          'LIV_032_2016','LIV_032_2017','LIV_032_2018', 
          'LIV_038_2016','LIV_038_2017','LIV_038_2018', 
          'LIV_050_2016','LIV_050_2017','LIV_050_2018','LIV_050_2019',
          'LIV_058_2016','LIV_058_2017','LIV_058_2018', 
          'LIV_061_2016','LIV_061_2017','LIV_061_2018','LIV_061_2019', 
          'LIV_062_2016','LIV_062_2017','LIV_062_2018', 
          'LIV_063_2016','LIV_063_2017','LIV_063_2018','LIV_063_2019', 
          'LIV_064_2016','LIV_064_2017','LIV_064_2018',
          'LIV_066_2016','LIV_066_2017','LIV_066_2019', 
          'LIV_068_2016','LIV_068_2017', 
          'LIV_070_2016','LIV_070_2017','LIV_070_2018', 
          'LIV_073_2016','LIV_073_2017','LIV_073_2018', 
          'LIV_076_2016','LIV_076_2017','LIV_076_2018','LIV_076_2019', 
          'LIV_077_2016','LIV_077_2017','LIV_077_2018','LIV_077_2019', 
          'LIV_089_2016','LIV_089_2017','LIV_089_2018','LIV_089_2019', 
          'LIV_090_2016','LIV_090_2017','LIV_090_2018','LIV_090_2019', 
          'LIV_094_2016','LIV_094_2017','LIV_094_2018','LIV_094_2019', 
          'LIV_102_2016','LIV_102_2017','LIV_102_2018','LIV_102_2019', 
          'LIV_103_2016','LIV_103_2017','LIV_103_2018','LIV_103_2019', 
          'LIV_105_2016','LIV_105_2017','LIV_105_2018','LIV_105_2019', 
          'LIV_107_2016','LIV_107_2017','LIV_107_2018','LIV_107_2019', 
          'LIV_111_2016','LIV_111_2017','LIV_111_2018','LIV_111_2019', 
          'LIV_114_2016','LIV_114_2017','LIV_114_2018','LIV_114_2019', 
          'LIV_123_2016','LIV_123_2017','LIV_123_2018','LIV_123_2019', 
          'LIV_125_2016','LIV_125_2017','LIV_125_2018','LIV_125_2019', 
          'LIV_128_2016','LIV_128_2017','LIV_128_2018','LIV_128_2019', 
          'LIV_135_2016','LIV_135_2017','LIV_135_2018','LIV_135_2019', 
          'LIV_136_2016','LIV_136_2017','LIV_136_2018','LIV_136_2019', 
          'LIV_163_2016','LIV_163_2017','LIV_163_2018', 
          'LIV_172_2016','LIV_172_2017','LIV_172_2018','LIV_172_2019', 
          'LIV_175_2016','LIV_175_2017','LIV_175_2018','LIV_175_2019', 
          'LIV_176_2016','LIV_176_2017','LIV_176_2018','LIV_176_2019', 
          'LIV_177_2016','LIV_177_2017','LIV_177_2018','LIV_177_2019', 
          'LIV_178_2016','LIV_178_2017','LIV_178_2018','LIV_178_2019', 
          'LIV_181_2016','LIV_181_2017','LIV_181_2018','LIV_181_2019', 
          'LIV_182_2016','LIV_182_2017','LIV_182_2018','LIV_182_2019', 
          'LIV_186_2016','LIV_186_2017','LIV_186_2018','LIV_186_2019', 
          'LIV_193_2016','LIV_193_2017','LIV_193_2018','LIV_193_2019']    


