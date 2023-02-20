import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from src.configs import blocks_information_dict



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




class EarlyStopping():
    def __init__(self, tolerance=30, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, status):

        if status is True:
            self.counter = 0
        elif status is False: 
            self.counter +=1

        print(f"count: {self.counter}")
        if self.counter >= self.tolerance:  
                self.early_stop = True
