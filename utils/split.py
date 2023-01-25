import pandas as pd
from chemprop.data import utils as chem_utils
from random import randint
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
from utils.utils import GLOBALS 

class DataSplitter():
    def __init__(self) -> None:
        pass
    
    Scaffold_split = GLOBALS.SCAFFOLD_SPLIT
    Random_split = GLOBALS.RANDOM_SPLIT
    
    split_type: int = Random_split or Scaffold_split
    assert split_type == Random_split or split_type == Scaffold_split, "Invalid split type"

    def get_partition_as_df(self,partition:chem_utils.split_data):
        partition_dict = {'ID': partition.compound_names(), 'SMILES': partition.smiles()}
        df = pd.DataFrame(partition_dict)
        return df

    def random_split(self,dataset:str, sizes:tuple[float,float,float]=(.8,.1,.1) ,seed_val:int=42) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        # get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
        #df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'ID': str})
        #cols = list(df.columns)
        #target_names = cols[2:]

        assert sum(sizes) == 1, "Sum of sizes should be 1"
        mol_dataset = chem_utils.get_data(dataset, use_compound_names=True)
        train, valid, test = chem_utils.split_data(mol_dataset, sizes=sizes, seed=seed_val)
        train_df = self.get_partition_as_df(train)
        train_df = train_df[['ID']]
        valid_df = self.get_partition_as_df(valid)
        valid_df = valid_df[['ID']]
        test_df = self.get_partition_as_df(test)
        test_df = test_df[['ID']]
        return train_df, valid_df, test_df

    def scaffold_split(self,dataset:str, sizes:tuple[float,float,float]=(.8,.1,.1), seed:int=42) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        # get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
        # df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'ID': str})
        # cols = list(df.columns)
        # target_names = cols[2:]

        assert sum(sizes) == 1, "Sum of sizes should be 1"
        mol_dataset = chem_utils.get_data(dataset, use_compound_names=True)
        train, valid, test = chem_utils.split_data(mol_dataset, split_type="scaffold_balanced", sizes=sizes, seed=seed)
        train_df = self.get_partition_as_df(train)
        train_df = train_df[['ID']]
        valid_df = self.get_partition_as_df(valid)
        valid_df = valid_df[['ID']]
        test_df = self.get_partition_as_df(test)
        test_df = test_df[['ID']]
        return train_df, valid_df, test_df

    def split_data(self,dataset:str,n_folds:int=5 ,sizes:tuple[float,float,float]=(.8,.1,.1), seed:int=42, split_type=split_type) -> list[tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]]:
        """
        Generates splits for cross validation one at a time to save memory
        """
        for i in range(n_folds):
            if split_type == self.Random_split:
                yield self.random_split(dataset, sizes, seed),i, seed
            elif split_type == self.Scaffold_split:
                yield self.scaffold_split(dataset, sizes, seed),i, seed
    
    def merge_data(self,dataset:str, train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        
        df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'ID': str})
        
        train_df = pd.merge(df, train_df, on='ID')
        valid_df = pd.merge(df, valid_df, on='ID')
        test_df = pd.merge(df, test_df, on='ID')
        
        return train_df, valid_df, test_df


