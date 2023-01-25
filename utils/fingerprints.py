from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import numpy as np


# calculate ECFP4 (defaut) fingerprints using RDKit

FP_SIZE = 2048  #bit string size
RADIUS = 2 #diameter 4
Feat = False #used when you consider pharmacophoric features


def calc_fp(SMILES, fp_size, radius):
    """
    calcs morgan fingerprints as a numpy array.
    """
    mol = Chem.MolFromSmiles(SMILES, sanitize=True)
    mol.UpdatePropertyCache(False)
    
    Chem.GetSSSR(mol)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=FP_SIZE, useFeatures=Feat)
    a = np.zeros((0,), dtype=np.float32)
    a = Chem.DataStructs.ConvertToNumpyArray(fp, a)
    return np.asarray(fp)
 

def assing_fp(smiles,FP_SIZE,RADIUS):
    canon_smiles=[]
    for smile in smiles:
        #print(smile)
        try:
            cs= Chem.CanonSmiles(smile)
            canon_smiles.append(cs)
        except:
            #canon_smiles.append(smile)
            print(f"not valid smiles {smile}")
    #print(canon_smiles)
    #Make sure that the column where the smiles are is named SMILES
    descs = [calc_fp(smi, FP_SIZE, RADIUS) for smi in canon_smiles]
    descs = np.asarray(descs, dtype=np.float32)
    return descs
