import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import pandas as pd
from typing import Any, Optional, List


current_dir = os.path.dirname(__file__)
periodic_table_csv = os.path.join(current_dir, 'periodic_table_v2.csv')

class PeriodicTable():
    """Utility class to provide further element type information for crystal graph node embeddings."""
    
    def __init__(self, csv_path=periodic_table_csv,
                 normalize_atomic_mass=True,
                 normalize_atomic_radius=True,
                 normalize_electronegativity=True,
                 normalize_ionization_energy=True,
                 normalize_electron_affinity=True,
                 imputation_atomic_radius=209.46, # mean value
                 imputation_electronegativity=1.18, # educated guess (based on neighbour elements)
                 imputation_ionization_energy=8.): # mean value
        self.data = pd.read_csv(csv_path)
        self.data['AtomicRadius'].fillna(imputation_atomic_radius, inplace=True)
        # Pm, Eu, Tb, Yb are inside the mp_e_form dataset, but have no electronegativity value
        self.data['Electronegativity'].fillna(imputation_electronegativity, inplace=True)
        self.data['IonizationEnergy'].fillna(imputation_ionization_energy, inplace=True)

        if normalize_atomic_mass:
            self._normalize_column('AtomicMass')
        if normalize_atomic_radius:
            self._normalize_column('AtomicRadius')
        if normalize_electronegativity:
            self._normalize_column('Electronegativity')
        if normalize_ionization_energy:
            self._normalize_column('IonizationEnergy')
        if normalize_electron_affinity:
            self._normalize_column('ElectronAffinity')    
    
    def _normalize_column(self, column):
        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def get_symbol(self, z: Optional[int] = None):
        if z is None:
            return self.data['Symbol'].to_list()
        else:
            return self.data.loc[z-1]['Symbol']
    
    def get_atomic_mass(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicMass'].to_list()
        else:
            return self.data.loc[z-1]['AtomicMass']
    
    def get_atomic_radius(self, z: Optional[int] = None):
        if z is None:
            return self.data['AtomicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AtomicRadius']
    
    def get_electronegativity(self, z: Optional[int] = None):
        if z is None:
            return self.data['Electronegativity'].to_list()
        else:
            return self.data.loc[z-1]['Electronegativity']
    
    def get_ionization_energy(self, z: Optional[int] = None):
        if z is None:
            return self.data['IonizationEnergy'].to_list()
        else:
            return self.data.loc[z-1]['IonizationEnergy']

    def get_oxidation_states(self, z: Optional[int] = None):
        if z is None:
            return list(map(self.parse_oxidation_state_string, self.data['OxidationStates'].to_list()))
        else:
            oxidation_states = self.data.loc[z-1]['OxidationStates']
            return self.parse_oxidation_state_string(oxidation_states, encode=True)
        
    def get_electron_affinity(self, z: Optional[int] = None):
        if z is None:
            return list(map(self.parse_oxidation_state_string, self.data['ElectronAffinity'].to_list()))
        else:
            oxidation_states = self.data.loc[z-1]['ElectronAffinity']
            return self.parse_oxidation_state_string(oxidation_states, encode=True)    
        
    def get_valence_electron(self, z: Optional[int]):
        NVs =  [self.data.loc[z-1]["NsValence"], 
                self.data.loc[z-1]["NpValence"],
                self.data.loc[z-1]["NdValence"],
                self.data.loc[z-1]["NfValence"],
                self.data.loc[z-1]["NValence"] ]
        return NVs
    
    def get_symbol_feature(self, element):
        elements = self.get_symbol()
        index = elements.index(element)+1
        feature_1 = np.array([
            self.get_atomic_mass(index),
            self.get_atomic_radius(index),
            self.get_electronegativity(index),
            self.get_ionization_energy(index),
        ])   
        return feature_1
    
    def atom_feature_map(self):
        self.feature = np.array([
            self.get_atomic_mass(),
            self.get_atomic_radius(),
            self.get_electronegativity(),
            self.get_ionization_energy(),
        ]).T

        new_feature = F.pad(torch.tensor(self.feature), (0, 0, 1, 9), mode="constant", value=0, )
        return new_feature
    
    @staticmethod
    def parse_oxidation_state_string(s: str, encode: bool=True):
        if encode:
            oxidation_states = [0] * 14
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(','):
                oxidation_states[int(i)-7] = 1
        else:
            oxidation_states = []
            if isinstance(s, float):
                return oxidation_states
            for i in s.split(','):
                oxidation_states.append(int(i))
        return oxidation_states

class AtomFeatureEncoder(nn.Module):
    def __init__(self, input_dim,  out_dim):
        super(AtomFeatureEncoder, self).__init__()
        self.pt = PeriodicTable()
        self.feature_map = self.pt.atom_feature_map()
        self.linear1 = nn.Linear(input_dim, out_dim)
        
    def forward(self, src):
        feature_map = self.feature_map.to(src.device)
        atom_fea = feature_map[src].to(torch.float32)
        atom_fea = self.linear1(atom_fea)

        return atom_fea

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    a = AtomFeatureEncoder()
    b = a.feature_map
    print("d") 