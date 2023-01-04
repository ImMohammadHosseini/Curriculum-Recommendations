#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:30:56 2023

@author: mohammad
"""

from typing import Callable, List, Optional
#from langdetect import detect_langs
from googletrans import Translator
from sentence_transformers import SentenceTransformer

import os
import torch
import pandas as pd

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class CurriculumDataset (InMemoryDataset) :
    '''data_ref: 
       https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data'''
    url = ('')
    
    def __init__ (self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "all-MiniLM-L6-v2") :
        self.model_name = model_name
        self.translator = Translator()
        super().__init__(os.getcwd()+root, transform, pre_transform)
        print(self.raw_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.translator = Translator()
        print(self.raw_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return [
            'topics.csv', 'content.csv', 'correlations.csv', 
            'sample_submission.csv'
        ]
    
    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'
    
    def download(self):
        pass

    
    def process(self):
        data = HeteroData()
        model = SentenceTransformer(self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        node_types = ['topic', 'content']
        
        topic_df = pd.read_csv(self.raw_paths[0], index_col='id', header=0)
        
        topic_mapping = {idx: i for i, idx in enumerate(topic_df.index)}
        
        title = topic_df['title'].apply(lambda x: self.translator.translate(x).text)
        description = topic_df['description'].apply(
            lambda x: self.translator.translate(x).text)
        with torch.no_grad():
            title_c = model.encode(title.values, 
                                   show_progress_bar=True,
                                   convert_to_tensor=True).to(device)
        
            description_c = model.encode(description.values, 
                                         show_progress_bar=True,
                                         convert_to_tensor=True).to(device)
        #channel_c
        category_c = topic_df['category'].astype('category').cat.codes.values
        language_c = topic_df['language'].astype('category').cat.codes.values
        #parent_c
        level_c = topic_df['level'].astype('category').cat.codes.values
        has_content_c = topic_df['has_content'].astype('category').cat.codes.values
        
        data['topic'].x= torch.cat([title_c,
                                    description_c,
                                    #channel_c,
                                    category_c,  
                                    language_c,
                                    #parent_c,
                                    level_c,
                                    has_content_c], dim=-1)
        #TODO
# =============================================================================
#         title
#         description
#         language
#         kind
#         text
#         copyright_holder
#         licenses
# =============================================================================
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
