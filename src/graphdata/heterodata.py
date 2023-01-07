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
        
        topic_df = pd.read_csv(self.raw_paths[0], header=0)
        
        topic_mapping = {idx: i for i, idx in enumerate(topic_df['id'])}
        
        title = topic_df['title'].apply(lambda x: self.translator.translate(x).text)
        #eng_translate = lambda x: self.translator.translate(x).text
        #title = np.array(list(map(eng_translate,topic[:,1])
        description = topic_df['description'].apply(
            lambda x: self.translator.translate(x).text)
        with torch.no_grad():
            title_c = model.encode(title.values, 
                                   show_progress_bar=True,
                                   convert_to_tensor=True).to(device)
        
            description_c = model.encode(description.values, 
                                         show_progress_bar=True,
                                         convert_to_tensor=True).to(device)
        channel_c = topic_df['category'].astype('category').cat.codes.values
        category_c = topic_df['category'].astype('category').cat.codes.values
        language_c = topic_df['language'].astype('category').cat.codes.values
        #parent_c
        parent = topic_df['parent'].values
        
        dst_parent = []
        src_parent = []
        for i,idx in enumerate(parent) :
            if idx != 'bg':
                dst_parent.append(topic_mapping[idx])
                src_parent.append(i)
        parent_edge_index = torch.tensor([src_parent, dst_parent])  
        
        level_c = topic_df['level'].astype('category').cat.codes.values
        has_content_c = topic_df['has_content'].astype('category').cat.codes.values
        
        data['topic'].x= torch.cat([title_c,
                                    description_c,
                                    channel_c,
                                    category_c,  
                                    language_c,
                                    level_c,
                                    has_content_c], dim=-1)
        
        
        content_df = pd.read_csv(self.raw_paths[1], header=0)
        
        content_mapping = {idx: i for i, idx in enumerate(content_df['id'])}
 
        c_title = content_df['title'].apply(
            lambda x: self.translator.translate(x).text)
        c_description = content_df['description'].apply(
            lambda x: self.translator.translate(x).text)
        c_text = content_df['text'].apply(
            lambda x: self.translator.translate(x).text)
        with torch.no_grad():
            c_title_c = model.encode(c_title.values, 
                                     show_progress_bar=True,
                                     convert_to_tensor=True).to(device)
        
            c_description_c = model.encode(c_description.values, 
                                           show_progress_bar=True,
                                           convert_to_tensor=True).to(device)
            
            c_text_c = model.encode(c_text.values, 
                                    show_progress_bar=True,
                                    convert_to_tensor=True).to(device)
            
        c_language_c = topic_df['language'].astype('category').cat.codes.values
        c_kind_c = topic_df['kind'].astype('category').cat.codes.values
        c_copyright_holder_c = topic_df['copyright_holder'].astype(
            'category').cat.codes.values
        c_license_c =topic_df['license'].astype('category').cat.codes.values

        data['content'].x= torch.cat([c_title_c,
                                      c_description_c,
                                      c_kind_c,
                                      c_text_c,  
                                      c_language_c,
                                      c_copyright_holder_c,
                                      c_license_c], dim=-1)

        correlation_df = pd.read_csv(self.raw_paths[2], header=0)
        
        topic_correlation = correlation_df['topic_id'].values
        
        dst_correlation = []
        src_correlation = []
        for i, to_id in enumerate(topic_correlation) :
            for co_id in topic_correlation['content_ids'][i]:
                src_correlation.append(topic_mapping[to_id])
                dst_correlation.append(content_mapping[co_id])
            
        correlation_edge_index = torch.tensor([src_correlation, dst_correlation])
        
        data['topic','parent','topic'].edge_index = parent_edge_index
        data['topic','parent','content'].edge_index = correlation_edge_index
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
