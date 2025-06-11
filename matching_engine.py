"""
Product Matching Engine
=======================

Core matching algorithms including exact, fuzzy (baseline), vector similarity, 
and LLM-based matching using Ollama and Gemma.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple
from pathlib import Path

# Third-party imports
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import faiss
import ollama
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class ProductMatchingEngine:
    """
    Core engine for product matching using multiple algorithms.
    
    Implements a hierarchical matching approach:
    1. Exact string matching
    2. Fuzzy string matching (baseline for comparison) 
    3. Vector similarity matching
    4. LLM validation using Ollama/Gemma
    5. Size validation using LLM
    """
    
    def __init__(self, config: Config):
        """
        Initialize the matching engine.
        
        Args:
            config: Configuration object with matching parameters
        """
        self.config = config
        self.embedder = None
        self.vector_index = None
        self.internal_embeddings = None
        self.internal_data = None
        
    def exact_match(
        self, 
        internal_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform exact string matching between internal and external products.
        
        Args:
            internal_data: Internal products DataFrame
            external_data: External products DataFrame
            
        Returns:
            External data with exact matches populated
        """
        external_data = external_data.copy()
        
        # Match against NAME column
        mask_name = external_data['PRODUCT_NAME'].isin(internal_data['NAME'])
        external_data.loc[mask_name, 'MATCHED_VALUE'] = \
            external_data.loc[mask_name, 'PRODUCT_NAME']
        
        # Match against LONG_NAME column (only if still unmatched)
        mask_long = (
            external_data['PRODUCT_NAME'].isin(internal_data['LONG_NAME']) & 
            external_data['MATCHED_VALUE'].isna()
        )
        external_data.loc[mask_long, 'MATCHED_VALUE'] = \
            external_data.loc[mask_long, 'PRODUCT_NAME']
        
        exact_matches = external_data['MATCHED_VALUE'].notna().sum()
        logger.info(f"Exact matching found {exact_matches} matches")
        
        return external_data
    
    def fuzzy_match(
        self, 
        internal_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply fuzzy string matching for products without exact matches.
        
        Args:
            internal_data: Internal products DataFrame
            external_data: External products DataFrame
            
        Returns:
            External data with fuzzy matches populated
        """
        external_data = external_data.copy()
        external_data_names = external_data['PRODUCT_NAME'].dropna().unique()
        
        for idx, row in internal_data.iterrows():
            internal_name = row['NAME']
            internal_long = row['LONG_NAME']
            
            best_score = 0
            best_match = None

            # Check against external names
            for external_name in external_data_names:
                if pd.notna(internal_name):
                    score_name = fuzz.token_set_ratio(str(internal_name), str(external_name))
                    if score_name >= self.config.fuzzy_threshold and score_name > best_score:
                        best_score = score_name
                        best_match = external_name

                if pd.notna(internal_long):
                    score_long = fuzz.token_set_ratio(str(internal_long), str(external_name))
                    if score_long >= self.config.fuzzy_threshold and score_long > best_score:
                        best_score = score_long
                        best_match = external_name

            if best_match:
                # Update external data with the match
                mask = external_data['PRODUCT_NAME'] == best_match
                external_data.loc[mask, 'MATCHED_VALUE'] = best_match
        
        fuzzy_matches = external_data['MATCHED_VALUE'].notna().sum()
        logger.info(f"Fuzzy matching found {fuzzy_matches} total matches")
        
        return external_data
    
    def build_vector_database(self, internal_data: pd.DataFrame) -> None:
        """
        Build vector database from internal product data.
        
        Args:
            internal_data: Internal products DataFrame
        """
        logger.info("Loading sentence transformer model...")
        self.embedder = SentenceTransformer(self.config.embedding_model)
        self.internal_data = internal_data
        
        # Create embeddings for both NAME and LONG_NAME columns
        logger.info("Creating embeddings for internal products...")
        column_embeddings = []
        
        for col in ['NAME', 'LONG_NAME']:
            col_texts = internal_data[col].fillna('').tolist()
            col_embeds = self.embedder.encode(col_texts, convert_to_numpy=True)
            column_embeddings.append(col_embeds)
        
        # Average the embeddings from both columns
        self.internal_embeddings = np.mean(np.stack(column_embeddings, axis=0), axis=0)
        
        # Build FAISS index
        logger.info("Building FAISS vector index...")
        dimension = self.internal_embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(self.internal_embeddings)
        
        logger.info(f"Vector database built with {len(internal_data)} products")
    
    def vector_match(
        self, 
        internal_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform vector similarity matching.
        
        Args:
            internal_data: Internal products DataFrame
            external_data: External products DataFrame
            
        Returns:
            External data with vector similarity results
        """
        if self.vector_index is None:
            raise ValueError("Vector database not built. Call build_vector_database() first.")
        
        external_data = external_data.copy()
        matched_results = []
        
        logger.info("Performing vector similarity matching...")
        
        for idx, ext_row in external_data.iterrows():
            ext_name = ext_row['PRODUCT_NAME']
            ext_vector = self.embedder.encode([ext_name], convert_to_numpy=True)
            
            # Get top-k matches
            distances, indices = self.vector_index.search(ext_vector, self.config.top_k_matches)
            
            # Get best match (lowest distance)
            best_idx = indices[0][0]
            best_distance = distances[0][0]
            matched_row = internal_data.iloc[best_idx]
            
            matched_results.append({
                'EXTERNAL_PRODUCT_NAME': ext_name,
                'INTERNAL_NAME': matched_row['NAME'],
                'INTERNAL_LONG_NAME': matched_row['LONG_NAME'],
                'DISTANCE': best_distance,
                'EXTERNAL_INDEX': idx
            })
        
        # Add vector matching results to external data
        vector_df = pd.DataFrame(matched_results)
        
        # Only keep matches below threshold
        good_matches = vector_df[vector_df['DISTANCE'] < self.config.vector_similarity_threshold]
        
        for _, match in good_matches.iterrows():
            ext_idx = match['EXTERNAL_INDEX']
            external_data.loc[ext_idx, 'VECTOR_MATCH'] = match['INTERNAL_LONG_NAME']
            external_data.loc[ext_idx, 'MATCH_DISTANCE'] = match['DISTANCE']
        
        vector_matches = good_matches.shape[0]
        logger.info(f"Vector matching found {vector_matches} good matches")
        
        return external_data
    
    def llm_validation_match(
        self, 
        internal_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply LLM validation using Ollama/Gemma for intelligent matching.
        
        Args:
            internal_data: Internal products DataFrame
            external_data: External products DataFrame
            
        Returns:
            DataFrame with LLM-validated matches
        """
        if self.vector_index is None:
            raise ValueError("Vector database not built. Call build_vector_database() first.")
        
        # Load prompt template
        prompt_path = Path(self.config.prompt_dir) / 'prompt_0shot.txt'
        if not prompt_path.exists():
            # Create default prompt if file doesn't exist
            self._create_default_prompts()
        
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        
        combined_results = []
        
        logger.info("Applying LLM validation using Ollama/Gemma...")
        
        for idx, ext_row in tqdm(external_data.iterrows(), total=len(external_data)):
            ext_name = ext_row['PRODUCT_NAME']
            ext_vector = self.embedder.encode([ext_name], convert_to_numpy=True)
            
            # Get top-k vector matches for LLM evaluation
            distances, indices = self.vector_index.search(ext_vector, self.config.top_k_matches)
            
            matched_long_name = None  # Default to None if no valid match found
            
            # Try each of the top-k matches
            for j in range(self.config.top_k_matches):
                best_idx = indices[0][j]
                best_distance = distances[0][j]
                matched_row = internal_data.iloc[best_idx]
                
                name = matched_row['NAME']
                long_name = matched_row['LONG_NAME']
                
                # Call LLM for validation using Ollama/Gemma
                llm_result = self._llm_match_check(prompt_template, ext_name, name, long_name)
                is_valid_match = llm_result or best_distance < self.config.vector_similarity_threshold * 0.4
                
                if is_valid_match:
                    matched_long_name = long_name
                    break  # Stop if we find a valid match
            
            combined_results.append({
                'EXTERNAL_PRODUCT_NAME': ext_name,
                'INTERNAL_LONG_NAME': matched_long_name,
            })
        
        # Convert to DataFrame
        final_df = pd.DataFrame(combined_results)
        
        llm_matches = final_df['INTERNAL_LONG_NAME'].notna().sum()
        logger.info(f"LLM validation found {llm_matches} validated matches")
        
        return final_df
    
    def size_validation_check(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final size validation using LLM to ensure exact size matching.
        
        Args:
            results_df: DataFrame with initial matches
            
        Returns:
            DataFrame with size-validated matches
        """
        # Load size validation prompt
        prompt_path = Path(self.config.prompt_dir) / 'prompt_size.txt'
        if not prompt_path.exists():
            self._create_default_prompts()
        
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        
        results_df = results_df.copy()
        
        logger.info("Applying final size validation...")
        
        for idx, row in results_df.iterrows():
            external_name = row['EXTERNAL_PRODUCT_NAME']
            internal_name = row['INTERNAL_LONG_NAME']
            
            # Only check if internal name exists
            if pd.notnull(internal_name):
                llm_result = self._llm_size_check(prompt_template, external_name, internal_name)
                if llm_result == 'YES':  # YES means there's a size discrepancy
                    results_df.at[idx, 'INTERNAL_LONG_NAME'] = None
                    logger.debug(f"Size discrepancy found: {external_name} vs {internal_name}")
        
        final_matches = results_df['INTERNAL_LONG_NAME'].notna().sum()
        logger.info(f"Final matches after size validation: {final_matches}")
        
        return results_df
    
    def _llm_match_check(
        self, 
        prompt_template: str, 
        external_name: str, 
        name: str, 
        long_name: str
    ) -> bool:
        """
        Using Ollama/Gemma LLM to validate if products match exactly.
        
        Args:
            prompt_template: Template for the LLM prompt
            external_name: External product name
            name: Internal product short name
            long_name: Internal product long name
            
        Returns:
            True if LLM confirms exact match, False otherwise
        """
        prompt = prompt_template.format(
            external_name=external_name,
            name=name,
            long_name=long_name
        )
        
        try:
            # Call Ollama with Gemma model
            response = ollama.chat(
                model=self.config.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': self.config.llm_temperature}
            )
            
            result = response['message']['content'].strip().upper()
            return result == 'YES'
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return False
    
    def _llm_size_check(
        self, 
        prompt_template: str, 
        external_name: str, 
        internal_name: str
    ) -> str:
        """
        Use Ollama/Gemma LLM to check for size discrepancies.
        
        Args:
            prompt_template: Template for the size validation prompt
            external_name: External product name
            internal_name: Internal product name
            
        Returns:
            'YES' if there's a size discrepancy, 'NO' if sizes match
        """
        prompt = prompt_template.format(
            external_name=external_name,
            internal_name=internal_name
        )
        
        try:
            # Call Ollama with Gemma model
            response = ollama.chat(
                model=self.config.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': self.config.llm_temperature}
            )
            
            return response['message']['content'].strip().upper()
            
        except Exception as e:
            logger.error(f"LLM size check failed: {str(e)}")
            return 'NO'  # Default to no discrepancy if LLM fails
    