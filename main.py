"""
Product Matching System - Main Application
==========================================

An intelligent system for automatically matching external supplier products 
with internal inventory items using vector similarity and LLM validation.

Author: Taha Azizi
Date: 2025-05-01
"""

import logging
import argparse
from pathlib import Path
from typing import Optional

from data_processor import DataProcessor
from matching_engine import ProductMatchingEngine
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('product_matching.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ProductMatchingSystem:
    """
    Main orchestrator for the product matching system.
    
    This class coordinates the entire matching pipeline from data loading
    to final result generation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the product matching system.
        
        Args:
            config: Configuration object containing system parameters
        """
        self.config = config
        self.data_processor = DataProcessor()
        
        # Verify prompts are loaded
        if not hasattr(config, 'prompts') or not config.prompts:
            raise ValueError("Configuration must include loaded prompts")
            
        self.matching_engine = ProductMatchingEngine(config)
        logger.info("Product matching system initialized with prompts loaded")
        
    def run_matching_pipeline(
        self, 
        internal_data_path: str, 
        external_data_path: str,
        output_path: str
    ) -> None:
        """
        Execute the complete product matching pipeline.
        
        Args:
            internal_data_path: Path to internal products CSV
            external_data_path: Path to external products CSV  
            output_path: Path for output results CSV
        """
        try:
            logger.info("Starting product matching pipeline...")
            
            # Step 1: Load and clean data
            logger.info("Loading and cleaning data...")
            internal_data, external_data = self.data_processor.load_and_clean_data(
                internal_data_path, external_data_path
            )
            
            # Step 2: Attempt exact matching
            logger.info("Attempting exact string matching...")
            external_data = self.matching_engine.exact_match(
                internal_data, external_data
            )
            
            # Step 3: Apply fuzzy matching for unmatched items
            logger.info("Applying fuzzy matching...")
            external_data = self.matching_engine.fuzzy_match(
                internal_data, external_data
            )
            
            # Step 4: Build vector database and apply semantic matching
            logger.info("Building vector database...")
            self.matching_engine.build_vector_database(internal_data)
            
            logger.info("Applying vector similarity matching...")
            external_data = self.matching_engine.vector_match(
                internal_data, external_data
            )
            
            # Step 5: Apply LLM validation using few-shot prompting
            logger.info("Applying LLM validation with few-shot prompting...")
            final_results = self.matching_engine.llm_validation_match(
                internal_data, external_data
            )
            
            # Step 6: Apply size validation as final check
            logger.info("Applying final size validation...")
            final_results = self.matching_engine.size_validation_check(
                final_results
            )
            
            # Step 7: Save results
            logger.info(f"Saving results to {output_path}...")
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            final_results.to_csv(output_path, index=False)
            
            # Generate summary statistics
            self._generate_summary_stats(final_results)
            
            logger.info("Product matching pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in matching pipeline: {str(e)}")
            raise
    
    def _generate_summary_stats(self, results_df) -> None:
        """Generate and log summary statistics for the matching results."""
        total_external = len(results_df)
        matched_count = results_df['INTERNAL_LONG_NAME'].notna().sum()
        match_rate = (matched_count / total_external) * 100
        
        # Count size discrepancies if column exists
        size_discrepancies = 0
        if 'SIZE_DISCREPANCY' in results_df.columns:
            size_discrepancies = (results_df['SIZE_DISCREPANCY'] == 'YES').sum()
        
        logger.info(f"=== MATCHING SUMMARY ===")
        logger.info(f"Total external products: {total_external}")
        logger.info(f"Successfully matched: {matched_count}")
        logger.info(f"Match rate: {match_rate:.2f}%")
        logger.info(f"Unmatched products: {total_external - matched_count}")
        logger.info(f"Size discrepancies found: {size_discrepancies}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Intelligent Product Matching System"
    )
    parser.add_argument(
        "--internal-data", 
        required=True,
        help="Path to internal products CSV file"
    )
    parser.add_argument(
        "--external-data", 
        required=True,
        help="Path to external products CSV file"
    )
    parser.add_argument(
        "--output", 
        default="results/final_matches.csv",
        help="Output path for results CSV"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration (this will also load prompts)
        logger.info("Loading configuration and prompts...")
        config = Config.from_file(args.config)
        
        # Initialize and run the system
        system = ProductMatchingSystem(config)
        system.run_matching_pipeline(
            args.internal_data,
            args.external_data, 
            args.output
        )
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()