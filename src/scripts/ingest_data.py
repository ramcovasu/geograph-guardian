from src.data_processing.arango_ingestor import ArangoIngestor, get_ingestion_configs, ingest_all_files
from src.utils.logging import Logger
import time

def main():
    logger = Logger().get_logger()
    start_time = time.time()

    logger.info("Starting ArangoDB ingestion pipeline...")
    
    try:
        results = ingest_all_files()
        
        # Print summary
        total_time = time.time() - start_time
        logger.info("\nIngestion Summary:")
        logger.info("=" * 50)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(results) - successful}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        logger.info("\nDetailed Results:")
        logger.info("=" * 50)
        for filename, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{filename}: {status}")
            
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()