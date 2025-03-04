from src.data_processing.pipeline_manager import DataPipelineManager, get_processing_configs
from src.utils.logging import Logger
import time

def main():
    logger = Logger().get_logger()
    pipeline = DataPipelineManager()
    configs = get_processing_configs()

    logger.info("Starting data processing pipeline...")
    start_time = time.time()

    # Process each file and collect results
    results = {}
    for filename, config in configs.items():
        logger.info(f"Processing {filename}...")
        file_start = time.time()
        
        try:
            success = pipeline.process_file(filename, config)
            file_time = time.time() - file_start
            
            results[filename] = {
                'success': success,
                'time': f"{file_time:.2f} seconds"
            }
            
            status = "✓" if success else "✗"
            logger.info(f"{filename}: {status} ({file_time:.2f}s)")
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results[filename] = {
                'success': False,
                'time': 'N/A',
                'error': str(e)
            }

    # Print summary
    total_time = time.time() - start_time
    logger.info("\nProcessing Summary:")
    logger.info("=" * 50)
    
    successful = sum(1 for r in results.values() if r['success'])
    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(results) - successful}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    logger.info("\nDetailed Results:")
    logger.info("=" * 50)
    for filename, result in results.items():
        status = "✓" if result['success'] else "✗"
        logger.info(f"{filename}: {status} ({result['time']})")
        if not result['success'] and 'error' in result:
            logger.info(f"  Error: {result['error']}")

if __name__ == "__main__":
    main()