import sys
import os
import logging
import traceback
from datetime import datetime
from PySide6.QtWidgets import QApplication
from login import LoginScreen

# Set up error logging to file
def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'file_tagger_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Also log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logging.error("Uncaught exception", 
                     exc_info=(exc_type, exc_value, exc_traceback))
        
    # Install exception handler
    sys.excepthook = handle_exception
    
    return log_file

def main():
    # Initialize logging first
    log_file = setup_logging()
    
    # Fix for ChromaDB ONNX issue in PyInstaller
    os.environ["CHROMADB_DISABLE_ONNX"] = "1"
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Create and show the login screen
        login_screen = LoginScreen()
        login_screen.show()
        
        logging.info(f"Application started successfully. Logs will be written to {log_file}")
        
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Error in main application: {str(e)}")
        logging.error(traceback.format_exc())
        raise  # Re-raise the exception after logging

if __name__ == '__main__':
    main()