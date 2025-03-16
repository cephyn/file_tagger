import sys
from PySide6.QtWidgets import QApplication
from login import LoginScreen

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show the login screen
    login_screen = LoginScreen()
    login_screen.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()