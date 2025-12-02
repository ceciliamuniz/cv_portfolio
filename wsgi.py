# WSGI configuration for PythonAnywhere
# This file tells PythonAnywhere how to serve your Flask application

# Add your project directory to the Python path
import sys
import os

# The path to your project directory - PythonAnywhere typically places this in /home/yourusername/mysite
path = '/home/ceciliamuniz/cv_portfolio'
if path not in sys.path:
    sys.path.append(path)

# Change to your project directory
os.chdir(path)

# Import your Flask application and rename it to 'application' for WSGI
from app import app as application

# Make sure Flask knows it's in production
application.config['ENV'] = 'production'
application.config['DEBUG'] = False

# This is needed for PythonAnywhere
if __name__ == "__main__":
    application.run()