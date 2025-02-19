On command prompt
1. Cd to your project
2. To create the virtual environment:
python -m venv venv
3. To activate the virtual environment:
venv\Scripts\activate
4. To install all the requirements:
Pip install requiremnts.txt
Requirements.txt
opencv-pyhton==4.5.3
numpy==1.21.0
sklearn==0.24.2
Flask
5. Run your python file by:
python app.py

Change the path for data files to where you have stored your files


docker build -t flask-signature-verification .


docker run -p 5001:5001 flask-signature-verification
http://localhost:5001/


# Signature-Verification-using-SVM-and-Flask
