//Activating venv stuff
venv\Scripts\activate

//if venv is not there
python -m venv venv
venv\Scripts\activate

//Then install tensorflow
pip install tensorflow

//check tensorflow
pip show tensorflow

//Cmd to test
python -m src.test_data

//verifying tensorflow installation
python
>>import tensorflow as tf
>>print(tf.__version__)
>>exit()

//run this cmd after evrything
python -m src.test_data



//To train basically 20 epoch not necessary to complete all 20 stop for 7 or 10
//so that auc is increased (this tep takes more time)
python -m src.train

//incase above cmd have error run the below ones first then againrun that
pip install scipy


//Installing modules after training so that we can run code
pip install scikit-learn

//To see if the module is installed
pip show scikit-learn


//to check the confusion matrix thingy
python -m src.evaluate

//module to run frontend
pip install flask

//command to run the frontend
python app/app.py