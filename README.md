# Hand Written Digit Classifier With KNN
This project utilizes a subset of the MNIST handwriten dataset and a K-Nearest 
Neighbor approach to compare a sample to existing data in n-space via euclidean distance 

Training Data from https://yann.lecun.com/exdb/mnist/

Statistics:<br>
With a K value of 10 and a sample of 5000 images (Excluded from training)
we get 97.0% accuracy.

Manual testing results vary when writing digits digitally.

## Run Project 

Clone the project  

~~~bash  
git clone https://github.com/BeefStew34/DigitIdentifierKNN-Python
~~~

Go to the project directory  

~~~bash  
cd DigitIdentifierKNN-Python
~~~

Install dependencies  
~~~bash  
pip install -r requirements.txt
~~~

Finally...
~~~bash  
python main.py
~~~
