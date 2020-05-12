Credit Card Fraud Detection Dataset
NOTE: The dataset 'creditcard.csv' was too large to be committed, but it can be found on Kaggle
THIS IS A COST-SENSITIVE CLASSIFICATION PROBLEM

The dataset contains transactions made by credit cards, they are labeled as 
fraudulent or genuine. This is important for companies that have transaction 
systems to build a model for detecting fraudulent activities.

Classification methods are used to predict the class of different examples given
 their features. Standard methods aim at maximizing the accu racy of the 
predictions, in which an example is correctly classified if the predicted class 
is the same the as true class. This traditional approach assumes that all 
correctly classified and misclassified examples carry the same cost. This, 
however, is not the case in many real-world appli cations. Methods that use 
different misclassification costs are known as cost-sensitive classifiers. 
Typical cost-sensitive approaches assume a constant cost for each type of error,
 in the sense that, the cost depends on the class and is the same among examples 
[Elkan, 2001; Kim et al., 2012]. Nevertheless, this class-dependent approach is 
not realistic in many real world applications.

For example in credit card fraud detection, failing to detect a fraudulent 
transaction may have an economical impact from a few to thousands of Euros, 
depending on the particular transaction and card holder [Sahin et al., 2013].
