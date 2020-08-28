# TextCategorization 
  This module contains several function/methods which will be executed in training and testing of a model
  
  ### Module is mainly divided in four parts
    
    1) Data Load
    2) Text Pre-processing
    3) Word Embedding
    4) Machine Learnig model and Prediction
    
   This methods will be called in the training and testing phase of a model

To run the module you need to run follwing commands, first is to train a modeland save model in pickle format
Another is to test the model on test dataset 

**python -c "from TextCategorization import*; TextCategorization.train()"**

**python -c "from TextCategorization import*; TextCategorization.test()"**
