# Challenge 21 Neural Networks Supervised Deep Machine Learning, using Google Colab

## Objective 
* To build a tool for the nonprofit foundation 'Alphabet Soup' to help in selecting those organization applicants for funding who will have the best chance of success in their ventures. 

* Apply Deep Machine Learning and Neural Networks using features (metadata) from the 'Charity' Dataset provided by Alphabet Soup to create a binary classifier to predict the applicant(s) / organization(s) who will most probably be successful if funded by Alphabet Soup. 

* Optimize your model to achieve a target predictive accuracy higher than 75%.

## Project Overview 
* Designed a deep learning neural network model using TensorFlow in file 'AlphabetSoupCharity_Model.ipynb' to create the binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset.

* Designed additional 5 Models with auto-Optimization , and other models in code file 'AlphabetSoupCharity_Optimization. 

## Analysis
* The Accuracy did not improve significantly across the different optimization models. 

* The Loss increased slightly across the different auto-optimization models in comparison to the First manual model. 

* Accuracy decreased with use of fewer Features from the 72.5% using 9 columns to 69.4% using 5 columns when all else is constant/same.  The Loss increased from 59.5% to 60.5%.

* The target predictive accuracy higher than 75% was not attainable with the features and data provided in the input dataset.  

* The highest Accuracy attained is in the first Model with 80 neurons in First hidden layer, and 30 in the Second Layer using 9 column Features:
    * 74.1% in the Train Data which is about 70% of the entire DataSet, and 
    * 71.5% in the Test Data which makes up about 30% of the entire input dataset.  

## Details of findings: 
* First model in the AlphabetSoupCharity_Model.ipynb uses 9 columns of the metadata, with 80 neurons in the first hidden layer, and 30 in the second hidden layer has:  
        74.1% Accuracy in the Training Data, Loss of 53.4%
        72.5% Accuracy in the Testing Data,  Loss of 56%

* Optimization Models, and other Models created in the AlphabetSoupCharity_Optimization.ipynb. 
 These are steps taken in attempts to create models that may improve / increase the Accuracy, and decrease the Loss. 

        Sequential Models created using auto-optimization with hyperparameter options using keras tuner. 
        Activation Choices: 'relu','tanh','sigmoid'. Top 3 Models displayed in code file
            The Best Model and the Second Best Models both have an Accuracy of 72.6%. 
            Best Model Loss is 57.7%, and Second Best Model Loss is 56.5%

        More Neurons Model created with 129 neurons in the first hidden layer, and 50 in the second hidden layer. 
            74.2% Accuracy in the Training Data, and Loss of 53.1%
            72.3% Accuracy in the Test Data, and Loss of 58.6%
        
        Fewer Features Model created with using 5 colums (Features) with 80 neurons in the first hidden layer and 30 in the second hidden layer.
            70.1% Accuracy in the Training Data, and Loss of 59.5%
            69.4% Accuracy in the Test Data, and Loss of 60.5% 
            Accuracy in this case decreased with consideration to Fewer Features. 

Model files are in .keras file formats in the Output folder. 

* Images of First Model


<img src="Images\nn_model_first_1.png" alt="First Model" width="500" height="800"> <img src="Images\nn_model_first.png" alt="First Model" width="500" height="800">
 
## Programs 
* Jupyter Notebook file used in Google Colab to create the Prediction Models.  
* Google Colab is a hosted Jupyter Notebook service that requires no setup to use and provides free access to computing resources, including GPUs (Graphic Processing Units) and TPUs (Tensor Processing Units).
* Pandas DataFrame
* TensorFlow 
* Neural Network & Deep Machine Learning. (A Neural Network Model with more than one hidden layer is Deep Neural Network or Deep Learning Model.)  

## Process
* Upload Jupyter Notebook files in Google Colab
    'AlphabetSoupCharity_Model.ipynb' - the renamed provided starter file, and 
    'AlphabetSoupCharity_Optimization.ipynb - new file for additional optimization models

* Import dependencies and modules. 

* Input Dataset file imported, read and saved from url "https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv" 
The charity_data.csv downloaded from the website and saved in the Resources folder for exploration. 

* Explore the data. 
    There are 12 columns, and 34299 rows of data for each organization that have received funding from the company Alphabet Soup in previous years. DateTime not included in data. There is a column header row in the .csv data.  Downloaded the .csv data file from the website url to explore the data. 
    .csv input data file downloaded from the website and placed in the Resources folder. 

    Columns / Metadata: 
    * EIN  - Identification  
    * NAME — Identification 
    * APPLICATION_TYPE — Alphabet Soup application type
    * AFFILIATION — Affiliated sector of industry
    * CLASSIFICATION — Government organization classification
    * USE_CASE — Use case for funding
    * ORGANIZATION — Organization type
    * STATUS — Active status
    * INCOME_AMT — Income classification
    * SPECIAL_CONSIDERATIONS — Special considerations for application
    * ASK_AMT — Funding amount requested
    * IS_SUCCESSFUL — Was the money used effectively

* Preprocess the Data
    Drop / Remove the unbeneficial columns.  The EIN and Name columns dropped from the DataFrame 

    Model Target y variable:'IS_SUCCESSFUL'  

    Model Features X variables: 
        APPLICATION_TYPE 
        AFFILIATION 
        CLASSIFICATION 
        USE_CASE 
        ORGANIZATION 
        STATUS 
        INCOME_AMT 
        SPECIAL_CONSIDERATIONS 
        ASK_AMT 

* Neural Network Models compiled, trained (fitted) and evaluated.

* Bins created for Feature(s) with more than 10 unique values. 
Determine the 'cutoff' point for each Feature(s) using the number of unique value data points and grouping these 'rate' categorical variables, the outliers, in a bin called 'Other'. 

* pd.get_dummies() used to encode categorical variables with Data Type: Object to numeric data type.
For example, in the AlphabetSoupCharity_Model code file which contains one Model, there are 7 Categorical Variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, SPECIAL_CONSIDERATIONS. 

* Preprocessed data is split:  
    first into X Features array, and y Target array, then
    using the train_test_split function to split into training and testing datasets using a specific random_state.  
    In this case random_state of 42 is used. 

    Split data: 
        X_train shape: (25724, 43)
        X_test shape: (8575, 43)
        y_train shape: (25724,)
        y_test shape: (8575,)
    There are 43 Features / Columns in the X variables. 

Total Rows: 34,299.  Each row is an organization.  The .csv file included a column header row. 

* StandardScaler used to normalize and scale the training and testing features X variables by creating a StandardScaler instance, and then fitting it to the training data and then using the transform function.

* There is more data in the Training dataset about 70% typically, and 30% in Testing dataset.  
Data is distributed evenly by measuring weight of each Feature, and split between Training and Testing Data Set. 

* Optimizing the Model: 
    * Auto-optimization used with hyperparameters to arrive at top 3 models.  Step = 5. 
    QUESTION: is is what is meant by 'Create a callback that saves the model's weights every five epochs.' 

    * Adding neurons, adding hidden layers, number of epoch, reducing the number of Features, selecting the appropriate Activation Function(s) are levers or methods for perhaps improving performance, run times, and Accuracy and decreasing Loss.  
    Removing of outliers, or binning  with having the outliers or rare occurances of data points in a bin group called 'rare' or 'other' 

    * A good rule of thumb is for first hidden layer to have 2-3 times the number of neurons as there are input dimensions/features. 
    There are 43 features in our processed dataset. 

    * Typically at the most 3 hidden layers are used in most Deep Learning Models. 

    * Particularly since we have (not too much) 8575 rows with 43 columns of data in our Testing Data set, having too many neurons or too many hidden layers could overfit the data


* Output files: 

    First Model from the AlphabetSoupCharity_Model.ipynb: 

        AlphabetsoupCharity_model.h5, 

        AlphabetSoupCharity_model.keras  Keras format is the updated file format to save models. 

    Subsequent models generated in the AlphabetSoupCharity_Optimization.ipynb: 

        AlphabetSoupChaity_optimization_1_best.keras - the best optimization model

        AlphabetSoupChaity_optimization_2.keras - the second best optimization model

        AlphabetSoupChaity_optimization_3.keras - the third optimization model 

        AlphabetSoupChaity_model_2.keras - model with more neurons
        
        AlphabetSoupChaity_model_3.keras - model with fewer features
