We implement here two approaches for the kaggle challenge of predicting taxi destinations
(https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/overview).

The goal of this challenge is to build a supervised learning model to predict the arrival point of a taxi
trip given certain information about the latter. The training data describes a full year (01/07/2013
to 30/06/2014) of the trajectories of the 442 taxis circulating in Porto (Portugal). More detail can be found in the link above.

We propose two solutions:
- a naive algorithm using the first points of the trajectory to predict the direction in wich the taxi is moving, then returning the most probable destination in that direction,
- The second method utilizes a neural network inspired by the research paper "Artificial Neural Networks Applied to Taxi Destination Prediction" from 2015. However, we have made numerous modifications to enable training the model using limited computation resources.


Download the data from 
https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data,
put in a folder "Data", then run the file data_cleaning.py. This will create a folder CleanData with a file "data_clean.csv", which will be used in our models.


Detailed information about our two solutions, including methodologies, results, and comparisons with alternative approaches are provided in "report.pdf"

