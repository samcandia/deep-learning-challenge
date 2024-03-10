# Background

The nonprofit foundation Alphabet Soup wanted a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Steps Taken

### 1) Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset. 
- Read in the charity_data.csv to a Pandas DataFrame, and was sure to identify the following in the dataset:
##### - What variable(s) are the target(s) for my model?
- The 'IS_SUCCESSFUL' column 
##### - What variable(s) are the feature(s) for my model?
- 1. AFFILIATION—Affiliated sector of industry
  2. CLASSIFICATION—Government organization classification
  3. USE_CASE—Use case for funding
  4. ORGANIZATION—Organization type
  5. STATUS—Active status
  6. INCOME_AMT—Income classification
  7. SPECIAL_CONSIDERATIONS—Special considerations for application
  8. ASK_AMT—Funding amount requested
* The "EIN" and "NAME" columns were dropped since they are neither targets or features.

- Determined the number of unique values for each column. For columns that have more than 10 unique values, determined the number of data points for each unique value.

- Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then checked if the binning was successful.

- Used pd.get_dummies() to encode categorical variables.

- Split the preprocessed data into a features array, X, and a target array, y. Used these arrays and the train_test_split function to split the data into training and testing datasets.

- Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### 2) Using my knowledge of TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset by compiling, Training, and Evaluating the Model

- For the neural network model, I selected two hidden layers with 80 and 30 neurons respectively, and used 'relu' as the activation function for hidden layers, with 'sigmoid' for the output layer to cater to binary classification. This structure was chosen based on the principle that a multi-layer network with 'relu' can learn complex patterns efficiently, while 'sigmoid' is suitable for binary output.
-  The achieved accuracy was approximately 73%, which is slightly below the target performance of 75%.

### 3) Model optimization
##### Optimization 1:
- To enhance the neural network model's performance, I employed Keras Tuner for hyperparameter optimization, allowing it to dynamically select the optimal number of neurons in each layer, the number of hidden layers, and the activation functions from a set of ['relu', 'tanh', 'sigmoid', 'leaky_relu']. Despite these efforts, the model achieved an accuracy of approximately 73.19%, falling slightly short of the 75% target. This optimization process involved evaluating various configurations over multiple epochs to find the best combination of parameters for improving accuracy.

##### Optimization 2:
- I optimized the model by adjusting the training and testing split ratio to allocate 80% of the data for training and 20% for testing, aiming to provide the model with more data to learn from. This adjustment, combined with maintaining the same neural network architecture as the previous iterations—two hidden layers with 80 and 30 neurons respectively, and 'relu' activation for hidden layers and 'sigmoid' for the output layer—resulted in a slight improvement, achieving an accuracy of approximately 73.66%.

##### Optimization 3:
- In this attempt to achieve a 75% accuracy target, the test size was further reduced to 15%, giving the model more training data. Additionally, I introduced an extra hidden layer with 60 neurons into the neural network, resulting in a three-layer architecture (with 80, 60, and 30 neurons in each consecutive layer), all using 'relu' activation functions, and a 'sigmoid' function for the output layer. Despite these changes, the model reached an accuracy of approximately 73.29%, indicating a marginal increase in performance but still below the desired 75% threshold.

# Summary
Throughout the optimization process of the deep learning model, various strategies were employed, including adjusting the training-test split, experimenting with hyperparameter tuning through Keras Tuner, and modifying the neural network architecture by adding more layers and changing neuron counts. Despite these efforts to enhance model performance, the highest accuracy achieved was approximately 73.66%, falling short of the targeted 75%. This suggests that while the adjustments made incremental improvements, further exploration of model parameters, data preprocessing, or alternative modeling techniques might be necessary to reach the desired performance level.
