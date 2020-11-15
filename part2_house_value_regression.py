import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from pickle import dump, load
        

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
    
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.fillval = {}
        
        
        X, _ = self._preprocessor(x, training = True)
        self.X = X
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        
        #ADJUST Y NONE
        
        #cont_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']
        column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income','ocean_proximity'] 
        numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']
        categorical_features = ['ocean_proximity']
        
        features = x[column_names]
        
        #Potentially do this with model saving
        #If we've seen no data before fit preprocessors
        if(training):
            
            numeric_transformer = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps = [
                ('ordinal', OrdinalEncoder()),
                ('scaler',StandardScaler())
            ])
            
            ct = ColumnTransformer(
                transformers = [
                    ('num', numeric_transformer, numeric_features),
                    ('cat',categorical_transformer,categorical_features)
                ])
            
            df_processed = ct.fit_transform(X = features)
            
            #Save Transfomer"
            dump(ct, open("x_transformer.pkl","wb"))
            
            #Transform y
            if y is not None:
                y_scaler = StandardScaler()
                processed_y = y_scaler.fit_transform(y)
                dump(y_scaler, open("y_transformer.pkl","wb"))

        #If we've seen data before transform with saved preprocessors
        else:
            
            #Load Column Transformer and Transform data
            ct = load(open('x_transformer.pkl', 'rb'))
            df_processed = ct.transform(features)
               
            #Transform y
            if y is not None:
                y_scaler = load(open('y_transformer.pkl', 'rb'))
                processed_y = y_scaler.transform(y)


                
        #Transform to Tensors
        x_tensor = torch.tensor(df_processed,dtype = torch.float32)
        
        if y is not None:
            y_tensor = torch.tensor(processed_y,dtype = torch.float32)
        
        
        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

