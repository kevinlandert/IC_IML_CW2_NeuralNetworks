import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,MinMaxScaler
from pickle import dump, load
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,explained_variance_score,r2_score

#from skorch import NeuralNetRegressor
#from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt
        

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


        X, _ = self._preprocessor(x, training = True)
        self.X = X
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.batch_size = 64
        self.running_loss = 0
        self.folds = 10
        
        #Networkk -> Adjust structure at the bottom
        self.network = Network(input_dim = self.input_size,output_dim = self.output_size)
        
        #Optimizer
        #self.optimizer = optim.SGD(self.network.parameters(), lr = 0.01)
        #self.optimizer = optim.Adagrad(self.network.parameters(), lr=0.01, lr_decay=0, weight_decay=0.01, initial_accumulator_value=0, eps=1e-10)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        
        
        
        
        
        
        
        
        
        

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
        outputs = y
        
        #Potentially do this with model saving
        #If we've seen no data before fit preprocessors
        if(training):
            
            numeric_transformer = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps = [
                ('ordinal', OneHotEncoder(sparse=False)),
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
            
            #Transform y -> is probably not necessary
            if y is not None:
                y_scaler = MinMaxScaler()
                outputs = y_scaler.fit_transform(outputs)
                dump(y_scaler, open("y_transformer.pkl","wb"))

        #If we've seen data before transform with saved preprocessors
        else:
            
            #Load Column Transformer and Transform data
            ct = load(open('x_transformer.pkl', 'rb'))
            df_processed = ct.transform(features)
               
            #Transform y
            if y is not None:
                y_scaler = load(open('y_transformer.pkl', 'rb'))
                outputs = y_scaler.transform(outputs)


                
        #Transform to Tensors
        x_tensor = torch.tensor(df_processed,dtype = torch.float32)
        
        if y is not None:
            #y_tensor = torch.tensor(outputs,dtype = torch.float32)
            y_tensor = torch.tensor(y.values,dtype = torch.float32)
        
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

        #Preprocess input data
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        #Saves losses
        #[0,fold,epoch] -> for training losses
        #[1,fold,epoch] -> for validation losses
        rel_losses = np.zeros((2,self.folds,self.nb_epoch))
        abs_losses= np.zeros((2,self.folds,self.nb_epoch))
        
        #10 fold crossvalidation with 9 for training 1 for validation
        kfold = KFold(n_splits = self.folds)
        folds = kfold.split(np.arange(X.shape[0]))
        
        for fold, (train_index,val_index) in enumerate(folds):

            #Data to train
            x_train = X[train_index].detach()
            y_train = Y[train_index].detach()
            
            #Data to evaluate
            x_val = X[val_index].detach()
            y_val = Y[val_index].detach()
            
            #To do batching on training data
            torch_dataset_train = data.TensorDataset(x_train,y_train)
            data_loader_train = data.DataLoader(dataset = torch_dataset_train,batch_size = self.batch_size,shuffle = True)
            
            #To do batching on validation data -> for later
            #torch_dataset_val = data.TensorDataset(x_val,y_val)
            #data_loader_val = data.DataLoader(dataset = torch_dataset_val,batch_size = self.batch_size, shuffle = False)
            
        
            for epoch in range(self.nb_epoch):
            
                
                #Set network to training mode
                self.network.train()
                
                #Batching in every epoch
                for step, (batch_x, batch_y) in enumerate(data_loader_train):
                
                
                    #Forward prop
                    prediction = self.network(batch_x)
                    
                    #Compute Loss

                    loss = nn.MSELoss()(prediction,batch_y)
                    
                    #Backward prop
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    #Update parameters
                    self.optimizer.step()
        
        
                #Evaluate after every epoch
                self.network.eval()
                
                #Compute total loss on training data
                prediction = self.network.forward(x_train).detach()
                train_loss_abs = nn.MSELoss()(prediction,y_train)
                abs_losses[0,fold,epoch] = train_loss_abs
                
                #Average relative distance from true value
                train_loss_rel = torch.sum(torch.div(torch.abs(torch.sub(prediction,y_train)),y_train)) / y_train.shape[0]
                rel_losses[0,fold,epoch] = train_loss_rel
                
                
                #Compute total loss on validation data
                prediction = self.network.forward(x_val).detach()
                val_loss_abs = nn.MSELoss()(prediction,y_val)
                abs_losses[1,fold,epoch] = val_loss_abs
                
                val_loss_rel = torch.sum(torch.div(torch.abs(torch.sub(prediction,y_val)),y_val)) / y_val.shape[0]
                rel_losses[1,fold,epoch] = val_loss_rel
                
                
                #Print losses every 20 folds               
                if ((epoch % 20) == 0) & (epoch != 0):
                    print("Fold: {}\t - Episode: {}\t - Training Loss:  {}\t - Validation Loss: {}".format(fold,epoch,train_loss_rel,val_loss_rel))
                
            #Just one fold for now 
            break
        
        self.loss_abs = abs_losses
        self.loss_rel = rel_losses
        
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

        #Preprocess data we want to predict
        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        #Return data we want to evaluate
        self.network.eval()
        prediction = self.network.forward(X).detach().numpy()
        
        
        return prediction
        
        
        
        

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
        
        Y = Y.numpy() #Output from predict is array
        prediction = self.predict(x)
        
        #Typical regression evaluation scores
        mse = mean_squared_error(prediction,Y)
        ex_var = explained_variance_score(prediction,Y)
        r2 = r2_score(prediction,Y)
        
        print(mse)
        print(ex_var)
        print(r2)
        
        
        
        
        
        
        
        
        
        
        
        return mean_squared_error(prediction,Y) # Replace this code with your own

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
    data = pd.read_csv("housing.csv") 
    
    #Randomly shuffle the data
    data = data.sample(frac = 1).reset_index(drop = True)
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    X, Y = self._preprocessor(x, y = y, training = False)
    
    """ Does not work because we cannot use scorch
    net = NeuralNetRegressor(Network
                         , max_epochs=100
                         , optimizer=optim.Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
                         , lr=0.001
                         , verbose=1)
    
    params = {
    'lr': [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_epochs': list(range(500,5500, 500))
    }
    
    gs = GridSearchCV(net, params, refit=False, scoring='r2', verbose=1, cv=10)
    
    gs.fit(X,Y)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    #Randomly shuffle the data
    data = data.sample(frac = 1).reset_index(drop = True)
    
    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    
    #Todo: Adjust with shuffling
    x_train = x[2000:]
    y_train = y[2000:]
    x_test = x_train[0:2000]
    y_test = y_train[:2000]
    
    
    regressor = Regressor(x_train, nb_epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    #Plots our training and validation loss
    plt.plot(np.arange(regressor.loss_rel.shape[2]),regressor.loss_rel[0,0,:],label = 'training_loss')
    plt.plot(np.arange(regressor.loss_rel.shape[2]),regressor.loss_rel[1,0,:],label = 'validation_loss')
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.savefig("loss.png")
    
    pred = regressor.predict(x_test)
    
    #scaler = load(open('y_transformer.pkl', 'rb'))
    #print(scaler.inverse_transform(pred))
    print(pred)
    print(y_test)
    

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


        
    
class Network(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        
        super(Network,self).__init__()
        
        self.layer_1 = nn.Linear(in_features = input_dim,out_features = 100)
        self.layer_2 = nn.Linear(in_features = 100, out_features = 100)
        self.layer_3 = nn.Linear(in_features = 100, out_features = 100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dim)
        
    def forward(self,input):
        
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output    
    

if __name__ == "__main__":
    
    
    example_main()
    

        
       
    