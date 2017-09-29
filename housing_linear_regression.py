
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model    
from sklearn.model_selection import train_test_split

def main(): 
#      load the housing data in pandas dateframe
    df_housing = pd.read_csv("housing.csv")
    
#     feature
    X = df_housing['lotsize']
    
#     target
    y = df_housing['price']

#     new shape to an array without changing its data
    X=X.values.reshape(len(X),1)        
    y=y.values.reshape(len(y),1)

#     split the data in train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
     
#     plot the test data
    plt.scatter(X_test, y_test,  color='black')
    plt.title('Test Housing Price Data')
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.xticks(())
    plt.yticks(()) 
#     plt.show()
    
#     create linear regression object model
    regression_model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=False, n_jobs=1)
    
#     train the model using the training sets
    regression_model.fit(X_train, y_train)
    
#     get y predicted values
    y_predicted = regression_model.predict(X_test)
    
#     plot the linear model
    plt.plot(X_test, y_predicted, color='red', linewidth=3)
    plt.show()
    
    size = 5000
    price_predicted = regression_model.predict(size)
    
    print('House Size: ', str(size))
    print('House Predicted Price: ', str(price_predicted))

if __name__ == '__main__':
    main()