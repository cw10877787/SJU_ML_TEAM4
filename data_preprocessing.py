import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

class PreProcessData:
    """This class loads the data, contains the functions to clean it, and get it prepped for modeling
    Specifics include:
    1. leadData: Load data from a CSV
    2. describeData: Describe the dataset using info, describe, and outputs histograms, boxplots, and barplots
    3. logTransformData: Transforms numerical columns using a log transform
    4. Bivariate analysis through pairplots and correlation heatmaps
    6. Computes VIF scores to check for multicollinearity
    7. Encodes categorical variables using one-hot encoding
    8. Scales numerical data
    9. Addresses severe class imbalance using chosen method (oversampling, undersampling, or SMOTE    
    """

    def __init__(self, scaleType: str ='minmax'):
        """Initialize the PreProcessData class.
        
        Inputs:
        1. scaleType (str)      : The type of scaling to be applied to numerical data. 
           a. "Standardize"     : StandardScaler.
           b. "Normalize"       : MinMaxScaler.
        """

        self.scaleType = scaleType
        self.scaler = None # This will set based on the scale_type

        if self.scaleType == 'standardize':
            self.scaler = StandardScaler()
        # CONSTRAINT: WE CAN'T USE MINMAXSCALER
        elif self.scaleType == 'normalize':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scale_type must be either 'standardize' or 'normalize'.")
        
        self.data = None # While initializing the class, we don't have the data yet.
        self.numerical_imputer = None # This will be set in the impute_data method.
        self.categorical_imputer = None # This will be set in the impute_data method.
        self.outlier_method = None # This will be set in the detect_outliers method.
    
    def loadData(self, filePath1: str, filePath2: str, fileFormat: str='csv', **kwargs):
        """Load data from CSV files

        #Import Data and combine into DataFrame
        t1 = pd.read_csv("./transactions-1.csv")
        t2 = pd.read_csv("./transactions-2.csv")
        df = pd.concat([t1, t2], ignore_index=True)
        df.head()

        Inputs:
        1. filePath (str)       : Path to the file.
        2. fileFormat (str)     : Format of the file. Options are 'csv' or 'excel'.
        3. **kwargs             : Additional arguments for pd.read_csv or pd.read_excel.
        
        Returns:
        1. data (DataFrame)     : Loaded data as a pandas DataFrame.
        """
        
        t1 = pd.read_csv(filePath1, **kwargs)
        t2 = pd.read_csv(filePath2, **kwargs)
        self.data = pd.concat([t1, t2], ignore_index=True)
        self.data.head()
        print(f"Data loaded successfully")
    
    def describeData(self, displayRows: int = 5):
        """Display basic information about the loaded data.
        
            Inputs:
            1. displayRows (int)    : Number of rows to display. Defaults to 5.
            1. First few rows of the data. Defaults to 5 rows.
            2. Missing Value Counts
            2. Histograms
            3. Boxplots for numerical variables
            4. Barplots for categorical variables.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        print("-------------------- Data Overview --------------------")
        print("-------------------- Data Snapshot --------------------")
        print(self.data.head(displayRows))
        print("\n-------------------- Data Info --------------------")
        print(self.data.info())
        print("\n-------------------- Missing Values Counts --------------------")
        print(self.data.isnull().sum())

        print("\n-------------------- Histograms --------------------")
        self.data.hist(bins=50, figsize=(16,12))
        
        print("\n-------------------- Boxplots --------------------")
        self.data.plot(kind="box", subplots=True, layout=(4,4), figsize=(16,12), sharex=False, sharey=False)

        # Prep Categorical Features
        df_categorical_features = self.data.select_dtypes(include=['object'])

        print("\n-------------------- Barplots --------------------")
        for i in range(len(df_categorical_features.columns)):
            sns.catplot(data=df_categorical_features, y=df_categorical_features.columns[i], kind='count')
            plt.show()


    def logTransformData(self, columnName: str):
        """Transform the selected column using the logTransform method.
        
        Inputs:
        1. the column name. List of columns to be transformed
        2. Error handles for non-numerical columns or negative values
        """
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        #If column contains a null value, error
        if self.data[columnName].isnull().any():
            raise ValueError("Column contains null values.")
        #If column is not numeric, error
        elif self.data[columnName].dtype != 'float64' and self.data[columnName].dtype != 'int64':
            raise ValueError("Column must be numerical")
        #If columnn contains a negative value, error
        elif self.data[columnName].lt(0).any():
            raise ValueError("Log transform can't take a negative value. ")
    
        new_column = f"{columnName}_log"
        self.data[new_column] = np.log(self.data[columnName])
        print("Tranformed colum using log transformation")
        
        # Plot historgram of column
        self.data[new_column].hist()
        plt.show()
    
    def bivariateAnalysis(self):
        """Perform bivariate analysis using pairplots and correlation heatmaps.
        
        Inputs:
        1. Pairplots for numerical variables
        2. Correlation heatmap for numerical variables
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        numericalColumns = self.data.select_dtypes(include=['float64', 'int64']).columns
        # Pairplots for numerical variables
        sns.pairplot(self.data[numericalColumns])
        plt.show()

        # Correlation Matrix for numerical variables
        fig = plt.figure(figsize=(12, 8))
        corrMatrix = self.data[numericalColumns].corr()
        mask = np.triu(np.ones_like(corrMatrix, dtype=bool)) 
        sns.heatmap(corrMatrix, annot=True, cmap='viridis', center=0, fmt='.2f', mask=mask, annot_kws={"size": 8})
        plt.show()

        correlation_matrix = self.data[numericalColumns].corr()
        np.round(correlation_matrix['is_fraud'].sort_values(ascending=False), 4)
    
    def scaleData(self):
        """Scale numerical data using the specified scaler.
        
        Inputs:
        1. scaleType (str)      : The type of scaling to be applied to numerical data. 
           a. "Standardize"     : StandardScaler.
           b. "Normalize"       : MinMaxScaler.
        """
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        # Separate numerical columns
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        
        # Scale numerical data
        self.data[numerical_columns] = self.scaler.fit_transform(self.data[numerical_columns])
        print("Data scaled successfully using {self.scaleType} method.")
    



    def encodeCategoricalData(self, prefix:str = "cat"):
        """Create dummy variables for categorical variables with a user-chosen prefix.
        
        Inputs:
        1. prefix (str)        : Prefix for the dummy variable columns.
        
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        # Separate categorical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if len(categorical_columns) == 0:
            print("No categorical columns found to encode.")
            return
        else:
            # Create dummy variables
            self.data = pd.get_dummies(self.data, prefix=prefix, columns=categorical_columns, drop_first=True)
            print("Categorical data encoded successfully using prefix '{prefix}'.")
