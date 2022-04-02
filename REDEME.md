# Deep-Learning
## Overview
Alphabet Soup is a non-profit foundation and they are trying to create an algorithm to predict whether applicants for funding will be used successfully. Since ether response variables in this case is a categorical variables, a classification algorithm will be used. Specially we are using **Neural Network** for this study.  Neural networks is a series of algorithm that finding the relationship underlying a set of data with a process that mimic the way human brain operates. 
## Results
  - #### Data Preprocessing
    -   First identify the target variables. Based on the senario and the result we are predicting, the target variable in thie dataset is **"IS_SUCCESSFUL"** and the feature variables will be the rest of the variables.
    -   Drop columns that won't be used during the process (**'EIN' and 'NAME'**)
    -   Determine the number of unique values in each column
    ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%2011.36.04%20AM.png)
    -   Based on the number from last step, binng the column with unique number over 10. Based on the number, we are binning column **APPLICATION_TYPE** and **CLASSIFICATION** with chosen cutoff value.
        ```python
        # binning APPLICATION_TYPE
        application_types_to_replace = list(countType[countType < 700].index)

        # Replace in dataframe
        for app in application_types_to_replace:
            application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app,"Other")

        # Check to make sure binning was successful
        application_df['APPLICATION_TYPE'].value_counts()
        ```
        
        ```python
        # binning CLASSIFICATION
        classifications_to_replace = list(countClass[countClass < 300].index)

        # Replace in dataframe
        for cls in classifications_to_replace:
            application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(cls,"Other")

        # Check to make sure binning was successful
        application_df['CLASSIFICATION'].value_counts()
        ```
        **Before binning vs. After binning**
        <p float="center">
          <img src = "https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%2011.36.15%20AM.png",
           width="450" />
          <img src = "https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%2011.36.28%20AM.png",
           width="450"  >
        </p>
        
        <p float="center">
          <img src = "https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%2011.38.01%20AM.png",
           width="450" />
          <img src = "https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%2011.38.11%20AM.png",
           width="450"  >
        </p>
        
    - Convert categorical data to numeric 
      ```python
      application_df = pd.get_dummies(application_df, 
                       columns=['APPLICATION_TYPE', 
                       'CLASSIFICATION','AFFILIATION','USE_CASE','ORGANIZATION','INCOME_AMT','SPECIAL_CONSIDERATIONS'],
                       drop_first=True, dtype=float)
      ```
  - #### Compiling, Training, and Evaluating the Model
    - Define the model, first model we are using two hidden layers and first layers with h7 nerons and second layers with 14 nurons
      ```python
      nn = tf.keras.models.Sequential()

      # First hidden layer
      nn.add(tf.keras.layers.Dense(units=7, activation="relu", input_dim=36))

      # Second hidden layer
      nn.add(tf.keras.layers.Dense(units=14, input_dim=31,activation="relu"))

      # Output layer
      nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
      ```
      
      #### Model summary and accuracy rate:
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.20.55%20PM.png)
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.21.17%20PM.png)
    - In order to increase the model performance, we change the activate funciton from **"relu"** to **"sigmoid"**
      ```python
      nn2 = tf.keras.models.Sequential()

      # First hidden layer
      nn2.add(tf.keras.layers.Dense(units=14, activation="sigmoid", input_dim=36))

      # Second hidden layer
      nn2.add(tf.keras.layers.Dense(units=21, input_dim=31,activation="sigmoid"))

      # Thirdidden layer
      nn2.add(tf.keras.layers.Dense(units=14, input_dim=31,activation="sigmoid"))

      # Output layer
      nn2.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
      ```
      
      #### Model summary and accuracy rate:
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.26.20%20PM.png)
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.26.34%20PM.png)
    - Another way to improve model performance is to increase the number of hiddern layers and neurons
      ```python
      nn3 = tf.keras.models.Sequential()

      # First hidden layer
      nn3.add(tf.keras.layers.Dense(units=14, activation="sigmoid", input_dim=36))

      # Second hidden layer
      nn3.add(tf.keras.layers.Dense(units=14, input_dim=31,activation="sigmoid"))

      # Third hidden layer
      nn3.add(tf.keras.layers.Dense(units=14, input_dim=31,activation="sigmoid"))

      # Output layer
      nn3.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
      ```
      #### Model summary and accuracy rate:
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.26.47%20PM.png)
      ![image](https://github.com/ludanzhan/Deep-Learning/blob/main/Resources/Screen%20Shot%202022-04-02%20at%202.26.56%20PM.png)
  - #### Summary
    Inreasing number of hidden layers, change activate function, and increase number of neurons are efficient ways to improve the model performance. By applying those methods, we increase the model accuracy rate and decrease the loss. Although not meet the expected accuracy rate (75%), still proved those methods works.

