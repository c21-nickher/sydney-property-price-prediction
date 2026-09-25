This project focuses on predicting residential property sale prices across three Sydney suburbs: Mosman, Parramatta, and Campbelltown. The project uses machine learning techniques to analyse property characteristics and estimate sale prices.

The dataset contains 120 sold properties, with 40 properties collected from each suburb. The data includes information such as suburb, property type, bedrooms, bathrooms, car spaces, land size, sale date, and sale price.

The project follows a complete machine learning workflow including data collection, data cleaning, exploratory data analysis, feature engineering, model development, model evaluation, prediction error analysis, and deployment.

During the exploratory data analysis, property prices were compared across the three suburbs. Relationships between sale price and features such as bedrooms, bathrooms, land size, and sale date were also investigated. Missing values, correlations, price distributions, and potential outliers were examined.

Feature engineering was performed to prepare the data for machine learning. This included handling missing land size and car space values, grouping rare property types, creating total room count, extracting the sale month, and calculating the number of days since the first recorded sale.

Three machine learning models were compared: Ridge Regression, Random Forest Regression, and Gradient Boosting Regression. The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² score, and 5-fold cross-validation.

Random Forest Regression was selected as the final model and was used for deployment. The trained model and its preprocessing pipeline are stored in the model.pkl file.

A Streamlit web application was developed to allow users to enter property characteristics such as suburb, property type, bedrooms, bathrooms, car spaces, land size, and sale date. The application uses the trained Random Forest model to provide an estimated property sale price.

The project was developed using Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Pickle, and Jupyter Notebook.

To run the application, install the required Python packages and run:

pip install streamlit scikit-learn pandas numpy

Then navigate to the folder containing app.py and model.pkl and run:

streamlit run app.py
