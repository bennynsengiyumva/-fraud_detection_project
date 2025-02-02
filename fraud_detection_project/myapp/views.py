from django.shortcuts import render

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


african_cities = ['Kigali', 'Kampala', 'Bujumbura', 'Nairobi', 'Harare', 'Addis Ababa', 'Dar es Salaam', 'Lagos', 'Accra', 'Dakar', 'Luanda', 'Johannesburg']


def is_fraud(row):
    if int(row ['Transaction_Amount']) > 800:
        return 1
    if row["User_Location"] not in african_cities or row ["Merchant_Location"] not in african_cities:
        return 1
    if int(row["User_Age"]) < 16:
        return 1
    return 0


def home(request):
    return render(request, 'myapp/home.html')

def about(request):
    return render(request, 'myapp/about.html')

def fraud(request):
    if request.method == "POST":
        rf_model = joblib.load('Big_data_fraud_detection_model.pkl')
        training_features = joblib.load('training_features.pkl')

        single_transaction = {
            'Transaction_Amount': request.POST.get('Transaction_Amount', 0),
            'Transaction_Time': request.POST.get('Transaction_Time', 0),
            'User_Location': request.POST.get('User_Location', ''),
            'Merchant_Location': request.POST.get('Merchant_Location', ''),
            'Device_Type': request.POST.get('Device_Type', ''),
            'User_Age': request.POST.get('User_Age', 0)
        }

        single_transaction_df = pd.DataFrame([single_transaction])


        df_encoded = pd.get_dummies(single_transaction_df, columns= ['User_Location', 'Merchant_Location', 'Device_Type'])

        # columns must match the training features
        for col in training_features:
            if col not in df_encoded.columns:
                df_encoded[col] = False

        df_encoded = df_encoded[training_features]

        scaler = MinMaxScaler()
        df_encoded['Transaction_Amount'] = scaler.fit_transform(df_encoded[['Transaction_Amount']])

        fraud_flag = is_fraud(single_transaction)


        predictions = rf_model.predict(df_encoded)

        return render(request, 'myapp/fraud_detection.html', {'success_message':'Fraud' if predictions[0] == 1 or fraud_flag == 1 else 'Not Fraud'})

    return render(request, 'myapp/fraud_detection.html')

