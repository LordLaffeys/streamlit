import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('RF_class.pkl', 'rb'))

def main():
    # title
    st.title('Customer Churn Prediciton Web')

    #input interface
    CreditScore = st.text_input('Credit Score')
    Age = st.text_input('Age')
    Tenure = st.slider(
    'Tenure',
    0, 30)

    Balance = st.text_input('Balance')
    NumOfProducts = st.text_input('Number of Products')

    HasCrCard = 0
    option_HasCrCard = st.selectbox(
    'Has Credit Card?',
    ('No', 'Yes'))
    if option_HasCrCard == 'No':
        HasCrCard=0
    elif option_HasCrCard=='Yes':
        HasCrCard=1

    IsActiveMember = 0
    option_activeMember = st.selectbox(
    'Active Member?',
    ('No', 'Yes'))
    if option_activeMember == 'No':
        IsActiveMember=0
    elif option_activeMember=='Yes':
        IsActiveMember=1

    EstimatedSalary = st.text_input('Estimated Salary')

    Geography_France = 1
    Geography_Germany = 0
    Geography_Spain = 0
    option_geography = st.selectbox(
    'Select Geography',
    ('France', 'Germany', 'Spain'))
    if option_geography == 'France':
        Geography_France=1
    elif option_geography=='Germany':
        Geography_Germany=1
    elif option_geography=='Spain':
        Geography_Spain=1

    # Geography_France = st.text_input('Geography (France)')
    # Geography_Germany = st.text_input('Geography (Germany)')
    # Geography_Spain = st.text_input('Geography (Spain)')

    Gender_Female = 1
    Gender_Male = 0
    option_gender = st.selectbox(
    'Select Gender',
    ('Female', 'Male'))
    if option_gender == 'Female':
        Gender_Female=0
    elif option_gender =='Male':
        Gender_Male=1

    # Gender_Female = st.text_input('Gender (Female)')
    # Gender_Male = st.text_input('Gender (Male)')
    
    # code prediction
    prediction = ''

    # button for predict
    if st.button('Churn Predict Result'):
        prediction = make_prediction([CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography_France,Geography_Germany,Geography_Spain,Gender_Female,Gender_Male])

    st.success(prediction)




def make_prediction(input_data):

    input_data_asnumpy = np.array(input_data)

    input_datareshaped = input_data_asnumpy.reshape(1, -1)
    prediction = loaded_model.predict(input_datareshaped)

    print('prediction: ', prediction)

    if prediction == 0:
        return 'This customer did not churn'
    else:
        return 'This customer churn'

if __name__ == '__main__':
    main()
