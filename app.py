#streamlit UI
import pickle
import pandas as pd
import streamlit as st

#page configuration
st.set_page_config(page_title="Loan Approval Classification",
                   page_icon="logo.png",layout="wide")

#sidebar
with st.sidebar:
    st.title("Loan Approval Classification")
    st.image("logo.png")

#load the model
with open("best_model_pipeline.pkl","rb") as file:
    best_pipeline = pickle.load(file)

#load cleaned dataset
df = pd.read_csv("cleaned_data.csv")


#user input
col1,col2,col3,col4 = st.columns(4,border=True)

with col1:
    person_age= st.slider("Age: ",min_value=1,max_value=100)

    person_income = st.slider("Income: ",
                          min_value=10000,
                          max_value=10000000)
    person_emp_exp = st.slider("Experience: ",
                           min_value=0,
                           max_value=50)
    loan_amnt = st.slider("Loan amount: ",
                      min_value=30000,
                      max_value=10000000)

with col2:
    person_gender = st.radio("Gender: ",
                            options=df["person_gender"].unique(),
                            horizontal=True)
    loan_int_rate = st.slider("Loan Interest rate: ",
                          min_value=5.0,
                          max_value=20.0)
    loan_percent_income = st.slider("Loan Percent Income: ",
                                min_value=0.0,
                                max_value=1.0)
    cb_person_cred_hist_length = st.slider("Credit Hist length: ",
                                       min_value=1.0,
                                       max_value=30.0)
    credit_score = st.slider("Credit Score: ",
                         min_value=300,
                         max_value=900)

with col3:
    previous_loan_defaults_on_file = st.selectbox("Loan Defaults: ",
                                                options=df["previous_loan_defaults_on_file"].unique())
    loan_intent = st.selectbox("Loan Intent: ",
                            options=df["loan_intent"].unique())
    person_home_ownership = st.selectbox("Home Ownership: ",
                                        options=df["person_home_ownership"].unique())
    person_education = st.selectbox("Education: ",
                                    options=df["person_education"].unique())


#new data
new_data = {'person_age':person_age, 'person_gender':person_gender, 'person_education':person_education, 'person_income':person_income,
       'person_emp_exp':person_emp_exp, 'person_home_ownership':person_home_ownership, 'loan_amnt':loan_amnt, 'loan_intent':loan_intent,
       'loan_int_rate':loan_int_rate, 'loan_percent_income':loan_percent_income, 'cb_person_cred_hist_length':cb_person_cred_hist_length,
       'credit_score':credit_score, 'previous_loan_defaults_on_file':previous_loan_defaults_on_file}

new_data = pd.DataFrame(new_data,index=[0])

#predictions
with col4:
    if st.button("Predict Loan staus"):
        prediction = best_pipeline.predict(new_data)[0]
        if prediction==0:
            st.error("Loan Rejected")
        else:
            st.success("Loan Approved")
          
        
