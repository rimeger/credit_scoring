import requests
import streamlit as st

st.title("Premium Credit Card")
st.write("New credit card with immidiate approval")

with st.form("Send Request"):
    age = st.number_input("Age", min_value=18)
    income = st.number_input("Income", min_value=0)
    education = st.checkbox("College degree")
    work = st.checkbox("Stable income")
    car = st.checkbox("Car")
    submit = st.form_submit_button("Send Request")

if submit:
    data = {"age": age, 
            "income": income, 
            "education": education, 
            "work": work,
            "car": car}
    
    response = requests.post("http://service:7000/score", json=data)
    if response.json()["approved"]:
        st.success("Congratulations! Your credit card is approved.")
    else:
        st.warning("We can offer you debit card with 3% cashback on all purchases.")