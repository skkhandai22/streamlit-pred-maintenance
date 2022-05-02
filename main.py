import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import plotly.express as px

pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.set_page_config(page_title='Predictive Maintenance Explorer',page_icon="logo.png",layout='wide',initial_sidebar_state='auto',)
def pred(choice,air_temp,process_temp,rot_speed,torque,tool_wear):
    if choice == "Low":
        type = 1
    elif choice == "Medium":
        type = 2
    else:
        type = 3

    prediction = classifier.predict(
        [[type,air_temp,process_temp,rot_speed,torque,tool_wear]])



    if prediction == 1:
        pred = 'has No Failure'
    elif prediction ==2:
        pred = 'has Heat Dissipation Failure'
    elif prediction ==3:
        pred = 'has Overstrain Failure'
    elif prediction ==4:
        pred = 'has Power Failure'
    elif prediction == 5:
        pred = ' may be has Random Failure'
    elif prediction == 6:
        pred = 'has Tool wear Failure'
    else:
        pred = "Unknown Failure"

    return pred

if __name__ == '__main__':
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 6, 1])
    st.markdown("""
    <nav class ="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FF4B$B;">
    """,unsafe_allow_html=True)
    with col1:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)
    st.sidebar.title('''**Maintenance Explorer**''')
    st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"]
    > div:first-child {width: 450px;}[data-testid="stSidebar"][aria-expanded="false"]
    > div:first-child {width: 450px;margin-left: -400px;}</style>""",
    unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader("Upload Data File", type=['csv'], accept_multiple_files=False)

    if uploaded_files:
        selected = option_menu(
            menu_title="",
            options=["Home", "Visualization"],
            icons=["house", "clipboard-data"],
            orientation="horizontal"
        )
        if selected == "Home":
            st.title("Enter the details to check Prediction")


            choice = st.selectbox("Type of Product",("Low","Medium","High"),help="low (50% of all products), medium (30%), and high (20%) as product quality variants" )
            air_temp = st.number_input("Enter Air Temperture(K)",help="Air temperature [K] generated using a random walk process later normalized to a standard deviation of 2 K around 300 K")
            process_temp = st.number_input("Enter Process Temperture(K)",help="Process temperature [K] generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.")
            rot_speed = st.number_input("Enter Rotational Speed(RPM)", help="Rotational speed [rpm] calculated from powepower of 2860 W, overlaid with a normally distributed noise")
            torque= st.number_input("Enter Torque(Nm)",help="Torque [Nm] values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.")
            tool_wear = st.number_input("Enter tool wear(Min)",help="The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process")
            result = ""

            if st.button("Predict"):
                result = pred(choice,air_temp,process_temp,rot_speed,torque,tool_wear)
                st.success('Machine {}'.format(result))

        if selected == "Visualization":
            data=pd.read_csv(uploaded_files)
            st.info("Failure Type CountPlot with Target as Failure")
            fig1=plt.figure(figsize=(8, 2))
            sns.countplot(data=data[data['Target'] == 1], x="Failure Type")
            st.pyplot(fig1)

            st.info("Failure Type  Pie Chart")
            fig2 = px.pie(data, title='Failure Types', names='Failure Type')
            st.plotly_chart(fig2)

            st.info("Data Distribution")
            fig3=plt.figure(figsize=(20, 15))
            m = 1
            for i in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                      'Tool wear [min]']:
                plt.subplot(3, 2, m)
                sns.boxplot(data=data, y=i, x="Type", hue="Target")
                m += 1

            st.pyplot(fig3)







