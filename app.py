import streamlit as st
import pickle
import pandas as pd

def main():
    style = """<div style='background-color:blue; padding:12px'>
              <h1 style='color:black'>Rental Price Prediction App</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))

    # feature specifications
    completion_year = st.number_input('Enter the completion year of building', min_value=1977, max_value=2025, step=1, format="%d", value=2023)
    rooms = st.number_input('Enter the number of room', min_value=1, max_value=10, step=1, format="%d", value=1)
    parking = st.number_input('Enter the number of parking available', min_value=0, max_value=10, step=1, format="%d", value=1)
    bathroom = st.number_input('Enter the number of bathroom', min_value=1, max_value=10, step=1, format="%d", value=1)
    size = st.number_input('Enter the size of the property in sq.ft.', min_value=100.0, max_value=3000.0, step=1.0, format="%f", value=800.0)
    AirCond = st.radio('Air-Cond', ('Y', 'N'), index=0)
    WashingMachine = st.radio('Washing Machine', ('Y', 'N'), index=0)
    NearKTMLRT = st.radio('Near KTM or LRT', ('Y', 'N'), index=0)
    Internet = st.radio('Internet', ('Y', 'N'), index=0)
    CookingAllowed = st.radio('Cooking Allowed', ('Y', 'N'), index=0)
    property_type = st.selectbox('Select the type of property', ('Condominium', 'Service Residence', 'Apartment', 'Flat', 'Studio', 'Others', 'Duplex', 'Townhouse Condo'))
    furnished = st.selectbox('Select the furnish status of property', ('Fully Furnished', 'Not Furnished', 'Partially Furnished'))
    region = st.selectbox('Located at KL or Selangor', ('Kuala Lumpur', 'Selangor'))  
    
    # button
    button = st.button('Predict')
    
    # if button is pressed
    if button:
        # make prediction
        result = predict(completion_year, rooms, parking, bathroom, size, property_type, furnished, AirCond, WashingMachine, NearKTMLRT, Internet, CookingAllowed, region)
        st.success(f'The estimate rental price is RM {result}')
        
# load the train model
with open('rf_model.pkl', 'rb') as rf:
    model = pickle.load(rf)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)


def predict(completion_year, rooms, parking, bathroom, size, property_type, furnished, AirCond, WashingMachine, NearKTMLRT, Internet, CookingAllowed, region):
    # processing user input
    property_type = 0 if property_type == 'Apartment' else 1 if property_type == 'Condominium' else 2 if property_type == 'Duplex' else 3 if property_type == 'Flat' else 4 if property_type == 'Others' else 5 if property_type == 'Service Residence' else 6 if property_type == 'Studio' else 7
    furnished = 0 if furnished == 'Fully Furnished' else 1 if furnished == 'Not Furnished' else 2
    AirCond = True if AirCond == 'TRUE' else False
    WashingMachine = True if WashingMachine == 'TRUE' else False
    NearKTMLRT = True if NearKTMLRT == 'TRUE' else False
    Internet = True if Internet == 'TRUE' else False
    CookingAllowed = True if CookingAllowed == 'TRUE' else False
    region = 0 if region == 'Kuala Lumpur' else 1 
    
    lists = [completion_year, rooms, parking, bathroom, size, property_type, furnished, AirCond, WashingMachine, NearKTMLRT, Internet, CookingAllowed, region]
    df = pd.DataFrame(lists).transpose()

    # scaling the data
    scaler.transform(df)
    
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result
if __name__ == '__main__':
    main()
