import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
import PIL
from PIL import Image

# -------------------------------This is the configuration page for our Streamlit Application---------------------------

st.set_page_config(page_title="Industrial Copper Modeling Application",page_icon="🏨",
                   layout="wide",
                   )
# -------------------------------This is the main tabs in a Streamlit application, helps in navigation--------------------

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)
tab1,tab2,tab3 = st.tabs(["ABOUT","PREDICT SELLING PRICE", "PREDICT STATUS"]) 

# -----------------------------------------------About Project Section--------------------------------------------------

with tab1:
    im1 = Image.open("indus.jpg")
    st.image(im1, width=1500)

    col1, col2 = st.columns(2)

    with col1:
        im = Image.open("copper_info.png")
        st.markdown("")
        st.image(im)

    with col2:

        st.markdown("For many years, copper has been used in industry. Besides being essential in all electrical installations, typical applications are for the manufacture of pressure vessels, distillation equipment, piping systems and gaskets.")
        st.markdown("According to the Copper Development Association (CDA) there are four different areas of industry where copper is utilized:")
        st.markdown("Electrical: 65%,Construction: 25%],Transport: 7%,Other: 3%")


        st.markdown("As you can see the most common way we use copper today is in electrical applications, but there are many other important uses as well.")
        st.header(":rainbow[Electrical Copper]",divider=True)
       
        st.markdown("Copper is used in virtually all electrical wiring (except for power lines, which are made with aluminum) because it is the second most electrically conductive metal aside from silver which is much more expensive. In addition to being widely available and inexpensive, it is malleable and easy to stretch out into very thin, flexible but strong wires, making it ideal to use in electrical infrastructure.")
        st.markdown("Aside from electrical wiring, copper is also used in heating elements, motors, renewable energy, internet lines, and electronics.")
        
        st.header(":green[Construction: 25%]",divider=True)        
        st.markdown("Copper has been used as construction material for centuries. It develops a characteristic beautiful green patina, or verdigris, that was highly desired in certain architectural styles, and still is to this day. Copper is still used today in architecture due to its corrosion resistance, easy workability, and attractiveness; copper sheets make a beautiful roofing material and other exterior features on buildings.")
        st.markdown("On the interior, copper is used in door handles, trim, vents, railings, kitchen appliances and cookware, lighting fixtures, and more.")
        st.markdown("Because copper has antimicrobial and antiviral properties (it’s ability to inhibit the growth of bacteria, viruses, and other pathogens), copper has become a standard for potable water piping in the developed world. Other properties that make copper ideal for piping systems include its malleability and resistance to heat and corrosion. It’s commonly used in distillation, pharmaceutical production, and other highly specialized applications.")

        st.header(":blue[Use of Copper in Transportation]",divider=True)
        
        st.markdown("Aside from the copper wiring used in the electrical components of modern cars, copper and brass have been the industry standard for oil coolers and radiators since the 1970s. Alloys that include copper are used in the locomotive and aerospace industries as well. As demand for electric cars and other forms of transportation increases, demand for copper components also increases.")
        st.header(":red[Other Copper Uses]",divider=True)
        
        st.markdown("Because copper is a beautiful, easily worked material, it is used in art such as copper sheet metal sculptures, jewelry, signage, musical instruments, cookware, and more")

# ------------------------------------------------Selling a Predictions Price---------------------------------------------------

with tab2:    
        

        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                     '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                     '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                     '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                     '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options,key=1)
                item_type = st.selectbox("Item Type", item_type_options,key=2)
                country = st.selectbox("Country", sorted(country_options),key=3)
                application = st.selectbox("Application", sorted(application_options),key=4)
                product_ref = st.selectbox("Product Reference", product,key=5)
            with col3:               
                st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
    
            flag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons,thickness,width,customer]:             
                if re.match(pattern, i):
                    pass
                else:                    
                    flag=1  
                    break
            
        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)  
             
        if submit_button and flag==0:
            
            import pickle
            with open(r"model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
# -----------------------------------------------Predictions Status---------------------------------------------------
            
with tab3: 
    
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options,key=21)
                ccountry = st.selectbox("Country", sorted(country_options),key=31)
                capplication = st.selectbox("Application", sorted(application_options),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
            import pickle
            with open(r"cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred==1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
                
st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by kani</h6>', unsafe_allow_html=True )  
