import streamlit as st
import pandas as pd
import numpy as np
st.title("Hello, Streamlit")
st.write("This is my first Streamlit app")
st.text("lets get started")

name=st.text_input("enter your name:")
if st.button("greet"):
    st.success(f"hello,{name}!")
df=pd.DataFrame(np.random.randn(10,2),columns=['A','B'])
st.line_chart(df)
st.bar_chart(df)

upload_file=st.file_uploader("upload a csv",type='csv')
if upload_file:
    df=pd.read_csv(upload_file)
    st.dataframe(df)
st.title("Text and Markdowm Demo")
st.header("This a Header")
st.subheader("this is a subheader")
st.markdown("**Bold**,*Italic*,'code',[link](https://streamlit.io)")
st.code("for i in range(5):print(i)",language="python")
st.code("for i in range(5): print(i)", language="python")

st.text_input("What's your name?")
st.text_area("Write something...")
st.number_input("Pick a number", min_value=0, max_value=100)
st.slider("Choose a range", 0, 100)
st.selectbox("Select a fruit", ["Apple", "Banana", "Mango"])
st.multiselect("Choose toppings", ["Cheese", "Tomato", "Olives"])
st.radio("Pick one", ["Option A", "Option B"])
st.checkbox("I agree to the terms")    
if st.checkbox("show details"):
    st.info("here are your details")
    option=st.radio("choose view",["show chart","show table"])
if option=="show chart":
    st.write("chart would appear here")
else:
    st.write("table would appear here")    
with st.form("login_form"):
    username=st.text_input("username")
    password=st.text_input("password")
    submitted=st.form_submit_button("Login")
    if submitted:
        st.success(f"welcome,{username}!")
st.image("https://www.google.com/imgres?q=images&imgurl=https%3A%2F%2Fpicjumbo.com%2Fwp-content%2Fuploads%2Fbeautiful-nature-wallpaper-spring-flowers-mountain-sunset-free-image.jpg&imgrefurl=https%3A%2F%2Fpicjumbo.com%2Ffree-photos%2Fpure-nature%2F&docid=NJ6j8Zr_xdbJRM&tbnid=rlZ6mczC1gXPCM&vet=12ahUKEwiEv6PTsdyXAxUl4TgGHaZHLiwQnPAOegQIRhAA..i&w=5376&h=3584&hcb=2&ved=2ahUKEwiEv6PTsdyXAxUl4TgGHaZHLiwQnPAOegQIRhAA",
         caption="sample image",use_column_width=True)
st.video=("https://www.youtube.com/watch?v=wGOn7hDb3gk")        
