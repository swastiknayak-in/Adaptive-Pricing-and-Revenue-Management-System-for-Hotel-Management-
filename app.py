import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
from model_training import train_and_save, load_data

MODEL_PATH = "models/pricing_model.pkl"
DATA_PATH = "data/hotel_bookings.csv"

st.set_page_config(page_title="Adaptive Pricing & Revenue Management", layout="wide")

st.title("Adaptive Pricing and Revenue Management System for Hotel Management")

st.write("Customer behavior analysis and dynamic pricing dashboard")

# -------------------------------
# Train model automatically
# -------------------------------

if not os.path.exists(MODEL_PATH):

    with st.spinner("Training pricing model..."):

        train_and_save(DATA_PATH)

    st.success("Model trained successfully")


model = joblib.load(MODEL_PATH)

# -------------------------------
# Load dataset
# -------------------------------

df = load_data(DATA_PATH)

# -------------------------------
# Convert month
# -------------------------------

def month_to_num(month):

    months = {
        "January":1,"February":2,"March":3,"April":4,
        "May":5,"June":6,"July":7,"August":8,
        "September":9,"October":10,"November":11,"December":12
    }

    return months.get(month)


df["arrival_month_num"] = df["arrival_date_month"].map(month_to_num)

# -------------------------------
# Layout
# -------------------------------

left, right = st.columns([2,1])

# ===============================
# DASHBOARD
# ===============================

with left:

    st.subheader("Customer Behavior Dashboard")

    fig1 = px.histogram(df, x="lead_time", nbins=50, title="Lead Time Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="customer_type", title="Customer Type Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    room_counts = df["reserved_room_type"].value_counts().reset_index()
    room_counts.columns = ["room","count"]

    fig3 = px.bar(room_counts, x="room", y="count", title="Room Type Preference")
    st.plotly_chart(fig3, use_container_width=True)

    monthly_price = df.groupby("arrival_month_num")["adr"].mean().reset_index()

    fig4 = px.line(monthly_price, x="arrival_month_num", y="adr", title="Average Price Trend")
    st.plotly_chart(fig4, use_container_width=True)

# ===============================
# PRICE & REVENUE MANAGEMENT
# ===============================

with right:

    st.subheader("Booking Information")

    hotel = st.selectbox("Hotel Type",["Resort Hotel","City Hotel"])

    lead_time = st.slider("Lead Time",0,365,30)

    month = st.selectbox(
        "Arrival Month",
        ["January","February","March","April","May","June",
        "July","August","September","October","November","December"]
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["Transient","Contract","Transient-Party","Group"]
    )

    room_type = st.selectbox(
        "Room Type",
        ["A","B","C","D","E","F","G"]
    )

    previous_bookings = st.number_input(
        "Previous Successful Bookings",
        0,50,0
    )

    rooms_available = st.slider("Rooms Available",1,300,50)

    arrival_month = month_to_num(month)

    if st.button("Generate Price Strategy"):

        demand_score = lead_time + previous_bookings

        input_df = pd.DataFrame({

            "hotel":[hotel],
            "lead_time":[lead_time],
            "arrival_month":[arrival_month],
            "reserved_room_type":[room_type],
            "customer_type":[customer_type],
            "previous_bookings_not_canceled":[previous_bookings],
            "demand_score":[demand_score]

        })

        predicted_price = model.predict(input_df)[0]

        price_floor = predicted_price * 0.8
        price_ceiling = predicted_price * 1.3

        dynamic_price = max(price_floor, min(predicted_price, price_ceiling))

        expected_revenue = dynamic_price * rooms_available

        st.success(f"Recommended Price: ₹{round(dynamic_price,2)}")

        st.info(f"Minimum Price Floor: ₹{round(price_floor,2)}")

        st.info(f"Maximum Price Ceiling: ₹{round(price_ceiling,2)}")

        st.metric("Projected Revenue", f"₹{round(expected_revenue,2)}")
