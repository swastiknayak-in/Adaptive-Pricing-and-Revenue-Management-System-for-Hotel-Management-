# ===============================
# PRICE & REVENUE MANAGEMENT
# ===============================
with right:

    st.subheader("Booking Information")

    hotel = st.selectbox(
        "Hotel Type",
        ["Resort Hotel","City Hotel"]
    )

    lead_time = st.slider(
        "Lead Time (Days)",
        0,
        365,
        30
    )

    month = st.selectbox(
        "Arrival Month",
        [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
        ]
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
        0,
        50,
        0
    )

    rooms_available = st.slider(
        "Rooms Available",
        1,
        300,
        50
    )

    arrival_month = month_to_num(month)

    if st.button("Generate Price Strategy"):

        input_df = pd.DataFrame({

            "hotel":[hotel],
            "lead_time":[lead_time],
            "arrival_month":[arrival_month],
            "reserved_room_type":[room_type],
            "customer_type":[customer_type],
            "previous_bookings_not_canceled":[previous_bookings]

        })

        predicted_price = model.predict(input_df)[0]

        # -------------------------
        # Revenue Management Logic
        # -------------------------

        price_floor = predicted_price * 0.8
        price_ceiling = predicted_price * 1.3

        demand_score = lead_time + previous_bookings

        if demand_score > 120:
            demand_level = "High Demand"
            dynamic_price = predicted_price * 1.2

        elif demand_score > 60:
            demand_level = "Normal Demand"
            dynamic_price = predicted_price

        else:
            demand_level = "Low Demand"
            dynamic_price = predicted_price * 0.9

        dynamic_price = max(price_floor, min(dynamic_price, price_ceiling))

        expected_revenue = dynamic_price * rooms_available

        st.success(f"Recommended Price: ₹ {round(dynamic_price,2)}")

        st.info(f"Minimum Price Floor: ₹ {round(price_floor,2)}")

        st.info(f"Maximum Price Ceiling: ₹ {round(price_ceiling,2)}")

        st.write("Demand Level:", demand_level)

        st.subheader("Expected Revenue")

        st.metric(
            label="Projected Revenue",
            value=f"₹ {round(expected_revenue,2)}"
        )
