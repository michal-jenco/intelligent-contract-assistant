# ============================================================
#  Project:     Intelligent Contract Assistant - Sample task for Blocshop/Erste
#  Author:      Michal Jenčo
#  Created:     2025
#
#  Copyright (c) 2025 Michal Jenčo


import streamlit as st


# Title
st.title("Intelligent Contract Assistant")


# upload a file
uploaded_file = st.file_uploader("Choose a file to analyse", type=["pdf"])

