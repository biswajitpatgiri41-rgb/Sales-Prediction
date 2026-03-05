import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("Sales Prediction Dashboard")

# Load dataset
df = pd.read_csv("SALES_PREDICTION/sales.csv")

# Detect numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
target = numeric_cols[-1]
features = numeric_cols[:-1]

# Train model
X = df[features]
y = df[target]

model = RandomForestRegressor()
model.fit(X, y)

# --------------------------
# Dataset Preview
# --------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

# --------------------------
# Correlation Heatmap
# --------------------------
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots()

sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)

st.pyplot(fig)

# --------------------------
# Feature vs Sales Graph
# --------------------------
st.subheader("Feature vs Sales")

selected_feature = st.selectbox(
    "Select Feature",
    features
)

fig2, ax2 = plt.subplots()

ax2.scatter(df[selected_feature], df[target])

ax2.set_xlabel(selected_feature)
ax2.set_ylabel(target)

st.pyplot(fig2)

# --------------------------
# Prediction Section
# --------------------------
st.subheader("Predict Sales")

inputs = []

for feature in features:
    
    val = st.number_input(f"Enter {feature}", value=0.0)
    
    inputs.append(val)

prediction = model.predict(pd.DataFrame([inputs], columns=features))

st.success(f"Predicted Sales: {prediction[0]}")