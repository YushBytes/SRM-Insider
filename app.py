import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv"
)

# Features and target
X = df.drop("logS", axis=1)
y = df["logS"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Drug Solubility Prediction")

mol_logp = st.number_input("MolLogP", value=2.5)
mol_wt = st.number_input("MolWt", value=180.0)
num_rot_bonds = st.number_input("NumRotatableBonds", value=1.0)
aromatic_prop = st.number_input("AromaticProportion", value=0.5)

if st.button("Predict"):
    prediction = model.predict(
        [[mol_logp, mol_wt, num_rot_bonds, aromatic_prop]]
    )

    st.success(f"Predicted LogS: {prediction[0]:.3f}")
