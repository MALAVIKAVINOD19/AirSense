
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="AirSense",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AirSense")
st.subheader("AI-powered Air Quality Understanding and Forecasting")

st.write(
    "Ask questions about air-quality measurements "
    "or upload readings to generate a next-hour forecast."
)

# -----------------------------
# FEATURE INFORMATION
# -----------------------------

features = [
    "CO(GT)", "PT08.S1(CO)", "C6H6(GT)",
    "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
    "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)"
]

knowledge = {
    "CO(GT)": "Carbon monoxide. A harmful gas produced by incomplete combustion.",
    "PT08.S1(CO)": "Sensor response associated with carbon monoxide.",
    "C6H6(GT)": "Benzene concentration, a volatile organic compound.",
    "PT08.S2(NMHC)": "Sensor response associated with non-methane hydrocarbons.",
    "NOx(GT)": "Nitrogen oxides, a group of gases including nitric oxide and nitrogen dioxide.",
    "PT08.S3(NOx)": "Sensor response associated with nitrogen oxides.",
    "NO2(GT)": "Nitrogen dioxide, an air pollutant that can affect respiratory health.",
    "PT08.S4(NO2)": "Sensor response associated with nitrogen dioxide.",
    "PT08.S5(O3)": "Sensor response associated with ozone-related measurements."
}

# -----------------------------
# CHATBOT
# -----------------------------

def airsense_chatbot(question):

    q = question.lower()

    keywords = {
        "carbon monoxide": "CO(GT)",
        "benzene": "C6H6(GT)",
        "nitrogen oxides": "NOx(GT)",
        "nox": "NOx(GT)",
        "nitrogen dioxide": "NO2(GT)",
        "no2": "NO2(GT)",
        "ozone": "PT08.S5(O3)",
        "pt08.s1": "PT08.S1(CO)",
        "pt08.s2": "PT08.S2(NMHC)",
        "pt08.s3": "PT08.S3(NOx)",
        "pt08.s4": "PT08.S4(NO2)",
        "pt08.s5": "PT08.S5(O3)"
    }

    for keyword, feature in keywords.items():
        if keyword in q:
            return f"{feature}: {knowledge[feature]}"

    return "I'm sorry, I don't know or I'm not aware of that information."

# -----------------------------
# FORECASTING MODEL
# -----------------------------

class AirSenseTransformerExp5(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(9, 64)
        self.position_embedding = nn.Parameter(torch.zeros(1, 12, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 9)
        )

    def forward(self, x):

        x = self.input_projection(x)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = x[:, -1, :]

        return self.output_layer(x)


@st.cache_resource
def load_airsense_model():

    model = AirSenseTransformerExp5()

    checkpoint = torch.load(
        "model/experiment_5/airsense_exp5_best.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint)
    model.eval()

    mean = np.load("data/processed/normalized/mean.npy")
    std = np.load("data/processed/normalized/std.npy")

    return model, mean, std


# -----------------------------
# APPLICATION TABS
# -----------------------------

chat_tab, forecast_tab = st.tabs(
    ["💬 Ask AirSense", "📊 Air Quality Forecast"]
)

# CHAT TAB

with chat_tab:

    st.subheader("Ask an Air Quality Question")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask AirSense"):

        if question.strip():
            st.info(airsense_chatbot(question))
        else:
            st.warning("Please enter a question.")


# FORECAST TAB

with forecast_tab:

    st.subheader("Next-Hour Air Quality Forecast")

    st.write(
        "Upload a CSV containing exactly 12 hourly readings "
        "for the nine required features."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            if list(data.columns) != features:
                st.error("CSV columns do not match the required format.")

            elif data.shape[0] != 12:
                st.error("CSV must contain exactly 12 hourly readings.")

            else:

                data = data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if data.isnull().values.any() or not np.isfinite(data.values).all():

                    st.error("CSV contains missing or invalid numerical values.")

                else:

                    st.success("CSV uploaded and validated successfully!")

                    st.dataframe(data)

                    if st.button("Generate Forecast"):

                        model, mean, std = load_airsense_model()

                        selected_indices = np.arange(9)

                        normalized = (
                            data.values - mean[selected_indices]
                        ) / std[selected_indices]

                        input_tensor = torch.tensor(
                            normalized,
                            dtype=torch.float32
                        ).unsqueeze(0)

                        with torch.no_grad():
                            output = model(input_tensor)

                        predictions = (
                            output.numpy()[0] *
                            std[selected_indices] +
                            mean[selected_indices]
                        )

                        st.subheader("Predicted Next-Hour Readings")

                        results = pd.DataFrame({
                            "Feature": features,
                            "Predicted Value": np.round(predictions, 3)
                        })

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        st.caption(
                            "These are model-generated estimates, "
                            "not live sensor measurements or health assessments."
                        )

        except Exception as e:

            st.error(f"An error occurred: {e}")


st.divider()

st.caption(
    "AirSense | Academic project prototype | "
    "Forecasting model and rule-based conversational assistant"
)
