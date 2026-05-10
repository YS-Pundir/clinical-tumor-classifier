import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. Load the winning pipeline and dataset info
# Uses relative path to work on both your PC and GitHub/Hugging Face
model_path = "Exports/logistic_regression_pipeline.joblib"
try:
    pipeline = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Get feature names from the source dataset
data_info = load_breast_cancer()
feature_names = data_info.feature_names

def predict_tumor(*features):
    """Function to process inputs and return formatted results."""
    # Convert input list to a DataFrame for the pipeline
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Run prediction and get probabilities
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]
    
    # Mapping: 0 = Malignant, 1 = Benign
    label = "🚨 Malignant (Cancerous)" if prediction == 0 else "✅ Benign (Safe)"
    conf = probability[prediction]
    
    # Return formatted strings for the text boxes
    return label, f"{conf:.2%}"

# 2. Build the Interface (Gradio 6.0 standards)
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Clinical Tumor Classifier")
    gr.Markdown("### **IIT Module 2 Capstone: Diagnostic Inference Engine**")
    gr.Markdown("Adjust the cellular measurement sliders below to evaluate the patient's diagnostic data.")
    
    with gr.Row():
        # Left Column: User Inputs
        with gr.Column(scale=2):
            inputs = []
            with gr.Accordion("🔬 Cellular Measurements (30 Parameters)", open=True):
                # We iterate through all 30 features to create sliders
                for name in feature_names:
                    # Setting default values around the dataset means
                    inputs.append(gr.Slider(
                        label=name.title(), 
                        minimum=0, 
                        maximum=250, 
                        value=20,
                        step=0.1
                    ))
        
        # Right Column: Results and Analysis
        with gr.Column(scale=1):
            gr.Markdown("## 📋 Analysis Results")
            label_output = gr.Textbox(label="Predicted Diagnosis", interactive=False)
            conf_output = gr.Textbox(label="Model Confidence Score", interactive=False)
            
            predict_btn = gr.Button("Run Diagnostic Analysis", variant="primary")
            
            gr.Info("Note: This model is for educational purposes (IIT Module 2) and should not be used for actual medical advice.")

    # Link the button to the prediction function
    predict_btn.click(
        fn=predict_tumor, 
        inputs=inputs, 
        outputs=[label_output, conf_output]
    )

# 3. Launch with theme and sharing options
if __name__ == "__main__":
    # Moved 'theme' to launch() to comply with Gradio 6.0
    # Set share=True if you want a public URL for 72 hours
    demo.launch(theme=gr.themes.Soft(), share=False)