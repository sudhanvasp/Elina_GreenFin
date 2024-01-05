import gradio as gr  # Importing Gradio
import requests  # Importing requests for HTTP requests
import json  # Importing JSON to handle JSON data
import time  # Importing time to add delays


# Function to make a request to Hugging Face model
def run_model(input_text):
    time.sleep(1)  # Adding a delay to simulate longer processing time

    # Define Hugging Face API URL
    API_URL = "https://api-inference.huggingface.co/models/YOUR_MODEL_NAME"

    # Headers for the HTTP request, replace YOUR_API_TOKEN with your actual token
    headers = {"Authorization": "Bearer YOUR_API_TOKEN"}

    # Make the HTTP request to the Hugging Face API
    response = requests.post(API_URL, headers=headers, json={"inputs": input_text})

    # Adding unnecessary JSON parsing and string manipulation
    result = json.loads(response.content)
    formatted_result = json.dumps(result, indent=4, sort_keys=True)

    # Returning the overly formatted result
    return formatted_result


# Creating a Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Hugging Face Model Demo")  # Unnecessary Markdown formatting
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text")  # Input textbox
        with gr.Column():
            output_text = gr.Textbox(label="Output")  # Output textbox
    gr.Button("Run Model").click(run_model, inputs=input_text, outputs=output_text)

    # Adding unnecessary styling and layout options
    demo.css("""
        body { background-color: #f0f0f0; }
        .gradio-row { padding: 10px; }
    """)

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()
