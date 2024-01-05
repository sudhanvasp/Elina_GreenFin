import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
from threading import Thread
import numpy as np

import yfinance as yf
import matplotlib.pyplot as plt


# Function to fetch and plot company stock trends
def plot_stock_trend(stock_ticker):
    # Fetch historical stock price data for the last 60 days
    company = yf.Ticker(stock_ticker)
    hist_data = company.history(period="60d")

    # Plotting the Closing Price trend
    plt.figure(figsize=(10, 6))
    plt.plot(hist_data['Close'])
    plt.title(f'{stock_ticker} Stock Price Trend - Last 60 Days')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Return the plot
    return plt


np.random.seed(42)
dataset = {"x": np.random.rand(100), "y": np.random.rand(100)}
tokenizer = AutoTokenizer.from_pretrained("sudhanvasp/ELINA")
model = AutoModelForCausalLM.from_pretrained("sudhanvasp/ELINA", torch_dtype=torch.float16)
model = model.to('cuda:0')
drop_down_options = ["Auto", "1-10L", "50L-100L", "100L-200", "200+"]
drop_down_options2 = ["Auto", "Non Profit", "LLC", "Public", "OPC" ]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    # Append the specific request for risk analysis to the end of each prompt
    additional_request = " and provide me with financial risk analysis. You must always give a risk coefficient for it(0-100) "+str(text_input)

    # Transform the history into the format expected by the transformer
    history_transformer_format = history + [[message + additional_request, ""]]
    stop = StopOnTokens()

    # Concatenate the messages in the format required by your model
    messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]]) for item in history_transformer_format])

    # Prepare the inputs for the model
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    # Define parameters for text generation
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    # Start the generation in a separate thread
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Iterate over the generated tokens and yield the result
    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message


#def generate_risk_coeff():
    #return np.random.randint(1, 101)


# Rest of your code for setting up the Gradio app
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # ELINA GreenFin AI Agent
        """)
    with gr.Row():
        animal_dropdown = gr.Dropdown(drop_down_options, label="Avg revenue of the company:")
        car_dropdown = gr.Dropdown(drop_down_options2, label="Organisation Type:")

    with gr.Row():
        text_input = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
        submit_button = gr.Button("Freeze")
    output_plot = gr.Plot()

    submit_button.click(
        fn=plot_stock_trend,
        inputs=text_input,
        outputs=output_plot
    )

    chatty = gr.ChatInterface(predict)
    gr.Markdown(
        """
        # E: Enhanced 
        # L: Learning for 
        # I: Intelligent and 
        # N: Novel 
        # A: Analytics
        Green 
        Finance
        """)
    #gr.Label(generate_risk_coeff(), label="Risk Coefficent ")

demo.launch()
