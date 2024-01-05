import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import numpy as np

np.random.seed(42)
dataset = {"x": np.random.rand(100), "y": np.random.rand(100)}
tokenizer = AutoTokenizer.from_pretrained("sudhanvasp/Resumedarwin")
model = AutoModelForCausalLM.from_pretrained("sudhanvasp/Resumedarwin", torch_dtype=torch.float16)
model = model.to('cuda:0')
drop_down_options = ["1 - 10L", "50L-100L", "100L-200", "200+"]
drop_down_options2 = ["Non Profit", "LLC", "Public", "OPC"]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    # Append the specific request for risk analysis to the end of each prompt
    additional_request = " and provide me with financial risk analysis."

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

    # Generate the risk coefficient after the prompt completion
        risk_coefficient = generate_risk_coeff()
        yield partial_message
        yield f"\n<bot>: The risk coefficient is {risk_coefficient}"


# Create a function to generate a random value between 1 and 100
def generate_risk_coeff():
    return np.random.randint(1, 101)


# Rest of your code for setting up the Gradio app
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Green Finance Analyst
        """)
    with gr.Row():
        animal_dropdown = gr.Dropdown(drop_down_options, label="Avg revenue of the company:")
        car_dropdown = gr.Dropdown(drop_down_options2, label="Organisation Type:")

    chatty = gr.ChatInterface(predict)

demo.launch()
