import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
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

    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
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
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message


    # Custom trigger for analysis:
    if message.lower().strip() == "analyze":  # Replace with your desired trigger condition
        risk_coeff = extract_risk_coeff(history)  # Extract risk coefficient
        risk_coeff_label.update(risk_coeff)  # Update the label


# Rest of your code for setting up the Gradio app
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Green Finance Analyst
        """)
    with gr.Row():
        animal_dropdown = gr.Dropdown(drop_down_options, label="Avg revenue of the company:")
        car_dropdown = gr.Dropdown(drop_down_options2, label="Organisation Type:")

    risk_coeff_label = gr.Label("", label="Risk Coefficient")

    chatty = gr.ChatInterface(predict, live=True)
    gr.Button("Analyze")  # Separate button for manual submission

demo.launch()
