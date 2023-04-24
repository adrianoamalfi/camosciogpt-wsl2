
# import delle librerie necessarie
import torch
from peft import PeftModel
import transformers
import gradio as gr


from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

# impostazione del runtime su CUDA
assert torch.cuda.is_available(), "Change the runtime type to GPU"
device = "cuda"


# Inizializzo Tokenizer e Model LLAMA 
tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

# Introduco il modello camoscio
model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-llama")

# Routine per la generazione del prompt
def generate_prompt(instruction, input=None):
    if input:
        return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Input:
{input}

### Risposta:"""
    else:
        return f"""Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta.

### Istruzione:
{instruction}

### Risposta:"""

# Configurazione dei parametri di generazione
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)

# Routine per la valutazione del prompt 
def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    with torch.no_grad():
      generation_output = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=256
      )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Risposta:")[1].strip()

# Rimuovo i warnings
import warnings
warnings.filterwarnings("ignore")

# Inizializzo l'interfaccia gradio
g = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Scrivi una breve biografia su Dante Alighieri"
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none")
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=7,
            label="Output",
        )
    ],
    title="🇮🇹🦙 Camoscio | adrianoamalfi.com guide")
g.launch()
