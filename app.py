import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Safe, lightweight model for CPU
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU only

pipe = load_model()

# UI
st.title("⚖️ Legal Chatbot - GPT2 (CPU Friendly)")
st.write("Ask a legal question (IPC, contracts, cyber law, etc.)")

user_input = st.text_input("You:", placeholder="What is Section 420 of IPC?")

if user_input:
    prompt = f"User: {user_input}\nAssistant:"
    response = pipe(prompt, max_new_tokens=150, temperature=0.7)[0]['generated_text']
    st.markdown("**Assistant:**")
    st.write(response.split("Assistant:")[-1].strip())
