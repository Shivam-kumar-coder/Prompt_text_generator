import streamlit as st
from transformers import pipeline, set_seed

# from transformers import pipeline
# pipeline("text-generation", model="gpt2")


@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()


st.title("Text Generator")
topic = st.text_input("Enter a topic to generate text:")

max_len = st.slider("Max length", 50, 300, 100, step=10)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        set_seed(42)
        result = generator(topic, max_length=max_len, num_return_sequences=1)
        st.success("Done!")
        st.write("### Generated Text:")
        st.write(result[0]["generated_text"])
