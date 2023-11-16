import streamlit as st
#from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv
import json
import os
from langchain import *
from langchain.chains import LLMChain

import spacy
import requests
import time
import requests
try:
    # Load the spaCy model 'en_core_web_sm'
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not installed, download it
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("API key is missing. Set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    pixil_api_key = os.getenv("PIXIL_API_KEY")
    if not pixil_api_key:
        raise Exception("Pixil API key is missing. Set the PIXIL_API_KEY environment variable.")

#def generate_response(input_text):
 #   pass
    


    st.title('👓📖 Story Generator with ChatGPT...')
    st.header('This is a :orange[Story]  Generator', divider='rainbow')
    discription = '''Provide the start of the story and ChatGpt will
                 complete the rest for you ! :sunglasses

    '''
    st.markdown(discription)


    primaryColor="#F63366"
    backgroundColor="#FFFFFF"
    secondaryBackgroundColor="#F0F2F6"
    textColor="#262730"
    font="sans serif"
    
    prompt = st.text_area("Enter the start of the story ...")
    if st.button("Generate Story"):
        # Generate response from ChatGPT
        response = generate_story(prompt)

        # Display the story
        st.subheader("Generated Story:")


        with st.spinner('Wait for it...'):
            time.sleep(5)
             
        st.write(response)
        st.success('Done!')  
        # Fetch and display images related to the story (replace with your image-fetching code)
        images = fetch_images(response,pixil_api_key)
        if images:
            st.subheader("Related Images:")
            for image_url in images:
                st.image(image_url, caption="Image related to the story", use_column_width=True)

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and token.is_alpha]

def generate_story(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the desired engine
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text

# Placeholder function for image fetching (using a fictional image API)
def fetch_images(story,pixil_api_key):
    try:
        # Extract keywords or entities from the story using NLP
        keywords = extract_keywords(story)

        # Use the Pixil API to fetch images based on keywords
        pixil_api_url = "https://pixilapi.com/api/images/search"
        params = {"keywords": ",".join(keywords), "limit": 3}  # Adjust parameters as needed
        headers = {"Authorization": f"Bearer {pixil_api_key}"}

        response = requests.get(pixil_api_url, params=params, headers=headers)

        if response.status_code == 200:
            # Assuming Pixil API returns a JSON response with image URLs
            return response.json()["image_urls"]
        else:
            return None

    except Exception as e:
        print(f"Error fetching images: {e}")
        return None

#main function
if __name__ == "__main__":
    main()