import streamlit as st
import pandas as pd
import numpy as np
import openai
from scipy.spatial.distance import cosine
import time
from generator_functions import write_short_copy


st.title("Welcome to The Email Copy Generator!")

'''
This mini-app was created to help small business owners create effective email copies quickly and easily.
If you want a thorough explanation of how I built this app and what to expect from it, please [read this article](https).

## How To Use This App
Simply fill in the text boxes below with information relevant to the promotion you are sending traffic to.
These questions are meant to help you retrieve the key "psychological triggers" of your offer in conversational format.

Feel free to test multiple angles and see what kind of copy the generator comes up with.

**PS:** The more specific you are in your answers, the better.
***

## Generate Your Email Copy:
'''

apikey = ''
apikey = st.text_input("**Your OpenAI API key:**")

'''
**I will NOT store this information!** This app used to be run using my own API key, but some people abused it...
So that's why I now have to ask you for your API key to be able to run this app ([here's how you can get a OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)).
It's worth repeating that I will not store your API key in any way shape or form and I advise you to generate a new one before using the app and deleting it afterwards.
'''

openai.api_key = apikey

df = pd.read_json('swipe file.json')



who_are_u = st.text_input('**Introduce yourself and talk about what makes you a credible person (highlight relevant qualifications, experience and/or credentials)**')

one_belief = st.text_input('**Briefly explain the offer you wish to promote with the email.**')

if one_belief != '':
    one_belief = "The email should be about a " + one_belief + '.\n'

unlike_anything = st.text_input('**What makes your offer completely unique and mind-blowing?**')
if unlike_anything != '':
    unlike_anything = "What makes this opportunity so intriguing is that " + unlike_anything + '.\n'

y_care = st.text_input('**What is the underlying promise of your offer and why the reader should care about it?**')
if y_care != '':
    y_care = "Now, the reason why the reader should care about it is because " + y_care + '.\n'

true = st.text_input('**Please show relevant proof points about the offer you are promoting.**')
if true != '':
    true = "What's more, I know this is true because " + true + '.\n'


y_now = st.text_input('**Is there is something inherently time-sensitive or scarce about your offer? If so, explain.**')
if y_now != '':
    y_now =  "However, people have to hurry, since " + y_now + '.\n'


user_prompt = f'''{who_are_u} {one_belief} {unlike_anything} {y_care} {true} {y_now} '''


accepted_terms = st.checkbox('I have read and accept the terms and conditions')
isClicked = st.button('Generate Email Copy', key='generate_copy_button', type='primary')

if isClicked:
    if apikey == '':
        st.error('You need to provide your API key.')
    elif not accepted_terms:
        st.error('You must first read and accept the [terms and conditions](https:terms.com).')
    else:
        # Use st.spinner() to display a spinner while the email copy is being generated
        with st.spinner('Generating email copy...'):
            short_copy = write_short_copy(user_prompt, df)
        
        # Display the generated email copy once it's ready
        st.success('Email copy generated successfully:')
        st.text_area('Email Copy:', short_copy, height = 200)
