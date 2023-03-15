import streamlit as st
import pandas as pd
import numpy as np
import openai
from scipy.spatial.distance import cosine
import time



def write_short_copy(user_prompt: str, df: pd.DataFrame) -> str:

    def get_embedded_vector(text: str, sleep:bool = True) -> list[float]:
        """
        Returns the embedded vector for a given text string using OpenAI API.
        
        Args:
            text (str): Input text to be embedded.
        
        Returns:
            list[float]: Embedded vector for the input text.
        """
        # Use OpenAI API to generate embedded vector for input text
        embedding_dict = openai.Embedding.create(
            input = [text],
            model = 'text-embedding-ada-002'
        )
        
        # Extract embedded vector from the API response
        embedded_vector = embedding_dict['data'][0]['embedding']

        return embedded_vector
    
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Returns the cosine similarity of two vectors.
        
        Args:
            v1 (np.ndarray): A numpy array representing the first vector.
            v2 (np.ndarray): A numpy array representing the second vector.
        
        Returns:
            float: The cosine similarity of the two vectors.
        """
        # Compute the cosine similarity of the two vectors using the cosine distance formula
        cosine_sim = 1 - cosine(v1, v2)
        
        return cosine_sim

    def create_final_prompt(reference,user_prompt):

        final_prompt = f'''
        You are a world-class email copy generator. Your goal is to create an email that will spark the reader's curiosity and get them to click on a link.

        For reference, use these high-converting email copies (surrounded by brackets) as templates.

        Template 1: [{reference[0]}]

        Template 2: [{reference[1]}]

        Template 3: [{reference[2]}]

        Please use its structure, curiosity-inducing phrasing, tone of voice, and length. Please exclude the names presented on them.   
        Now, I have already briefed the client on what is the type of offer he is promoting, and here is what he said verbatim:

        "{user_prompt}"

        So, now that you are armed with all of the relevant information, please create a email copy (using punchy tone of voice) pushing readers to the offer our client is promoting:'''

        return final_prompt

    def generate_short_copy(prompt: str, temperature: int = 0.5, top_p:int = 0.5, max_tokens:int = 1000) -> str:
        """
        Generates text based on the given prompt using the default OpenAI model (text-davinci-002)
        and temperature (0.5).
        
        Args:
            prompt (str): The prompt to use for generating the text.
        
        Returns:
            str: The generated text.
        """
        # Create the completion call using the OpenAI API
        completion_dict = openai.Completion.create(
            prompt=prompt,
            model="text-davinci-003",
            temperature=temperature,
            max_tokens= max_tokens,
            top_p = top_p
        )
        
        # Extract and return the generated text from the completion call response
        generated_text = completion_dict['choices'][0]['text'].strip()
        return generated_text



    """
    Generates short copy by finding most similar email contents in a dataframe
    based on a user prompt and then using it to engineer a final prompt
    and generate short copy using that final prompt.

    Args:
        user_prompt (str): The user prompt in string format.
        df (pd.DataFrame): The dataframe containing email content vectors.

    Returns:
        str: The generated short copy based on the provided user prompt and email contents in the dataframe.
    """
    # Get the embedded vector for the user prompt
    user_prompt_vector = get_embedded_vector(user_prompt, sleep=False)
    
    # Find the cosine similarity between user_prompt_vector and email_content_vector for each row in the dataframe
    df = df.assign(similarity=lambda x: x['email_content_vector'].apply(lambda x: cosine_similarity(user_prompt_vector, x)))
    
    # Get the indices of the top 3 most similar email contents
    top_n_indices = df['similarity'].nlargest(3).index
    
    # Create a dictionary of the most similar email contents
    most_similar_email_contents_dict = df.loc[top_n_indices].reset_index()['email_content'].to_dict()
    
    # Use the most similar email contents to engineer the final prompt
    engineered_prompt = create_final_prompt(most_similar_email_contents_dict, user_prompt)
    
    # Generate short copy using the engineered prompt
    short_copy = generate_short_copy(engineered_prompt, 0.75, 1)
    
    return short_copy
