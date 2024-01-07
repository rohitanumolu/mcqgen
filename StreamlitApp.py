import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st
#from langchain.callbacks import get_openai_callback
from src.mcqgenerator.mcqgenerator import generate_evaluate_chain
from langchain_community.callbacks import get_openai_callback

file_path=r"C:\Users\rohit\Desktop\genai tuts\mcqgen\Response.json"
with open(file_path, 'r') as file:
    RESPONSE = json.load(file)

# Creating a title for the app
st.title("MCQs Creator App with Langchian and OpenAI")

#Create a form using st.form
with st.form("user_inputs"):

    uploaded_file=st.file_uploader("Upload a pdf or text file")

    mcq_count=st.number_input("Number of MCQs needed", min_value=3, max_value=30)

    subject=st.text_input("Subject name", max_chars=20)

    tone=st.text_input("Difficulty of the questions", max_chars=20, placeholder="Simple")

    button=st.form_submit_button("Create MCQs")

    #Check if the button is clicked and all fields have inputs

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading....."):
            try:
                text=read_file(uploaded_file)
                #Counting tokens and the cost of API call
                
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject":subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE)
                        }
                        )

            except Exception as e:
               traceback.print_exception(type(e), e, e.__traceback__)
               st.error("error")
            
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")

                if isinstance(response, dict):
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_data=get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index=df.index+1
                            st.table(df)

                            st.text_area(label="Review", value=response["reveiw"])

                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)