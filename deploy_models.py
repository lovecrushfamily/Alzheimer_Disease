"""
Created on Dec, 23, 2024 , 14:00

@author: lovecrush
"""
import numpy as np
import pickle
import streamlit as st


import pickle

saved_model_file_path ="saved_models/SVC.pickle"

loaded_models = pickle.load(open(saved_model_file_path, "rb"))         


def make_prediction(input_data):
    """"""

    # Change input_data into numpy array
    input_data_arr = np.asarray(input_data, dtype=np.float32)

    print("input data arr:", input_data_arr)

    # Reshape the input data
    input_data_reshaped = input_data_arr.reshape(1,-1)
    print("input data reshaped:", input_data_reshaped)

    # make prediction with model[0]
    prediction = loaded_models[0].predict(input_data_reshaped)

    print(prediction)

    if (prediction[0] == 0):
        return "True"
    else:
        return "False"


def main():
    """"""

    # Giving a title
    st.title("Web model title here?")

        
    inputs = [st.number_input(val, value=idx) for idx, val in enumerate("Lovecrus")]

    # parser_input = inputs.split(',')

    # print(parser_input)

    # parser_input_ = [float(num) for num in parser_input]

    # print("inputs:", inputs)

    result = ""
    if st.button("Click on this!"):
        result = make_prediction(parser_input_)
        


    st.success(result)



if __name__ == "__main__":
    main()
    

    




