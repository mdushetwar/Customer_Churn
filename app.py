from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

#creating flask app
app=Flask(__name__)

@app.route('/')
def Home():
    return 'Hellow'



