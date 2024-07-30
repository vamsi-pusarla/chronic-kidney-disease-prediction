from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)