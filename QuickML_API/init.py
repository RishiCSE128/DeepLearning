from common import data_preprocessing as dp 
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return('Hello World')

def data_pre_processing():
    preprocessed_data = dp.data_pre_processing()
    for key in preprocessed_data.keys():
        print(key)
        dp.print_tab(preprocessed_data[key])

if __name__ == "__main__":
    app.run()