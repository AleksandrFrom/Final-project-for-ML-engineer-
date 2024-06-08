from flask import Flask, request, jsonify
import pickle
import numpy as np
from lightfm import lightfm

with open(r'model_recom.pkl', 'rb') as pkl_file1: 
    model = pickle.load(pkl_file1)

with open(r'unique_items.pkl', 'rb') as pkl_file2: 
    unique_items = pickle.load(pkl_file2)
    
with open(r'rate_matrix.pkl', 'rb') as pkl_file3: 
    rate_matrix = pickle.load(pkl_file3)    
    
app = Flask(__name__)

@app.route('/recommendation')
def hello_func():
    user = int(request.args.get('user_id'))
    item_ids = np.arange(0, rate_matrix['train'].shape[1])
    user_id = user
    list_pred = model.predict(user_id, item_ids)
    recomendations_ids = np.argsort(-list_pred)[:3]
    recomendations = unique_items[recomendations_ids]
    return f'recommendation for user {user}:{recomendations[:3]}!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)    