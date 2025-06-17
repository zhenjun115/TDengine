import pandas as pd
from flask import Flask, request, jsonify
from prophet import Prophet
# from transformers import AutoModelForCausalLM

app = Flask(__name__)

device = Prophet()

@app.route('/ds_predict', methods=['POST'])
def prophet():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({
                'status':'error',
                'error': 'Invalid input, please provide "input" field in JSON'
            }), 400

        input_data = data['input']
        prediction_length = data['next_len']

        if len(set(input_data)) == 1:
            # for identical array list, std is 0, return directly
            pred_y = [input_data[0] for _ in range(prediction_length)]
        else:
            df = pd.read_json(input_data, orient='split')
            device.fit(df)
            future = device.make_future_dataframe(periods=365)
            forecast = device.predict(future)
            pred_y = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        response = {
            'status': 'success',
            'output': pred_y
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

def main():
    app.run(
            host='0.0.0.0',
            port=7001,
            threaded=True,  
            debug=False     
        )


if __name__ == "__main__":
    main()

