
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

@app.route('/ds_predict', methods=['POST'])
def deepseek_server():
    """
    Start the DeepSeek server.
    """
    try:
        data = request.get_json()
        # 打印data数据
        print(f"Received data: {data}")
        if not data or 'input' not in data:
            return jsonify({
                'status':'error',
                'error': 'Invalid input, please provide "input" field in JSON'
            }), 400

        input_data = data['input']
        prediction_length = data['next_len']

        if len(set(input_data)) == 1:
            pred_y = [input_data[0] for _ in range(prediction_length)]
        else:
            client = OpenAI(api_key="sk-90541eb010af4dfe9e4efa4998ad4804", base_url="https://api.deepseek.com/v1")

            # data 
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": "基于当前数据，预测一下未来数据。输入数据为:{data}"}
                ],
                stream=False
            )
            # 打印response
            print("#######################################################\n")
            print(response.choices[0].message.content)
            print("#######################################################\n")
            # print(f"response data: {response}")
            pred_y = [input_data[0] for _ in range(prediction_length)]
        
        # pred_y = [input_data[0] for _ in range(prediction_length)]
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
    """ Main function to start the DeepSeek server.
    """
    app.run(
            host='0.0.0.0',
            port=8001,
            threaded=True,
            debug=False  
        )

if __name__ == "__main__":
    main()