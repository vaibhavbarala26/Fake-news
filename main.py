from flask import Flask, request, jsonify
from run import predict_label  # Ensure this exists

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = predict_label(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
