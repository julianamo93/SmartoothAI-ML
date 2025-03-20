from flask import jsonify, request
import pandas as pd
from app.ml_models import get_dental_tips, get_recommendations, predict, generate_recommendation

def init_routes(app):
    @app.route('/')
    def home():
        return jsonify(message="Bem vindo ao Smartooth AI - Uma solução ODONTOPREV")

    @app.route('/filter_procedures', methods=['GET'])
    def filter_procedures():
        try:
            procedures = pd.read_csv('data/procedures.csv')  
            if procedures.empty:
                return jsonify({"error": "No procedures data available"}), 404
            return jsonify(procedures.to_dict(orient='records'))
        except FileNotFoundError:
            return jsonify({"error": "Procedures file not found"}), 500
        except Exception as e:
            return jsonify({"error": f"Error reading procedures: {str(e)}"}), 500

    @app.route('/dental_tips', methods=['GET'])
    def dental_tips():
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        try:
            tips = get_dental_tips(patient_id)
            return jsonify(tips=tips)
        except Exception as e:
            return jsonify({"error": f"Error fetching dental tips: {str(e)}"}), 500

    @app.route('/recommendations', methods=['GET'])
    def recommendations():
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        try:
            recommendations = get_recommendations(patient_id)
            return jsonify(recommendations=recommendations)
        except Exception as e:
            return jsonify({"error": f"Error fetching recommendations: {str(e)}"}), 500

    @app.route('/predict', methods=['POST'])
    def predict_route():
        try:
            data = request.get_json()
            if not data or not all(k in data for k in ['age', 'history', 'severity']):
                return jsonify({"error": "Missing required data: 'age', 'history', and 'severity'"}), 400

            # Previsão e recomendação
            prediction = predict(data)
            recommendation = generate_recommendation(prediction)
            
            return jsonify({
                'prediction': prediction,
                'recommendation': recommendation
            })
        
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500
