# runtime.py
from flask import Flask, request, jsonify
from db_ops import add_record, fetch_record, update_record
import json

app = Flask(__name__)

@app.route('/add', methods=['POST'])
def http_add():
    try:
        data = request.get_json()
        if not data or 'table' not in data or 'data' not in data:
            return jsonify({"error": "Missing 'table' or 'data'"}), 400
        table = data['table']
        record_data = data['data']
        result_id = add_record(table, record_data)
        return jsonify({"id": result_id, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/fetch', methods=['GET'])
def http_fetch():
    try:
        table = request.args.get('table')
        id_key = request.args.get('id')
        filters = request.args.get('filters')  # e.g., "name=Alice,email=test@example.com"

        if not table:
            return jsonify({"error": "Missing 'table' parameter"}), 400

        result = fetch_record(table, id_key, filters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/update', methods=['POST'])
def http_update():
    try:
        data = request.get_json()
        if not data or 'table' not in data or 'id' not in data or 'updates' not in data:
            return jsonify({"error": "Need 'table', 'id', 'updates'"}), 400
        table = data['table']
        id_key = data['id']
        updates = data['updates']
        success = update_record(table, id_key, updates)
        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)