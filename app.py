from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import csv
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Data storage directory
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/api/save-trial', methods=['POST'])
def save_trial():
    """Save a single trial data"""
    try:
        data = request.json
        test_type = data.get('testType')
        session_id = data.get('sessionId', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Create CSV file path
        csv_file = os.path.join(DATA_DIR, f'{test_type}_{session_id}.csv')
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(csv_file)
        
        # Write to CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'trial', 'testType', 'word', 'color', 'userAnswer', 
                         'correct', 'reactionTime', 'isGo', 'stimulusType', 'responded', 'congruent',
                         'hour', 'minute', 'second', 'millisecond', 'errorType']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Get detailed time information
            now = datetime.now()
            detailed_time = data.get('detailedTime', {})
            
            # Prepare row data
            row = {
                'timestamp': data.get('timestamp', now.isoformat()),
                'trial': data.get('trial'),
                'testType': test_type,
                'hour': detailed_time.get('hour', now.hour),
                'minute': detailed_time.get('minute', now.minute),
                'second': detailed_time.get('second', now.second),
                'millisecond': detailed_time.get('millisecond', int(now.microsecond / 1000)),
            }
            
            # Stroop specific fields
            if test_type == 'stroop':
                row.update({
                    'word': data.get('word'),
                    'color': data.get('color'),
                    'userAnswer': data.get('userAnswer'),
                    'correct': data.get('correct'),
                    'reactionTime': data.get('reactionTime'),
                    'congruent': data.get('congruent'),  # True if word matches color
                    'errorType': data.get('errorType', ''),  # Error type classification
                    'isGo': '',
                    'stimulusType': '',
                    'responded': ''
                })
            # Go/No-Go specific fields
            elif test_type == 'gonogo':
                row.update({
                    'word': '',
                    'color': '',
                    'userAnswer': '',
                    'correct': data.get('correct'),
                    'reactionTime': data.get('reactionTime') or '',
                    'congruent': '',
                    'errorType': data.get('errorType', ''),
                    'isGo': data.get('isGo'),
                    'stimulusType': data.get('stimulusType'),
                    'responded': data.get('responded')
                })
            
            writer.writerow(row)
        
        return jsonify({'success': True, 'message': 'Trial saved'}), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-session', methods=['POST'])
def save_session():
    """Save complete session data"""
    try:
        data = request.json
        test_type = data.get('testType')
        session_id = data.get('sessionId', datetime.now().strftime('%Y%m%d_%H%M%S'))
        trials = data.get('trials', [])
        
        if not trials:
            return jsonify({'success': False, 'error': 'No trials data'}), 400
        
        # Create CSV file path
        csv_file = os.path.join(DATA_DIR, f'{test_type}_{session_id}.csv')
        
        # Write all trials to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'trial', 'testType', 'word', 'color', 'userAnswer', 
                         'correct', 'reactionTime', 'isGo', 'stimulusType', 'responded', 'congruent',
                         'hour', 'minute', 'second', 'millisecond', 'errorType']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trial in trials:
                detailed_time = trial.get('detailedTime', {})
                now = datetime.now()
                
                row = {
                    'timestamp': trial.get('timestamp', datetime.now().isoformat()),
                    'trial': trial.get('trial'),
                    'testType': test_type,
                    'hour': detailed_time.get('hour', now.hour),
                    'minute': detailed_time.get('minute', now.minute),
                    'second': detailed_time.get('second', now.second),
                    'millisecond': detailed_time.get('millisecond', int(now.microsecond / 1000)),
                }
                
                if test_type == 'stroop':
                    row.update({
                        'word': trial.get('word', ''),
                        'color': trial.get('color', ''),
                        'userAnswer': trial.get('userAnswer', ''),
                        'correct': trial.get('correct'),
                        'reactionTime': trial.get('reactionTime'),
                        'congruent': trial.get('congruent'),
                        'errorType': trial.get('errorType', ''),
                        'isGo': '',
                        'stimulusType': '',
                        'responded': ''
                    })
                elif test_type == 'gonogo':
                    row.update({
                        'word': '',
                        'color': '',
                        'userAnswer': '',
                        'correct': trial.get('correct'),
                        'reactionTime': trial.get('reactionTime') or '',
                        'congruent': '',
                        'errorType': trial.get('errorType', ''),
                        'isGo': trial.get('isGo'),
                        'stimulusType': trial.get('stimulusType', ''),
                        'responded': trial.get('responded')
                    })
                
                writer.writerow(row)
        
        return jsonify({
            'success': True, 
            'message': 'Session saved',
            'file': csv_file
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/list-sessions', methods=['GET'])
def list_sessions():
    """List all saved sessions"""
    try:
        sessions = []
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith('.csv'):
                    filepath = os.path.join(DATA_DIR, filename)
                    stat = os.stat(filepath)
                    sessions.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({'success': True, 'sessions': sessions}), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-csv/<filename>', methods=['GET'])
def download_csv(filename):
    """Download a CSV file"""
    try:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, mimetype='text/csv')
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Data will be saved to:", os.path.abspath(DATA_DIR))
    app.run(debug=True, port=5000)

