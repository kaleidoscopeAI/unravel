        result["target_language"] = task.get("target_language")
        result["source_task_id"] = task.get("source_task_id")
        
        if task.get("status") == "completed":
            result["mimicked_files"] = [os.path.basename(f) for f in task.get("mimicked_files", [])]
            result["mimicked_dir"] = task.get("mimicked_dir")
    
    if task.get("status") == "error" and "error" in task:
        result["error"] = task.get("error")
    
    return jsonify(result)

@app.route('/download/<task_id>/<path:file_type>')
def download_results(task_id, file_type):
    """Download results as a ZIP file"""
    # Find task in history
    task = None
    for t in task_history:
        if t.get("id") == task_id:
            task = t
            break
    
    if not task or task["status"] != "completed":
        return jsonify({"error": "Task not found or not completed"}), 404
    
    import zipfile
    import tempfile
    
    # Create a temporary file for the ZIP
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create a ZIP file
        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if task["type"] == "ingest":
                if file_type == "decompiled":
                    for file_path in task.get("decompiled_files", []):
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                elif file_type == "specs":
                    for file_path in task.get("spec_files", []):
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                elif file_type == "reconstructed":
                    # Get the directory containing reconstructed files
                    if task.get("reconstructed_files"):
                        recon_dir = os.path.dirname(task["reconstructed_files"][0])
                        for root, _, files in os.walk(recon_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, os.path.relpath(file_path, recon_dir))
            elif task["type"] == "mimic":
                if file_type == "mimicked":
                    # Get the directory containing mimicked files
                    mimic_dir = task.get("mimicked_dir")
                    if mimic_dir and os.path.exists(mimic_dir):
                        for root, _, files in os.walk(mimic_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                zipf.write(file_path, os.path.relpath(file_path, mimic_dir))
        
        # Determine download filename
        if task["type"] == "ingest":
            filename = f"{task.get('file_name', 'unknown')}_{file_type}.zip"
        else:
            filename = f"{task.get('target_language', 'unknown')}_mimicked.zip"
        
        # Send the file
        return send_from_directory(os.path.dirname(temp_path), 
                                  os.path.basename(temp_path),
                                  as_attachment=True,
                                  attachment_filename=filename)
    
    except Exception as e:
        logger.error(f"Error creating ZIP file: {str(e)}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": "Failed to create download file"}), 500

def start_server(work_dir=None, host="127.0.0.1", port=5000):
    """
    Start the web server
    
    Args:
        work_dir: Working directory for Kaleidoscope
        host: Host address to bind
        port: Port to listen on
    """
    global kaleidoscope, worker_thread, running
    
    # Initialize Kaleidoscope
    work_dir = work_dir or os.path.join(os.getcwd(), "kaleidoscope_workdir")
    kaleidoscope = KaleidoscopeCore(work_dir=work_dir)
    
    # Start worker thread
    running = True
    worker_thread = threading.Thread(target=worker_loop)
    worker_thread.daemon = True
    worker_thread.start()
    
    # Create HTML templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create static files directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
    os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)
    
    # Create index.html template if it doesn't exist
    index_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaleidoscope AI</title>
    <link rel="stylesheet" href="/static/css/styles.css">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">Kaleidoscope AI</a>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>Software Ingestion & Mimicry</h5>
                    </div>
                    <div class="card-body">
                        <div id="upload-section">
                            <h5>Upload Software for Analysis</h5>
                            <form id="upload-form" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="file" class="form-label">Select File</label>
                                    <input class="form-control" type="file" id="file" name="file">
                                    <div class="form-text">Upload binaries, JavaScript, Python, C/C++, or other supported file types.</div>
                                </div>
                                <button type="submit" class="btn btn-primary">Upload & Analyze</button>
                            </form>
                            <div id="upload-progress" class="progress mt-3 d-none">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                            </div>
                        </div>

                        <div id="task-section" class="mt-4 d-none">
                            <h5>Current Task</h5>
                            <div class="alert alert-info">
                                <div id="current-task-info">No task running</div>
                            </div>
                        </div>

                        <div id="mimic-section" class="mt-4 d-none">
                            <h5>Create Mimicked Version</h5>
                            <form id="mimic-form">
                                <div class="mb-3">
                                    <label for="language" class="form-label">Target Language</label>
                                    <select class="form-select" id="language" name="language">
                                        <option value="python">Python</option>
                                        <option value="javascript">JavaScript</option>
                                        <option value="cpp">C++</option>
                                        <option value="java">Java</option>
                                    </select>
                                </div>
                                <input type="hidden" id="source-task-id" name="task_id" value="">
                                <button type="submit" class="btn btn-success">Generate Mimicked Version</button>
                            </form>
                        </div>

                        <div id="result-section" class="mt-4 d-none">
                            <h5>Results</h5>
                            <div id="result-content"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Task History</h5>
                    </div>
                    <div class="card-body">
                        <ul id="task-history" class="list-group">
                            <li class="list-group-item">No tasks yet</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>""")
    
    # Create CSS file if it doesn't exist
    css_path = os.path.join(static_dir, "css", "styles.css")
    if not os.path.exists(css_path):
        with open(css_path, 'w') as f:
            f.write("""body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.navbar-brand {
    font-weight: bold;
    letter-spacing: 1px;
}

.card {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.card-header {
    background-color: #f1f8ff;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
}

.task-item {
    transition: all 0.3s ease;
}

.task-item:hover {
    background-color: #f1f8ff;
}

.task-item.active {
    background-color: #e2f0fd;
    border-color: #90c8f8;
}

.alert-processing {
    background-color: #fff8e1;
    border-color: #ffe082;
    color: #ff8f00;
}

.alert-completed {
    background-color: #e8f5e9;
    border-color: #a5d6a7;
    color: #2e7d32;
}

.alert-error {
    background-color: #ffebee;
    border-color: #ef9a9a;
    color: #c62828;
}""")
    
    # Create JavaScript file if it doesn't exist
    js_path = os.path.join(static_dir, "js", "main.js")
    if not os.path.exists(js_path):
        with open(js_path, 'w') as f:
            f.write("""document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const uploadProgress = document.getElementById('upload-progress');
    const taskSection = document.getElementById('task-section');
    const currentTaskInfo = document.getElementById('current-task-info');
    const mimicSection = document.getElementById('mimic-section');
    const mimicForm = document.getElementById('mimic-form');
    const sourceTaskId = document.getElementById('source-task-id');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    const taskHistory = document.getElementById('task-history');
    
    // State
    let statusPollInterval;
    let latestIngestTaskId = null;
    
    // Start polling for status
    startStatusPolling();
    
    // Event listeners
    uploadForm.addEventListener('submit', handleUpload);
    mimicForm.addEventListener('submit', handleMimic);
    
    // Functions
    function handleUpload(event) {
        event.preventDefault();
        
        const fileInput = document.getElementById('file');
        if (!fileInput.files.length) {
            alert('Please select a file to upload');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show progress
        uploadProgress.classList.remove('d-none');
        
        // Send request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Upload successful:', data);
                // Hide progress
                uploadProgress.classList.add('d-none');
                // Show task section
                taskSection.classList.remove('d-none');
                currentTaskInfo.textContent = `Processing ${fileInput.files[0].name}...`;
                // Store task ID
                latestIngestTaskId = data.task_id;
            } else {
                console.error('Upload failed:', data);
                alert('Upload failed: ' + data.error);
                uploadProgress.classList.add('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during upload');
            uploadProgress.classList.add('d-none');
        });
    }
    
    function handleMimic(event) {
        event.preventDefault();
        
        const language = document.getElementById('language').value;
        const taskId = sourceTaskId.value;
        
        if (!taskId) {
            alert('No source task selected');
            return;
        }
        
        // Send request
        fetch('/mimic', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                language: language,
                task_id: taskId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Mimic request successful:', data);
                // Show task section
                taskSection.classList.remove('d-none');
                currentTaskInfo.textContent = `Generating ${language} version...`;
            } else {
                console.error('Mimic request failed:', data);
                alert('Mimic request failed: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during mimic request');
        });
    }
    
    function startStatusPolling() {
        // Poll every 2 seconds
        statusPollInterval = setInterval(fetchStatus, 2000);
    }
    
    function fetchStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                updateTaskStatus(data);
                updateTaskHistory(data);
            })
            .catch(error => {
                console.error('Error fetching status:', error);
            });
    }
    
    function updateTaskStatus(data) {
        // Update current task info
        if (data.current) {
            taskSection.classList.remove('d-none');
            
            const task = data.current;
            let statusText = '';
            
            if (task.type === 'ingest') {
                statusText = `Processing ${task.file_name}... Status: ${task.status}`;
            } else if (task.type === 'mimic') {
                statusText = `Generating ${task.target_language} version... Status: ${task.status}`;
            }
            
            currentTaskInfo.textContent = statusText;
        } else {
            // No current task
            let foundCompleted = false;
            
            // Check if we have a completed ingest task
            if (data.history && data.history.length > 0) {
                for (const task of data.history) {
                    if (task.type === 'ingest' && task.status === 'completed') {
                        foundCompleted = true;
                        
                        // Show mimic section
                        mimicSection.classList.remove('d-none');
                        sourceTaskId.value = task.id;
                        
                        // Show result section
                        resultSection.classList.remove('d-none');
                        
                        // Fetch task details
                        fetch(`/task/${task.id}`)
                            .then(response => response.json())
                            .then(taskData => {
                                if (taskData.status === 'completed') {
                                    let resultHtml = `<div class="alert alert-success">
                                        <h6>Ingestion Results for ${taskData.file_name}</h6>
                                        <p>
                                            <strong>Decompiled Files:</strong> ${taskData.decompiled_files.length}<br>
                                            <strong>Specification Files:</strong> ${taskData.spec_files.length}<br>
                                            <strong>Reconstructed Files:</strong> ${taskData.reconstructed_files.length}
                                        </p>
                                        <div class="mt-2">
                                            <a href="/download/${task.id}/decompiled" class="btn btn-sm btn-outline-primary">Download Decompiled</a>
                                            <a href="/download/${task.id}/specs" class="btn btn-sm btn-outline-primary">Download Specs</a>
                                            <a href="/download/${task.id}/reconstructed" class="btn btn-sm btn-outline-primary">Download Reconstructed</a>
                                        </div>
                                    </div>`;
                                    
                                    resultContent.innerHTML = resultHtml;
                                }
                            });
                        
                        break;
                    }
                }
            }
            
            if (!foundCompleted) {
                mimicSection.classList.add('d-none');
                resultSection.classList.add('d-none');
            }
        }
    }
    
    function updateTaskHistory(data) {
        if (!data.history || data.history.length === 0) {
            taskHistory.innerHTML = '<li class="list-group-item">No tasks yet</li>';
            return;
        }
        
        // Clear history
        taskHistory.innerHTML = '';
        
        // Add tasks to history
        for (const task of data.history.reverse()) {
            const li = document.createElement('li');
            li.className = 'list-group-item task-item';
            if (task.id === latestIngestTaskId) {
                li.className += ' active';
            }
            
            let taskTypeText = task.type === 'ingest' ? 'Ingestion' : 'Mimicry';
            let statusBadgeClass = 'bg-secondary';
            
            if (task.status === 'completed') {
                statusBadgeClass = 'bg-success';
            } else if (task.status === 'error') {
                statusBadgeClass = 'bg-danger';
            } else if (task.status === 'processing') {
                statusBadgeClass = 'bg-warning';
            }
            
            let taskDetails = '';
            if (task.type === 'ingest') {
                taskDetails = task.file_name || '';
            } else if (task.type === 'mimic') {
                taskDetails = task.target_language || '';
            }
            
            li.innerHTML = `
                <div>
                    <span class="badge ${statusBadgeClass}">${task.status}</span>
                    <strong>${taskTypeText}</strong> ${taskDetails}
                </div>
                <small class="text-muted">
                    ${new Date(task.timestamp * 1000).toLocaleString()}
                </small>
            `;
            
            // Add click handler
            li.addEventListener('click', () => {
                // Fetch task details
                fetch(`/task/${task.id}`)
                    .then(response => response.json())
                    .then(taskData => {
                        // Update UI based on task type and status
                        if (taskData.type === 'ingest' && taskData.status === 'completed') {
                            // Show mimic section
                            mimicSection.classList.remove('d-none');
                            sourceTaskId.value = taskData.id;
                            latestIngestTaskId = taskData.id;
                            
                            // Show result section
                            resultSection.classList.remove('d-none');
                            
                            let resultHtml = `<div class="alert alert-success">
                                <h6>Ingestion Results for ${taskData.file_name}</h6>
                                <p>
                                    <strong>Decompiled Files:</strong> ${taskData.decompiled_files.length}<br>
                                    <strong>Specification Files:</strong> ${taskData.spec_files.length}<br>
                                    <strong>Reconstructed Files:</strong> ${taskData.reconstructed_files.length}
                                </p>
                                <div class="mt-2">
                                    <a href="/download/${task.id}/decompiled" class="btn btn-sm btn-outline-primary">Download Decompiled</a>
                                    <a href="/download/${task.id}/specs" class="btn btn-sm btn-outline-primary">Download Specs</a>
                                    <a href="/download/${task.id}/reconstructed" class="btn btn-sm btn-outline-primary">Download Reconstructed</a>
                                </div>
                            </div>`;
                            
                            resultContent.innerHTML = resultHtml;
                            
                            // Update active task
                            document.querySelectorAll('.task-item').forEach(item => {
                                item.classList.remove('active');
                            });
                            li.classList.add('active');
                        } else if (taskData.type === 'mimic' && taskData.status === 'completed') {
                            // Show result section
                            resultSection.classList.remove('d-none');
                            
                            let resultHtml = `<div class="alert alert-success">
                                <h6>Mimicry Results for ${taskData.target_language}</h6>
                                <p>
                                    <strong>Mimicked Files:</strong> ${taskData.mimicked_files.length}<br>
                                    <strong>Output Directory:</strong> ${taskData.mimicked_dir}
                                </p>
                                <div class="mt-2">
                                    <a href="/download/${task.id}/mimicked" class="btn btn-sm btn-outline-primary">Download Mimicked Files</a>
                                </div>
                            </div>`;
                            
                            resultContent.innerHTML = resultHtml;
                        } else if (taskData.status === 'error') {
                            // Show error
                            resultSection.classList.remove('d-none');
                            
                            let resultHtml = `<div class="alert alert-danger">
                                <h6>Error</h6>
                                <p>${taskData.error || 'An unknown error occurred'}</p>
                            </div>`;
                            
                            resultContent.innerHTML = resultHtml;
                        }
                    });
            });
            
            taskHistory.appendChild(li);
        }
    }
});""")
    
    # Run the Flask app
    logger.info(f"Starting web server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

def main():
    """Main entry point for the web interface"""
    parser = argparse.ArgumentParser(description="Kaleidoscope AI Web Interface")
    parser.add_argument("--work-dir", "-w", help="Working directory", default=None)
    parser.add_argument("--host", help="Host address to bind", default="127.0.0.1")
    parser.add_argument("--port", "-p", help="Port to listen on", type=int, default=5000)
    
    args = parser.parse_args()
    
    try:
        start_server(work_dir=args.work_dir, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Error in web server: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())#!/usr/bin/env python3
"""
Kaleidoscope AI Web Interface
============================
A simple web interface for the Kaleidoscope AI system that allows users
to interact with the software ingestion and mimicry system through a browser.

This provides a more user-friendly interface compared to the command-line
chatbot and makes the powerful features more accessible.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Web server imports
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Ensure the kaleidoscope_core module is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaleidoscope_core import KaleidoscopeCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_web.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), "uploads")
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload size
app.config['SECRET_KEY'] = str(uuid.uuid4())

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global state
kaleidoscope = None
task_queue = queue.Queue()
current_task = None
task_history = []
worker_thread = None
running = False

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    # Allow most executable and code file extensions
    allowed_extensions = {
        'exe', 'dll', 'so', 'dylib',  # Binaries
        'js', 'mjs',                  # JavaScript
        'py',                         # Python
        'c', 'cpp', 'h', 'hpp',       # C/C++
        'java', 'class', 'jar',       # Java
        'go',                         # Go
        'rs',                         # Rust
        'php',                        # PHP
        'rb',                         # Ruby
        'cs',                         # C#
        'asm', 's'                    # Assembly
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def worker_loop():
    """Background worker thread to process tasks"""
    global current_task, running
    
    while running:
        try:
            # Get task from queue
            task = task_queue.get(timeout=1.0)
            
            # Set as current task
            current_task = task
            current_task["status"] = "processing"
            
            # Process task
            if task["type"] == "ingest":
                worker_ingest(task)
            elif task["type"] == "mimic":
                worker_mimic(task)
            
            # Mark task as done
            task_queue.task_done()
            
            # Add to history
            task_history.append(task)
            
            # Limit history size
            if len(task_history) > 10:
                task_history.pop(0)
                
            # Clear current task
            current_task = None
            
        except queue.Empty:
            # No tasks in queue
            pass
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            
            # Update current task status
            if current_task:
                current_task["status"] = "error"
                current_task["error"] = str(e)
                
                # Add to history
                task_history.append(current_task)
                
                # Clear current task
                current_task = None

def worker_ingest(task):
    """
    Worker function to ingest software
    
    Args:
        task: Task information
    """
    try:
        # Ingest software
        file_path = task["file_path"]
        
        logger.info(f"Ingesting {file_path}...")
        
        # Run ingestion
        result = kaleidoscope.ingest_software(file_path)
        
        # Update task with result
        task.update(result)
        task["status"] = result["status"]
        
        logger.info(f"Ingestion completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)

def worker_mimic(task):
    """
    Worker function to mimic software
    
    Args:
        task: Task information
    """
    try:
        # Mimic software
        spec_files = task["spec_files"]
        target_language = task["target_language"]
        
        logger.info(f"Generating mimicked version in {target_language}...")
        
        # Run mimicry
        result = kaleidoscope.mimic_software(spec_files, target_language)
        
        # Update task with result
        task.update(result)
        task["status"] = result["status"]
        
        logger.info(f"Mimicry completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in mimicry: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for ingestion"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if file is allowed
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Create task
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "ingest",
        "file_path": file_path,
        "file_name": filename,
        "status": "queued",
        "timestamp": time.time()
    }
    
    # Add to queue
    task_queue.put(task)
    
    return jsonify({
        "status": "success",
        "message": f"File {filename} uploaded and queued for ingestion",
        "task_id": task_id
    })

@app.route('/mimic', methods=['POST'])
def mimic_software():
    """Handle request to mimic software"""
    data = request.json
    
    # Check required fields
    if not data or 'language' not in data or 'task_id' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get source task
    source_task = None
    for task in task_history:
        if task.get("id") == data['task_id']:
            source_task = task
            break
    
    if not source_task or source_task["type"] != "ingest" or source_task["status"] != "completed":
        return jsonify({"error": "Source task not found or not completed"}), 400
    
    # Check if we have specification files
    if "spec_files" not in source_task or not source_task["spec_files"]:
        return jsonify({"error": "No specification files available"}), 400
    
    # Validate target language
    target_language = data['language'].lower()
    valid_languages = ["python", "javascript", "c", "cpp", "c++", "java"]
    
    if target_language not in valid_languages:
        return jsonify({"error": f"Unsupported language: {target_language}"}), 400
    
    # Map language aliases
    if target_language in ["c++", "cpp"]:
        target_language = "cpp"
    
    # Create task
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "mimic",
        "source_task_id": data['task_id'],
        "spec_files": source_task["spec_files"],
        "target_language": target_language,
        "status": "queued",
        "timestamp": time.time()
    }
    
    # Add to queue
    task_queue.put(task)
    
    return jsonify({
        "status": "success",
        "message": f"Queued mimicry in {target_language}",
        "task_id": task_id
    })

@app.route('/status')
def get_status():
    """Get status of tasks"""
    # Prepare current task info if available
    current = None
    if current_task:
        current = {
            "id": current_task.get("id"),
            "type": current_task.get("type"),
            "status": current_task.get("status"),
            "file_name": current_task.get("file_name") if "file_name" in current_task else None,
            "target_language": current_task.get("target_language") if "target_language" in current_task else None,
            "timestamp": current_task.get("timestamp")
        }
    
    # Prepare task history info
    history = []
    for task in task_history:
        task_info = {
            "id": task.get("id"),
            "type": task.get("type"),
            "status": task.get("status"),
            "file_name": task.get("file_name") if "file_name" in task else None,
            "target_language": task.get("target_language") if "target_language" in task else None,
            "timestamp": task.get("timestamp")
        }
        
        # Add success counts for completed tasks
        if task.get("status") == "completed":
            if task.get("type") == "ingest":
                task_info["decompiled_count"] = len(task.get("decompiled_files", []))
                task_info["spec_count"] = len(task.get("spec_files", []))
                task_info["reconstructed_count"] = len(task.get("reconstructed_files", []))
            elif task.get("type") == "mimic":
                task_info["mimicked_count"] = len(task.get("mimicked_files", []))
                task_info["mimicked_dir"] = task.get("mimicked_dir")
        
        history.append(task_info)
    
    return jsonify({
        "current": current,
        "history": history
    })

@app.route('/task/<task_id>')
def get_task_details(task_id):
    """Get detailed information about a task"""
    # Find task in history or current task
    task = None
    if current_task and current_task.get("id") == task_id:
        task = current_task
    else:
        for t in task_history:
            if t.get("id") == task_id:
                task = t
                break
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # Prepare response based on task type and status
    result = {
        "id": task.get("id"),
        "type": task.get("type"),
        "status": task.get("status"),
        "timestamp": task.get("timestamp")
    }
    
    if task.get("type") == "ingest":
        result["file_name"] = task.get("file_name")
        result["file_path"] = task.get("file_path")
        
        if task.get("status") == "completed":
            result["decompiled_files"] = [os.path.basename(f) for f in task.get("decompiled_files", [])]
            result["spec_files"] = [os.path.basename(f) for f in task.get("spec_files", [])]
            result["reconstructed_files"] = [os.path.basename(f) for f in task.get("reconstructed_files", [])]
    
    elif task.get("type") == "mimic":
        result["target_language"] = task.get("target_language")