import os
import uuid
import logging
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from neural_style import load_cnn, run_style_transfer

# Конфигурация
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
STYLE_FOLDER = 'static/models'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("server.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Стили — имена файлов в папке models/
STYLES = {
    'van_gogh': 'van_gogh.jpg',
    'picasso': 'picasso.jpg',
    'dali': 'dali.jpg',
    'munch': 'munch.jpg',
    'klimt': 'klimt.jpg',
    'chagall': 'chagall.jpg',
    'mucha': 'mucha.jpg'
}

# Словарь прогресс задач: task_id -> {'progress': 0..100, 'status': 'running'|'done'|'error', 'result': path or None, 'message': str}
tasks = {}

# Предзагрузим vgg (при старте)
logger.info("Loading VGG model at app startup...")
cnn_bundle = load_cnn()  # возвращает (cnn, normalization_mean, normalization_std) или аналог
logger.info("VGG loaded.")

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return '.' in filename and ext in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html', styles=STYLES.keys())

@app.route('/start', methods=['POST'])
def start():
    """
    Обрабатывает AJAX-запрос. Возвращает JSON с task_id.
    """
    if 'content_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['content_image']
    style_name = request.form.get('style')
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or invalid extension'}), 400
    if style_name not in STYLES:
        return jsonify({'error': 'Invalid style selected'}), 400

    filename = secure_filename(file.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    file.save(content_path)

    style_path = os.path.join(app.config['STYLE_FOLDER'], STYLES[style_name])
    output_name = f"styled_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_name)

    task_id = uuid.uuid4().hex
    tasks[task_id] = {'progress': 0, 'status': 'running', 'result': None, 'message': None}
    logger.info(f"Task {task_id} started. content={content_path}, style={style_path}")

    # Запускаем обработку в отдельном потоке (работа начнётся немедленно)
    def worker():
        try:
            def progress_cb(step, total):
                # step: текущая итерация, total: общее число итераций
                pct = int((step / total) * 100)
                tasks[task_id]['progress'] = pct
                logger.info(f"Task {task_id}: progress {pct}% ({step}/{total})")

            # Запуск NST — передаём предзагруженный cnn_bundle и callback
            run_style_transfer(content_path, style_path, output_path,
                               cnn_bundle=cnn_bundle,
                               num_steps=60,
                               progress_callback=progress_cb)
            tasks[task_id]['status'] = 'done'
            tasks[task_id]['result'] = output_path
            tasks[task_id]['progress'] = 100
            logger.info(f"Task {task_id} finished. output={output_path}")
        except Exception as e:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['message'] = str(e)
            logger.exception(f"Task {task_id} failed: {e}")

    Thread(target=worker, daemon=True).start()

    # Возвращаем task_id клиенту
    return jsonify({'task_id': task_id})

@app.route('/stream/<task_id>')
def stream(task_id):
    """
    SSE endpoint: отдаёт события прогресса в формате text/event-stream.
    Клиент должен подключиться EventSource('/stream/<task_id>')
    """
    from flask import Response, stream_with_context
    if task_id not in tasks:
        return "Unknown task", 404

    def event_stream():
        last_sent = -1
        while True:
            t = tasks.get(task_id)
            if not t:
                break
            # Отправляем прогресс если изменился
            if t['progress'] != last_sent:
                last_sent = t['progress']
                yield f"event: progress\n"
                yield f"data: {t['progress']}\n\n"
            # Если задача завершена — отправим финальное событие и выйдем
            if t['status'] == 'done':
                yield f"event: done\n"
                # возвращаем относительный путь для <img src=...>
                rel = t['result'].replace('\\', '/')
                yield f"data: {rel}\n\n"
                break
            if t['status'] == 'error':
                yield f"event: error\n"
                yield f"data: {t['message']}\n\n"
                break
            import time
            time.sleep(0.5)

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route('/result/<task_id>')
def result_page(task_id):
    t = tasks.get(task_id)
    if not t:
        return "Task not found", 404
    if t['status'] != 'done':
        return "Task not finished", 400
    # Отдаём страницу с результатом
    return render_template('result.html', result_image=t['result'])

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
