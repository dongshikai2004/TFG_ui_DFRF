from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import sqlite3
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-me')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.sqlite3')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT,
            ip_address TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # 自动创建管理员账号（可通过环境变量覆盖）
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    admin_password = os.getenv('ADMIN_PASSWORD', 'admin123456')

    cur.execute("SELECT id FROM users WHERE username = ?", (admin_username,))
    exists = cur.fetchone()
    if not exists:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, is_admin, created_at)
            VALUES (?, ?, 1, ?)
            """,
            (
                admin_username,
                generate_password_hash(admin_password),
                datetime.utcnow().isoformat()
            )
        )

    conn.commit()
    conn.close()


def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, username, is_admin, created_at FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    return user


def log_operation(action, detail=''):
    user = current_user()
    if not user:
        return
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO operation_logs (user_id, username, action, detail, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user['id'],
            user['username'],
            action,
            detail,
            request.remote_addr,
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect(url_for('login'))
        if int(user['is_admin']) != 1:
            flash('仅管理员可访问该页面。', 'error')
            return redirect(url_for('index'))
        return view_func(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_user():
    return {'current_user': current_user()}


# --------------------- 认证相关 ---------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if not username or not password:
            flash('用户名和密码不能为空。', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('用户名至少 3 个字符。', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('密码至少 6 位。', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('两次输入的密码不一致。', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            conn.close()
            flash('用户名已存在，请更换。', 'error')
            return render_template('register.html')

        conn.execute(
            """
            INSERT INTO users (username, password_hash, is_admin, created_at)
            VALUES (?, ?, 0, ?)
            """,
            (username, generate_password_hash(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        flash('注册成功，请登录。', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        conn = get_db_connection()
        user = conn.execute(
            "SELECT id, username, password_hash, is_admin FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        conn.close()

        if not user or not check_password_hash(user['password_hash'], password):
            flash('用户名或密码错误。', 'error')
            return render_template('login.html')

        session['user_id'] = user['id']
        log_operation('login', '用户登录成功')
        flash('登录成功。', 'success')
        if int(user['is_admin']) == 1:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    log_operation('logout', '用户退出登录')
    session.clear()
    flash('你已退出登录。', 'success')
    return redirect(url_for('login'))


# --------------------- 业务页面 ---------------------

# 首页
@app.route('/')
@login_required
def index():
    return render_template('index.html')


# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
@login_required
def video_generation():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "ref_video": request.form.get('ref_video'),
            "ref_audio": request.form.get('ref_audio'),
            "iter": request.form.get('iter'),
            "target_text": request.form.get('target_text'),
        }

        log_operation('video_generation_start', f"参数: model={data['model_name']}, iter={data['iter']}")
        video_path = generate_video(data)
        log_operation('video_generation_done', f"输出: {video_path}")
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')


# 模型训练界面
@app.route('/model_training', methods=['GET', 'POST'])
@login_required
def model_training():
    if request.method == 'POST':
        data = {
            "model_choice": request.form.get('model_choice'),
            "ref_video": request.form.get('ref_video'),
            "gpu_choice": request.form.get('gpu_choice'),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params')
        }

        log_operation('model_training_start', f"参数: model={data['model_choice']}, epoch={data['epoch']}")
        video_path = train_model(data)
        video_path = "/" + video_path.replace("\\", "/")
        log_operation('model_training_done', f"输出: {video_path}")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('model_training.html')


# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
@login_required
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "iter": request.form.get('iter'),
            "ref_video": request.form.get('ref_video'),
            "voice_clone": request.form.get('voice_clone'),
            "ref_audio": request.form.get('ref_audio'),
            "api_choice": request.form.get('api_choice'),
        }

        log_operation('chat_start', f"参数: model={data['model_name']}, api={data['api_choice']}")
        video_path = chat_response(data)
        video_path = "/" + video_path.replace("\\", "/")
        log_operation('chat_done', f"输出: {video_path}")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('chat_system.html')


@app.route('/save_audio', methods=['POST'])
@login_required
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})

    # 确保目录存在
    os.makedirs('./static/audios', exist_ok=True)

    # 保存文件
    audio_file.save('./static/audios/input.wav')
    log_operation('save_audio', '保存录音文件 ./static/audios/input.wav')

    return jsonify({'status': 'success', 'message': '音频保存成功'})


# --------------------- 管理员查看数据 ---------------------

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_connection()

    users = conn.execute(
        """
        SELECT id, username, is_admin, created_at
        FROM users
        ORDER BY id DESC
        """
    ).fetchall()

    logs = conn.execute(
        """
        SELECT id, user_id, username, action, detail, ip_address, created_at
        FROM operation_logs
        ORDER BY id DESC
        LIMIT 500
        """
    ).fetchall()

    conn.close()
    return render_template('admin_dashboard.html', users=users, logs=logs)


@app.route('/admin/api/data')
@admin_required
def admin_api_data():
    conn = get_db_connection()

    users = conn.execute(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY id DESC"
    ).fetchall()

    logs = conn.execute(
        """
        SELECT id, user_id, username, action, detail, ip_address, created_at
        FROM operation_logs
        ORDER BY id DESC
        LIMIT 500
        """
    ).fetchall()

    conn.close()

    return jsonify({
        "users": [dict(row) for row in users],
        "operation_logs": [dict(row) for row in logs]
    })


@app.before_request
def audit_page_views():
    # 避免静态资源/登录注册接口刷日志；仅记录已登录用户访问页面
    ignored_prefixes = ('/static/', '/admin/api/data')
    ignored_exact = {'/login', '/register'}
    if request.path.startswith(ignored_prefixes) or request.path in ignored_exact:
        return

    if session.get('user_id') and request.method == 'GET':
        log_operation('page_view', request.path)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
