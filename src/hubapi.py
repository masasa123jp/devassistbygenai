# users.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
users_bp = Blueprint("users", __name__)

# ------------------------------------------------
# APIエンドポイント: ユーザー作成
# ------------------------------------------------
@users_bp.route("/users", methods=["POST"])
def create_user():
    """新しいユーザーを作成"""
    data = request.json
    required_fields = ["email", "password", "username", "role"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_user", [
        data["email"], data["password"], data["username"], data["role"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "User created successfully", "user": result}), 201

# ------------------------------------------------
# APIエンドポイント: ユーザー取得
# ------------------------------------------------
@users_bp.route("/users/<email>", methods=["GET"])
def get_user(email):
    """指定したメールのユーザーを取得"""
    result, error = execute_function("genai01.get_user", [email])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("User not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: ユーザー情報の更新
# ------------------------------------------------
@users_bp.route("/users/<email>", methods=["PUT"])
def update_user(email):
    """指定したメールのユーザー情報を更新"""
    data = request.json
    required_fields = ["password", "username", "role"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_user", [
        email, data["password"], data["username"], data["role"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "User updated successfully", "user": result}), 200

# ------------------------------------------------
# APIエンドポイント: ユーザー削除
# ------------------------------------------------
@users_bp.route("/users/<email>", methods=["DELETE"])
def delete_user(email):
    """指定したメールのユーザーを削除"""
    result, error = execute_function("genai01.delete_user", [email])
    if error:
        return handle_error(error)
    return jsonify({"message": "User deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべてのユーザーを取得
# ------------------------------------------------
@users_bp.route("/users/all", methods=["GET"])
def get_all_users():
    """すべてのユーザー情報を取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_all_users", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"users": result}), 200

# ------------------------------------------------
# APIエンドポイント: ユーザー検索
# ------------------------------------------------
@users_bp.route("/users/search", methods=["GET"])
def search_users():
    """ユーザーを検索"""
    username = request.args.get("username")
    role = request.args.get("role")
    requested_by = request.args.get("requested_by")  # 操作者のIDを取得

    result, error = execute_function("genai01.search_users", [username, role, requested_by])
    if error:
        return handle_error(error)
    return jsonify({"users": result}), 200



# utils.py でサポートする共通関数
# 1. バリデーション関数
def validate_request_data(data, required_fields):
    """リクエストデータのバリデーション"""
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, None

# 2. エラーハンドリング関数
def handle_error(error_msg, status_code=400):
    """エラーメッセージを統一フォーマットで返却"""
    return jsonify({"error": {"message": error_msg, "code": status_code}}), status_code

# 3. データベース関数実行
def execute_function(func_name, params):
    """PostgreSQL関数を実行し、結果を返す"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT * FROM {func_name}({','.join(['%s'] * len(params))})", params)
        result = cur.fetchall() if cur.description else None
        conn.commit()
        return result, None
    except Exception as e:
        return None, str(e)
    finally:
        cur.close()
        conn.close()

# sessions.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
sessions_bp = Blueprint("sessions", __name__)

# ------------------------------------------------
# APIエンドポイント: セッション作成
# ------------------------------------------------
@sessions_bp.route("/sessions", methods=["POST"])
def create_session():
    """新しいセッションを作成"""
    data = request.json
    required_fields = ["user_id", "session_string", "ip_address", "user_agent"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_session", [
        data["user_id"], data["session_string"], data["ip_address"], data["user_agent"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Session created successfully", "session": result}), 201

# ------------------------------------------------
# APIエンドポイント: セッション取得
# ------------------------------------------------
@sessions_bp.route("/sessions/<user_id>", methods=["GET"])
def get_session(user_id):
    """指定したユーザーIDのセッションを取得"""
    result, error = execute_function("genai01.get_session", [user_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Session not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: セッション更新
# ------------------------------------------------
@sessions_bp.route("/sessions/<user_id>", methods=["PUT"])
def update_session(user_id):
    """指定したユーザーIDのセッションを更新"""
    data = request.json
    required_fields = ["session_string"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_session", [
        user_id, data["session_string"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Session updated successfully", "session": result}), 200

# ------------------------------------------------
# APIエンドポイント: セッション削除
# ------------------------------------------------
@sessions_bp.route("/sessions/<user_id>", methods=["DELETE"])
def delete_session(user_id):
    """指定したユーザーIDのセッションを削除"""
    result, error = execute_function("genai01.delete_session", [user_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Session deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべてのセッションを取得
# ------------------------------------------------
@sessions_bp.route("/sessions/all", methods=["GET"])
def get_all_sessions():
    """すべてのセッション情報を取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_all_sessions", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"sessions": result}), 200



# projects.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
projects_bp = Blueprint("projects", __name__)

# ------------------------------------------------
# APIエンドポイント: プロジェクト作成
# ------------------------------------------------
@projects_bp.route("/projects", methods=["POST"])
def create_project():
    """新しいプロジェクトを作成"""
    data = request.json
    required_fields = ["user_id", "project_name"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_project", [
        data["user_id"], data["project_name"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Project created successfully", "project": result}), 201

# ------------------------------------------------
# APIエンドポイント: プロジェクト取得
# ------------------------------------------------
@projects_bp.route("/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):
    """指定したプロジェクトを取得"""
    result, error = execute_function("genai01.get_project_by_id", [project_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Project not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: プロジェクト更新
# ------------------------------------------------
@projects_bp.route("/projects/<int:project_id>", methods=["PUT"])
def update_project(project_id):
    """指定したプロジェクトを更新"""
    data = request.json
    required_fields = ["project_name"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_project", [
        project_id, data["project_name"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Project updated successfully", "project": result}), 200

# ------------------------------------------------
# APIエンドポイント: プロジェクト削除
# ------------------------------------------------
@projects_bp.route("/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id):
    """指定したプロジェクトを削除"""
    result, error = execute_function("genai01.delete_project", [project_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Project deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべてのプロジェクトを取得
# ------------------------------------------------
@projects_bp.route("/projects/all", methods=["GET"])
def get_all_projects():
    """すべてのプロジェクト情報を取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_all_projects", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"projects": result}), 200

# rag_data.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
rag_data_bp = Blueprint("rag_data", __name__)

# ------------------------------------------------
# APIエンドポイント: RAGデータ作成
# ------------------------------------------------
@rag_data_bp.route("/rag_data", methods=["POST"])
def create_rag_data():
    """新しいRAGデータを作成"""
    data = request.json
    required_fields = ["user_id", "index_data", "vector_data"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_rag_data", [
        data["user_id"], data["index_data"], data["vector_data"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "RAG data created successfully", "rag_data": result}), 201

# ------------------------------------------------
# APIエンドポイント: RAGデータ取得
# ------------------------------------------------
@rag_data_bp.route("/rag_data/<int:rag_id>", methods=["GET"])
def get_rag_data_by_id(rag_id):
    """指定したRAGデータを取得"""
    result, error = execute_function("genai01.get_rag_data_by_id", [rag_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("RAG data not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: RAGデータ更新
# ------------------------------------------------
@rag_data_bp.route("/rag_data/<int:rag_id>", methods=["PUT"])
def update_rag_data(rag_id):
    """指定したRAGデータを更新"""
    data = request.json
    required_fields = ["index_data", "vector_data"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_rag_data", [
        rag_id, data["index_data"], data["vector_data"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "RAG data updated successfully", "rag_data": result}), 200

# ------------------------------------------------
# APIエンドポイント: RAGデータ削除
# ------------------------------------------------
@rag_data_bp.route("/rag_data/<int:rag_id>", methods=["DELETE"])
def delete_rag_data(rag_id):
    """指定したRAGデータを削除"""
    result, error = execute_function("genai01.delete_rag_data", [rag_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "RAG data deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべてのRAGデータを取得
# ------------------------------------------------
@rag_data_bp.route("/rag_data/all", methods=["GET"])
def get_all_rag_data():
    """すべてのRAGデータを取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_all_rag_data", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"rag_data": result}), 200

# audit_logs.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error

# Blueprintの登録
audit_logs_bp = Blueprint("audit_logs", __name__)

# ------------------------------------------------
# APIエンドポイント: 監査ログ取得
# ------------------------------------------------
@audit_logs_bp.route("/audit_logs", methods=["GET"])
def get_audit_logs():
    """監査ログを取得"""
    start_date = request.args.get("start_date")  # 例: '2024-01-01'
    end_date = request.args.get("end_date")      # 例: '2024-12-31'

    result, error = execute_function("genai01.get_audit_logs", [start_date, end_date])
    if error:
        return handle_error(error)
    return jsonify({"audit_logs": result}), 200

# ------------------------------------------------
# APIエンドポイント: 古い監査ログ削除
# ------------------------------------------------
@audit_logs_bp.route("/audit_logs/cleanup", methods=["DELETE"])
def delete_old_audit_logs():
    """古い監査ログを削除"""
    retention_period = request.args.get("retention_period", 30)  # デフォルトは30日
    result, error = execute_function("genai01.delete_old_audit_logs", [retention_period])
    if error:
        return handle_error(error)
    return jsonify({"message": "Old audit logs deleted successfully"}), 200

# app.py
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)

if __name__ == "__main__":
    app.run(debug=True)

# feedback.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
feedback_bp = Blueprint("feedback", __name__)

# ------------------------------------------------
# APIエンドポイント: フィードバック作成
# ------------------------------------------------
@feedback_bp.route("/feedback", methods=["POST"])
def create_feedback():
    """新しいフィードバックを作成"""
    data = request.json
    required_fields = ["user_id", "project_id", "content"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_feedback", [
        data["user_id"], data["project_id"], data["content"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Feedback created successfully", "feedback": result}), 201

# ------------------------------------------------
# APIエンドポイント: フィードバック取得
# ------------------------------------------------
@feedback_bp.route("/feedback/<int:feedback_id>", methods=["GET"])
def get_feedback(feedback_id):
    """指定したフィードバックを取得"""
    result, error = execute_function("genai01.get_feedback", [feedback_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Feedback not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: フィードバック削除
# ------------------------------------------------
@feedback_bp.route("/feedback/<int:feedback_id>", methods=["DELETE"])
def delete_feedback(feedback_id):
    """指定したフィードバックを削除"""
    result, error = execute_function("genai01.delete_feedback", [feedback_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Feedback deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: フィードバック更新
# ------------------------------------------------
@feedback_bp.route("/feedback/<int:feedback_id>", methods=["PUT"])
def update_feedback(feedback_id):
    """指定したフィードバックを更新"""
    data = request.json
    required_fields = ["content"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_feedback", [
        feedback_id, data["content"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Feedback updated successfully", "feedback": result}), 200

# ai_models.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
ai_models_bp = Blueprint("ai_models", __name__)

# ------------------------------------------------
# APIエンドポイント: AIモデル作成
# ------------------------------------------------
@ai_models_bp.route("/ai_models", methods=["POST"])
def create_ai_model():
    """新しいAIモデルを作成"""
    data = request.json
    required_fields = ["model_name", "description", "api_type", "version_info", "usage_price"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_ai_model", [
        data["model_name"], data["description"], data["api_type"],
        data["version_info"], data["usage_price"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "AI model created successfully", "ai_model": result}), 201

# ------------------------------------------------
# APIエンドポイント: AIモデル取得
# ------------------------------------------------
@ai_models_bp.route("/ai_models/<int:model_id>", methods=["GET"])
def get_ai_model(model_id):
    """指定したAIモデルを取得"""
    result, error = execute_function("genai01.get_ai_model_by_id", [model_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("AI model not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: AIモデル削除
# ------------------------------------------------
@ai_models_bp.route("/ai_models/<int:model_id>", methods=["DELETE"])
def delete_ai_model(model_id):
    """指定したAIモデルを削除"""
    result, error = execute_function("genai01.delete_ai_model", [model_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "AI model deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: AIモデル更新
# ------------------------------------------------
@ai_models_bp.route("/ai_models/<int:model_id>", methods=["PUT"])
def update_ai_model(model_id):
    """指定したAIモデルを更新"""
    data = request.json
    required_fields = ["model_name", "description", "api_type", "version_info", "usage_price"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_ai_model", [
        model_id, data["model_name"], data["description"],
        data["api_type"], data["version_info"], data["usage_price"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "AI model updated successfully", "ai_model": result}), 200



# app.py への Blueprint 登録
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp
from hub_webapi.feedback import feedback_bp
from hub_webapi.ai_models import ai_models_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(ai_models_bp)

if __name__ == "__main__":
    app.run(debug=True)

# generated_texts.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
generated_texts_bp = Blueprint("generated_texts", __name__)

# ------------------------------------------------
# APIエンドポイント: 生成テキスト作成
# ------------------------------------------------
@generated_texts_bp.route("/generated_texts", methods=["POST"])
def create_generated_text():
    """新しい生成テキストを作成"""
    data = request.json
    required_fields = ["user_id", "model_id", "content"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_generated_text", [
        data["user_id"], data["model_id"], data["content"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Generated text created successfully", "generated_text": result}), 201

# ------------------------------------------------
# APIエンドポイント: 生成テキスト取得
# ------------------------------------------------
@generated_texts_bp.route("/generated_texts/<int:text_id>", methods=["GET"])
def get_generated_text(text_id):
    """指定した生成テキストを取得"""
    result, error = execute_function("genai01.get_generated_text_by_id", [text_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Generated text not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: 生成テキスト削除
# ------------------------------------------------
@generated_texts_bp.route("/generated_texts/<int:text_id>", methods=["DELETE"])
def delete_generated_text(text_id):
    """指定した生成テキストを削除"""
    result, error = execute_function("genai01.delete_generated_text", [text_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Generated text deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべての生成テキストを取得
# ------------------------------------------------
@generated_texts_bp.route("/generated_texts/all", methods=["GET"])
def get_all_generated_texts():
    """すべての生成されたテキストを取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_all_generated_texts", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"generated_texts": result}), 200



# system_settings.py

from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
system_settings_bp = Blueprint("system_settings", __name__)

# ------------------------------------------------
# APIエンドポイント: システム設定作成
# ------------------------------------------------
@system_settings_bp.route("/system_settings", methods=["POST"])
def create_system_setting():
    """新しいシステム設定を作成"""
    data = request.json
    required_fields = ["setting_name", "setting_value", "description"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.create_system_setting", [
        data["setting_name"], data["setting_value"], data["description"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "System setting created successfully", "system_setting": result}), 201

# ------------------------------------------------
# APIエンドポイント: システム設定取得
# ------------------------------------------------
@system_settings_bp.route("/system_settings/<setting_name>", methods=["GET"])
def get_system_setting(setting_name):
    """指定したシステム設定を取得"""
    result, error = execute_function("genai01.get_setting_by_name", [setting_name])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("System setting not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: システム設定削除
# ------------------------------------------------
@system_settings_bp.route("/system_settings/<setting_name>", methods=["DELETE"])
def delete_system_setting(setting_name):
    """指定したシステム設定を削除"""
    result, error = execute_function("genai01.delete_system_setting", [setting_name])
    if error:
        return handle_error(error)
    return jsonify({"message": "System setting deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: システム設定更新
# ------------------------------------------------
@system_settings_bp.route("/system_settings/<setting_name>", methods=["PUT"])
def update_system_setting(setting_name):
    """システム設定を更新"""
    data = request.json
    required_fields = ["setting_value", "description"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    # 実行
    result, error = execute_function("genai01.update_system_setting", [
        setting_name, data["setting_value"], data["description"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "System setting updated successfully", "system_setting": result}), 200



# app.py への Blueprint 登録
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp
from hub_webapi.feedback import feedback_bp
from hub_webapi.ai_models import ai_models_bp
from hub_webapi.generated_texts import generated_texts_bp
from hub_webapi.system_settings import system_settings_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(ai_models_bp)
app.register_blueprint(generated_texts_bp)
app.register_blueprint(system_settings_bp)

if __name__ == "__main__":
    app.run(debug=True)

# access_tokens.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error

# Blueprintの登録
access_tokens_bp = Blueprint("access_tokens", __name__)

# ------------------------------------------------
# APIエンドポイント: アクセストークン作成
# ------------------------------------------------
@access_tokens_bp.route("/access_tokens", methods=["POST"])
def create_access_token():
    """新しいアクセストークンを作成"""
    data = request.json
    user_id = data.get("user_id")
    if not user_id:
        return handle_error("User ID is required")

    result, error = execute_function("genai01.create_access_token", [user_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Access token created successfully", "access_token": result}), 201

# ------------------------------------------------
# APIエンドポイント: アクセストークン削除
# ------------------------------------------------
@access_tokens_bp.route("/access_tokens/<token_id>", methods=["DELETE"])
def delete_access_token(token_id):
    """指定したアクセストークンを削除"""
    result, error = execute_function("genai01.delete_access_token", [token_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Access token deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: アクセストークン取得
# ------------------------------------------------
@access_tokens_bp.route("/access_tokens", methods=["GET"])
def get_access_tokens():
    """すべてのアクセストークンを取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.get_access_tokens", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"access_tokens": result}), 200



# batch_jobs.py

from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
batch_jobs_bp = Blueprint("batch_jobs", __name__)

# ------------------------------------------------
# APIエンドポイント: バッチジョブ作成
# ------------------------------------------------
@batch_jobs_bp.route("/batch_jobs", methods=["POST"])
def create_batch_job():
    """新しいバッチジョブを作成"""
    data = request.json
    required_fields = ["job_name", "job_type", "parameters"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_batch_job", [
        data["job_name"], data["job_type"], data["parameters"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Batch job created successfully", "batch_job": result}), 201

# ------------------------------------------------
# APIエンドポイント: バッチジョブ取得
# ------------------------------------------------
@batch_jobs_bp.route("/batch_jobs/<int:job_id>", methods=["GET"])
def get_batch_job(job_id):
    """指定したバッチジョブを取得"""
    result, error = execute_function("genai01.get_batch_job", [job_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Batch job not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: バッチジョブ更新
# ------------------------------------------------
@batch_jobs_bp.route("/batch_jobs/<int:job_id>", methods=["PUT"])
def update_batch_job(job_id):
    """指定したバッチジョブを更新"""
    data = request.json
    required_fields = ["parameters"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.update_batch_job", [
        job_id, data["parameters"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Batch job updated successfully", "batch_job": result}), 200

# ------------------------------------------------
# APIエンドポイント: バッチジョブ削除
# ------------------------------------------------
@batch_jobs_bp.route("/batch_jobs/<int:job_id>", methods=["DELETE"])
def delete_batch_job(job_id):
    """指定したバッチジョブを削除"""
    result, error = execute_function("genai01.delete_batch_job", [job_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Batch job deleted successfully"}), 200



# feature_flags.py

from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error

# Blueprintの登録
feature_flags_bp = Blueprint("feature_flags", __name__)

# ------------------------------------------------
# APIエンドポイント: フィーチャーフラグ一覧取得
# ------------------------------------------------
@feature_flags_bp.route("/feature_flags", methods=["GET"])
def get_feature_flags():
    """すべてのフィーチャーフラグを取得"""
    result, error = execute_function("genai01.get_all_feature_flags", [])
    if error:
        return handle_error(error)
    return jsonify({"feature_flags": result}), 200

# ------------------------------------------------
# APIエンドポイント: フィーチャーフラグ状態変更
# ------------------------------------------------
@feature_flags_bp.route("/feature_flags/<feature_name>/toggle", methods=["POST"])
def toggle_feature_flag(feature_name):
    """フィーチャーフラグの状態をトグル"""
    toggled_by = request.json.get("toggled_by")  # 操作者のID
    if not toggled_by:
        return handle_error("Toggled by is required")

    result, error = execute_function("genai01.update_feature_flag_status", [feature_name, toggled_by])
    if error:
        return handle_error(error)
    return jsonify({"message": f"Feature flag '{feature_name}' toggled successfully"}), 200



# app.py に新規モジュールを登録
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp
from hub_webapi.feedback import feedback_bp
from hub_webapi.ai_models import ai_models_bp
from hub_webapi.generated_texts import generated_texts_bp
from hub_webapi.system_settings import system_settings_bp
from hub_webapi.access_tokens import access_tokens_bp
from hub_webapi.batch_jobs import batch_jobs_bp
from hub_webapi.feature_flags import feature_flags_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(ai_models_bp)
app.register_blueprint(generated_texts_bp)
app.register_blueprint(system_settings_bp)
app.register_blueprint(access_tokens_bp)
app.register_blueprint(batch_jobs_bp)
app.register_blueprint(feature_flags_bp)

if __name__ == "__main__":
    app.run(debug=True)

# scheduled_jobs.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
scheduled_jobs_bp = Blueprint("scheduled_jobs", __name__)

# ------------------------------------------------
# APIエンドポイント: スケジュールジョブ作成
# ------------------------------------------------
@scheduled_jobs_bp.route("/scheduled_jobs", methods=["POST"])
def create_scheduled_job():
    """新しいスケジュールジョブを作成"""
    data = request.json
    required_fields = ["job_name", "schedule", "parameters"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_scheduled_job", [
        data["job_name"], data["schedule"], data["parameters"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Scheduled job created successfully", "scheduled_job": result}), 201

# ------------------------------------------------
# APIエンドポイント: スケジュールジョブ取得
# ------------------------------------------------
@scheduled_jobs_bp.route("/scheduled_jobs/<int:job_id>", methods=["GET"])
def get_scheduled_job(job_id):
    """指定したスケジュールジョブを取得"""
    result, error = execute_function("genai01.get_scheduled_job", [job_id])
    if error:
        return handle_error(error)
    if not result:
        return handle_error("Scheduled job not found", 404)
    return jsonify(result), 200

# ------------------------------------------------
# APIエンドポイント: スケジュールジョブ更新
# ------------------------------------------------
@scheduled_jobs_bp.route("/scheduled_jobs/<int:job_id>", methods=["PUT"])
def update_scheduled_job(job_id):
    """指定したスケジュールジョブを更新"""
    data = request.json
    required_fields = ["schedule", "parameters"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.update_scheduled_job", [
        job_id, data["schedule"], data["parameters"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Scheduled job updated successfully", "scheduled_job": result}), 200

# ------------------------------------------------
# APIエンドポイント: スケジュールジョブ削除
# ------------------------------------------------
@scheduled_jobs_bp.route("/scheduled_jobs/<int:job_id>", methods=["DELETE"])
def delete_scheduled_job(job_id):
    """指定したスケジュールジョブを削除"""
    result, error = execute_function("genai01.delete_scheduled_job", [job_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Scheduled job deleted successfully"}), 200

# ------------------------------------------------
# APIエンドポイント: すべてのスケジュールジョブを取得
# ------------------------------------------------
@scheduled_jobs_bp.route("/scheduled_jobs/all", methods=["GET"])
def get_all_scheduled_jobs():
    """すべてのスケジュールジョブを取得"""
    result, error = execute_function("genai01.get_all_scheduled_jobs", [])
    if error:
        return handle_error(error)
    return jsonify({"scheduled_jobs": result}), 200



# statistics.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error

# Blueprintの登録
statistics_bp = Blueprint("statistics", __name__)

# ------------------------------------------------
# APIエンドポイント: モデルごとの生成テキスト数を取得
# ------------------------------------------------
@statistics_bp.route("/statistics/texts_by_model", methods=["GET"])
def count_texts_by_model():
    """モデルごとの生成テキスト数を取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.count_generated_texts_by_model", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"texts_by_model": result}), 200

# ------------------------------------------------
# APIエンドポイント: プロジェクトごとのフィードバック数を取得
# ------------------------------------------------
@statistics_bp.route("/statistics/feedback_by_project", methods=["GET"])
def count_feedback_by_project():
    """プロジェクトごとのフィードバック数を取得"""
    requested_by = request.args.get("requested_by")  # 操作者のIDをリクエストから取得

    result, error = execute_function("genai01.count_feedback_by_project", [requested_by])
    if error:
        return handle_error(error)
    return jsonify({"feedback_by_project": result}), 200

# ------------------------------------------------
# APIエンドポイント: システム全体の利用状況を取得
# ------------------------------------------------
@statistics_bp.route("/statistics/usage", methods=["GET"])
def get_system_usage_statistics():
    """システム全体の利用状況統計を取得"""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    result, error = execute_function("genai01.get_usage_statistics", [start_date, end_date])
    if error:
        return handle_error(error)
    return jsonify({"usage_statistics": result}), 200

# app.py に新規モジュールを登録
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp
from hub_webapi.feedback import feedback_bp
from hub_webapi.ai_models import ai_models_bp
from hub_webapi.generated_texts import generated_texts_bp
from hub_webapi.system_settings import system_settings_bp
from hub_webapi.access_tokens import access_tokens_bp
from hub_webapi.batch_jobs import batch_jobs_bp
from hub_webapi.feature_flags import feature_flags_bp
from hub_webapi.scheduled_jobs import scheduled_jobs_bp
from hub_webapi.statistics import statistics_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(ai_models_bp)
app.register_blueprint(generated_texts_bp)
app.register_blueprint(system_settings_bp)
app.register_blueprint(access_tokens_bp)
app.register_blueprint(batch_jobs_bp)
app.register_blueprint(feature_flags_bp)
app.register_blueprint(scheduled_jobs_bp)
app.register_blueprint(statistics_bp)

if __name__ == "__main__":
    app.run(debug=True)

# logging.py（ログ記録機能）
import logging

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def log_info(message):
    """情報ログを記録"""
    logging.info(message)

def log_error(message):
    """エラーログを記録"""
    logging.error(message)

def log_warning(message):
    """警告ログを記録"""
    logging.warning(message)


# utils.py の関数でログを組み込む例：
from logging import log_info, log_error

def execute_function(func_name, params):
    """PostgreSQL関数を実行し、結果を返す"""
    try:
        log_info(f"Executing function: {func_name} with params: {params}")
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"SELECT * FROM {func_name}({','.join(['%s'] * len(params))})", params)
        result = cur.fetchall() if cur.description else None
        conn.commit()
        return result, None
    except Exception as e:
        log_error(f"Error executing function {func_name}: {str(e)}")
        return None, str(e)
    finally:
        cur.close()
        conn.close()

# error_logs.py（エラーログ管理機能）
from flask import Blueprint, jsonify
from logging import log_error

# Blueprintの登録
error_logs_bp = Blueprint("error_logs", __name__)

# ------------------------------------------------
# APIエンドポイント: 最新のエラーログを取得
# ------------------------------------------------
@error_logs_bp.route("/error_logs", methods=["GET"])
def get_error_logs():
    """最新のエラーログを取得"""
    try:
        with open("app.log", "r") as log_file:
            logs = log_file.readlines()
        error_logs = [log for log in logs if "ERROR" in log]
        return jsonify({"error_logs": error_logs}), 200
    except Exception as e:
        log_error(f"Failed to retrieve error logs: {str(e)}")
        return jsonify({"error": "Failed to retrieve error logs"}), 500


# health_check.py（リソースの健康チェック）
from flask import Blueprint, jsonify

# Blueprintの登録
health_check_bp = Blueprint("health_check", __name__)

# ------------------------------------------------
# APIエンドポイント: 健康チェック
# ------------------------------------------------
@health_check_bp.route("/health", methods=["GET"])
def health_check():
    """アプリケーションの健康状態を確認"""
    try:
        # データベース接続確認
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()

        # 健康チェックのレスポンス
        return jsonify({
            "status": "healthy",
            "database": "connected",
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }), 500

# app.py に新規モジュールを登録
from flask import Flask
from hub_webapi.users import users_bp
from hub_webapi.sessions import sessions_bp
from hub_webapi.projects import projects_bp
from hub_webapi.rag_data import rag_data_bp
from hub_webapi.audit_logs import audit_logs_bp
from hub_webapi.feedback import feedback_bp
from hub_webapi.ai_models import ai_models_bp
from hub_webapi.generated_texts import generated_texts_bp
from hub_webapi.system_settings import system_settings_bp
from hub_webapi.access_tokens import access_tokens_bp
from hub_webapi.batch_jobs import batch_jobs_bp
from hub_webapi.feature_flags import feature_flags_bp
from hub_webapi.scheduled_jobs import scheduled_jobs_bp
from hub_webapi.statistics import statistics_bp
from hub_webapi.error_logs import error_logs_bp
from hub_webapi.health_check import health_check_bp

app = Flask(__name__)

# 各Blueprintを登録
app.register_blueprint(users_bp)
app.register_blueprint(sessions_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(rag_data_bp)
app.register_blueprint(audit_logs_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(ai_models_bp)
app.register_blueprint(generated_texts_bp)
app.register_blueprint(system_settings_bp)
app.register_blueprint(access_tokens_bp)
app.register_blueprint(batch_jobs_bp)
app.register_blueprint(feature_flags_bp)
app.register_blueprint(scheduled_jobs_bp)
app.register_blueprint(statistics_bp)
app.register_blueprint(error_logs_bp)
app.register_blueprint(health_check_bp)

if __name__ == "__main__":
    app.run(debug=True)

# tags, text_templates, task_management, user_project, project_ai_model, rag_data_generated_text, user_ai_model, audit_logs, api_usage_logs, scheduled_jobs, user_preferences, user_sessions_auditについても順にお願いします。

# tags.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
tags_bp = Blueprint("tags", __name__)

# ------------------------------------------------
# APIエンドポイント: タグ作成
# ------------------------------------------------
@tags_bp.route("/tags", methods=["POST"])
def create_tag():
    """新しいタグを作成"""
    data = request.json
    required_fields = ["tag_name", "description"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_tag", [
        data["tag_name"], data["description"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Tag created successfully", "tag": result}), 201

# ------------------------------------------------
# APIエンドポイント: タグ一覧取得
# ------------------------------------------------
@tags_bp.route("/tags", methods=["GET"])
def get_tags():
    """すべてのタグを取得"""
    result, error = execute_function("genai01.get_tags", [])
    if error:
        return handle_error(error)
    return jsonify({"tags": result}), 200

# ------------------------------------------------
# APIエンドポイント: タグ削除
# ------------------------------------------------
@tags_bp.route("/tags/<int:tag_id>", methods=["DELETE"])
def delete_tag(tag_id):
    """指定したタグを削除"""
    result, error = execute_function("genai01.delete_tag", [tag_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Tag deleted successfully"}), 200

# text_templates.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
text_templates_bp = Blueprint("text_templates", __name__)

# ------------------------------------------------
# APIエンドポイント: テキストテンプレート作成
# ------------------------------------------------
@text_templates_bp.route("/text_templates", methods=["POST"])
def create_text_template():
    """新しいテキストテンプレートを作成"""
    data = request.json
    required_fields = ["template_name", "content"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_text_template", [
        data["template_name"], data["content"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Text template created successfully", "text_template": result}), 201

# ------------------------------------------------
# APIエンドポイント: テキストテンプレート一覧取得
# ------------------------------------------------
@text_templates_bp.route("/text_templates", methods=["GET"])
def get_text_templates():
    """すべてのテキストテンプレートを取得"""
    result, error = execute_function("genai01.get_text_templates", [])
    if error:
        return handle_error(error)
    return jsonify({"text_templates": result}), 200

# ------------------------------------------------
# APIエンドポイント: テキストテンプレート削除
# ------------------------------------------------
@text_templates_bp.route("/text_templates/<int:template_id>", methods=["DELETE"])
def delete_text_template(template_id):
    """指定したテキストテンプレートを削除"""
    result, error = execute_function("genai01.delete_text_template", [template_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Text template deleted successfully"}), 200

# task_management.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
tasks_bp = Blueprint("tasks", __name__)

# ------------------------------------------------
# APIエンドポイント: タスク作成
# ------------------------------------------------
@tasks_bp.route("/tasks", methods=["POST"])
def create_task():
    """新しいタスクを作成"""
    data = request.json
    required_fields = ["task_name", "project_id", "assigned_to"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_task", [
        data["task_name"], data["project_id"], data["assigned_to"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Task created successfully", "task": result}), 201

# ------------------------------------------------
# APIエンドポイント: タスク一覧取得
# ------------------------------------------------
@tasks_bp.route("/tasks", methods=["GET"])
def get_tasks():
    """すべてのタスクを取得"""
    result, error = execute_function("genai01.get_tasks", [])
    if error:
        return handle_error(error)
    return jsonify({"tasks": result}), 200

# ------------------------------------------------
# APIエンドポイント: タスク削除
# ------------------------------------------------
@tasks_bp.route("/tasks/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    """指定したタスクを削除"""
    result, error = execute_function("genai01.delete_task", [task_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Task deleted successfully"}), 200


# 1. tags.py: タグの管理（作成、取得、削除）
# 2. text_templates.py: テキストテンプレートの管理（作成、取得、削除）
# 3. task_management.py: タスクの管理（作成、取得、削除）

# user_project.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
user_project_bp = Blueprint("user_project", __name__)

# ------------------------------------------------
# APIエンドポイント: ユーザープロジェクト作成
# ------------------------------------------------
@user_project_bp.route("/user_projects", methods=["POST"])
def create_user_project():
    """新しいユーザープロジェクトを作成"""
    data = request.json
    required_fields = ["user_id", "project_id", "role"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_user_project", [
        data["user_id"], data["project_id"], data["role"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "User project created successfully", "user_project": result}), 201

# ------------------------------------------------
# APIエンドポイント: ユーザープロジェクト取得
# ------------------------------------------------
@user_project_bp.route("/user_projects", methods=["GET"])
def get_user_projects():
    """すべてのユーザープロジェクトを取得"""
    result, error = execute_function("genai01.get_user_projects", [])
    if error:
        return handle_error(error)
    return jsonify({"user_projects": result}), 200

# ------------------------------------------------
# APIエンドポイント: ユーザープロジェクト削除
# ------------------------------------------------
@user_project_bp.route("/user_projects/<int:user_project_id>", methods=["DELETE"])
def delete_user_project(user_project_id):
    """指定したユーザープロジェクトを削除"""
    result, error = execute_function("genai01.delete_user_project", [user_project_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "User project deleted successfully"}), 200



# project_ai_model.py

from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
project_ai_model_bp = Blueprint("project_ai_model", __name__)

# ------------------------------------------------
# APIエンドポイント: プロジェクトAIモデル作成
# ------------------------------------------------
@project_ai_model_bp.route("/project_ai_models", methods=["POST"])
def create_project_ai_model():
    """新しいプロジェクトAIモデルを作成"""
    data = request.json
    required_fields = ["project_id", "ai_model_id"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_project_ai_model", [
        data["project_id"], data["ai_model_id"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "Project AI model created successfully", "project_ai_model": result}), 201

# ------------------------------------------------
# APIエンドポイント: プロジェクトAIモデル取得
# ------------------------------------------------
@project_ai_model_bp.route("/project_ai_models", methods=["GET"])
def get_project_ai_models():
    """すべてのプロジェクトAIモデルを取得"""
    result, error = execute_function("genai01.get_project_ai_models", [])
    if error:
        return handle_error(error)
    return jsonify({"project_ai_models": result}), 200

# ------------------------------------------------
# APIエンドポイント: プロジェクトAIモデル削除
# ------------------------------------------------
@project_ai_model_bp.route("/project_ai_models/<int:project_ai_model_id>", methods=["DELETE"])
def delete_project_ai_model(project_ai_model_id):
    """指定したプロジェクトAIモデルを削除"""
    result, error = execute_function("genai01.delete_project_ai_model", [project_ai_model_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "Project AI model deleted successfully"}), 200



# rag_data_generated_text.py

from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
rag_data_generated_text_bp = Blueprint("rag_data_generated_text", __name__)

# ------------------------------------------------
# APIエンドポイント: RAGデータ生成テキスト作成
# ------------------------------------------------
@rag_data_generated_text_bp.route("/rag_data_generated_texts", methods=["POST"])
def create_rag_data_generated_text():
    """新しいRAGデータ生成テキストを作成"""
    data = request.json
    required_fields = ["rag_data_id", "generated_text_id"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_rag_data_generated_text", [
        data["rag_data_id"], data["generated_text_id"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "RAG data generated text created successfully", "rag_data_generated_text": result}), 201

# ------------------------------------------------
# APIエンドポイント: RAGデータ生成テキスト取得
# ------------------------------------------------
@rag_data_generated_text_bp.route("/rag_data_generated_texts", methods=["GET"])
def get_rag_data_generated_texts():
    """すべてのRAGデータ生成テキストを取得"""
    result, error = execute_function("genai01.get_rag_data_generated_texts", [])
    if error:
        return handle_error(error)
    return jsonify({"rag_data_generated_texts": result}), 200

# ------------------------------------------------
# APIエンドポイント: RAGデータ生成テキスト削除
# ------------------------------------------------
@rag_data_generated_text_bp.route("/rag_data_generated_texts/<int:rag_data_generated_text_id>", methods=["DELETE"])
def delete_rag_data_generated_text(rag_data_generated_text_id):
    """指定したRAGデータ生成テキストを削除"""
    result, error = execute_function("genai01.delete_rag_data_generated_text", [rag_data_generated_text_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "RAG data generated text deleted successfully"}), 200

# 1. user_project.py: ユーザープロジェクトの管理
# 2. project_ai_model.py: プロジェクトとAIモデルの関係管理
# 3. rag_data_generated_text.py: RAGデータと生成テキストの関係管理

# user_ai_model.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data

# Blueprintの登録
user_ai_model_bp = Blueprint("user_ai_model", __name__)

# ------------------------------------------------
# APIエンドポイント: ユーザーAIモデル作成
# ------------------------------------------------
@user_ai_model_bp.route("/user_ai_models", methods=["POST"])
def create_user_ai_model():
    """新しいユーザーAIモデルを作成"""
    data = request.json
    required_fields = ["user_id", "ai_model_id", "access_level"]

    # バリデーション
    is_valid, error = validate_request_data(data, required_fields)
    if not is_valid:
        return handle_error(error)

    result, error = execute_function("genai01.create_user_ai_model", [
        data["user_id"], data["ai_model_id"], data["access_level"]
    ])
    if error:
        return handle_error(error)
    return jsonify({"message": "User AI model created successfully", "user_ai_model": result}), 201

# ------------------------------------------------
# APIエンドポイント: ユーザーAIモデル取得
# ------------------------------------------------
@user_ai_model_bp.route("/user_ai_models", methods=["GET"])
def get_user_ai_models():
    """すべてのユーザーAIモデルを取得"""
    result, error = execute_function("genai01.get_user_ai_models", [])
    if error:
        return handle_error(error)
    return jsonify({"user_ai_models": result}), 200

# ------------------------------------------------
# APIエンドポイント: ユーザーAIモデル削除
# ------------------------------------------------
@user_ai_model_bp.route("/user_ai_models/<int:user_ai_model_id>", methods=["DELETE"])
def delete_user_ai_model(user_ai_model_id):
    """指定したユーザーAIモデルを削除"""
    result, error = execute_function("genai01.delete_user_ai_model", [user_ai_model_id])
    if error:
        return handle_error(error)
    return jsonify({"message": "User AI model deleted successfully"}), 200



# audit_logs.py

from flask import Blueprint, request, jsonify

# Blueprintの登録
audit_logs_bp = Blueprint("audit_logs", __name__)

# ------------------------------------------------
# APIエンドポイント: 監査ログ取得
# ------------------------------------------------
@audit_logs_bp.route("/audit_logs", methods=["GET"])
def get_audit_logs():
    """監査ログを取得"""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    result, error = execute_function("genai01.get_audit_logs", [start_date, end_date])
    if error:
        return handle_error(error)
    return jsonify({"audit_logs": result}), 200

# ------------------------------------------------
# APIエンドポイント: 古い監査ログ削除
# ------------------------------------------------
@audit_logs_bp.route("/audit_logs/cleanup", methods=["DELETE"])
def delete_old_audit_logs():
    """古い監査ログを削除"""
    retention_period = request.args.get("retention_period", 30)  # デフォルト30日
    result, error = execute_function("genai01.delete_old_audit_logs", [retention_period])
    if error:
        return handle_error(error)
    return jsonify({"message": "Old audit logs deleted successfully"}), 200



# api_usage_logs.py

from flask import Blueprint, request, jsonify

# Blueprintの登録
api_usage_logs_bp = Blueprint("api_usage_logs", __name__)

# ------------------------------------------------
# APIエンドポイント: API使用ログ取得
# ------------------------------------------------
@api_usage_logs_bp.route("/api_usage_logs", methods=["GET"])
def get_api_usage_logs():
    """API使用ログを取得"""
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    result, error = execute_function("genai01.get_api_usage_logs", [start_date, end_date])
    if error:
        return handle_error(error)
    return jsonify({"api_usage_logs": result}), 200

# ------------------------------------------------
# APIエンドポイント: API使用ログ削除
# ------------------------------------------------
@api_usage_logs_bp.route("/api_usage_logs/cleanup", methods=["DELETE"])
def delete_old_api_usage_logs():
    """古いAPI使用ログを削除"""
    retention_period = request.args.get("retention_period", 30)  # デフォルト30日
    result, error = execute_function("genai01.delete_old_api_usage_logs", [retention_period])
    if error:
        return handle_error(error)
    return jsonify({"message": "Old API usage logs deleted successfully"}), 200



# user_preferences.py

from flask import Blueprint, request, jsonify

# Blueprintの登録
user_preferences_bp = Blueprint("user_preferences", __name__)

# ------------------------------------------------
# APIエンドポイント: ユーザープリファレンス取得
# ------------------------------------------------
@user_preferences_bp.route("/user_preferences/<user_id>", methods=["GET"])
def get_user_preferences(user_id):
    """指定したユーザーのプリファレンスを取得"""
    result, error = execute_function("genai01.get_user_preferences", [user_id])
    if error:
        return handle_error(error)
    return jsonify({"user_preferences": result}), 200

# ------------------------------------------------
# APIエンドポイント: ユーザープリファレンス更新
# ------------------------------------------------
@user_preferences_bp.route("/user_preferences/<user_id>", methods=["PUT"])
def update_user_preferences(user_id):
    """指定したユーザーのプリファレンスを更新"""
    data = request.json
    preference_key = data.get("preference_key")
    preference_value = data.get("preference_value")

    if not preference_key or not preference_value:
        return handle_error("Preference key and value are required")

    result, error = execute_function("genai01.update_user_preferences", [user_id, preference_key, preference_value])
    if error:
        return handle_error(error)
    return jsonify({"message": "User preferences updated successfully"}), 200



# user_sessions_audit.py

from flask import Blueprint, request, jsonify

# Blueprintの登録
user_sessions_audit_bp = Blueprint("user_sessions_audit", __name__)

# ------------------------------------------------
# APIエンドポイント: ユーザーセッション監査ログ取得
# ------------------------------------------------
@user_sessions_audit_bp.route("/user_sessions_audit", methods=["GET"])
def get_user_sessions_audit():
    """すべてのユーザーセッション監査ログを取得"""
    result, error = execute_function("genai01.get_user_sessions_audit", [])
    if error:
        return handle_error(error)
    return jsonify({"user_sessions_audit": result}), 200

# ------------------------------------------------
# APIエンドポイント: 古いユーザーセッション監査ログ削除
# ------------------------------------------------
@user_sessions_audit_bp.route("/user_sessions_audit/cleanup", methods=["DELETE"])
def delete_old_user_sessions_audit():
    """古いユーザーセッション監査ログを削除"""
    retention_period = request.args.get("retention_period", 30)  # デフォルト30日
    result, error = execute_function("genai01.delete_old_user_sessions_audit", [retention_period])
    if error:
        return handle_error(error)
    return jsonify({"message": "Old user sessions audit deleted successfully"}), 200
