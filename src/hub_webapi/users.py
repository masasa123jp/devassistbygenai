# users.py
from flask import Blueprint, request, jsonify
from utils import execute_function, handle_error, validate_request_data, get_db_connection
from psycopg2.extras import RealDictCursor

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
