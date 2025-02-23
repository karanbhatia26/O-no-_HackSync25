from flask import Flask, jsonify, request 
from flask_cors import CORS
import os 
import bcrypt
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

#Supabase Credentials
supabase_url = "https://wywakgsxojthkmouzpur.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind5d2FrZ3N4b2p0aGttb3V6cHVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzMDcxMDYsImV4cCI6MjA1NTg4MzEwNn0.IObnIgcKmZ4ODVvM6JCK_-wITBnS9vG71PlR4tFLTjg"

supabase: Client = create_client(supabase_url, supabase_key)

@app.route('/')
def index():
    return "Hello World!"

#Registering Users and Admins
@app.route('/register-user', methods=['POST'])
def register_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    mobile = data.get("mobile")
    location = data.get("location")

    if not (username and password and mobile and location):
        return jsonify({"error": "All fields are mandatory"}), 400
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    #Insert into Supabase
    response = supabase.table("users").insert(
        {"username": username, "password": hashed_password, "mobile": mobile, "location": location}
    ).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500
    
    return jsonify({"message": "User registered successfully!"}), 201

    
@app.route('/register-admin', methods=['POST'])
def register_admin():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    mobile = data.get("mobile")
    location = data.get("location")

    if not (username and password and mobile and location):
        return jsonify({"error": "All fields are mandatory"}), 400
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    #Insert into Supabase
    response = supabase.table("administrators").insert(
        {"username": username, "password": hashed_password, "mobile": mobile, "location": location}
    ).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500
    
    return jsonify({"message": "Admin registered successfully!"}), 201

#Login for Users and Admins
@app.route("/login-user", methods=["POST"])
def user_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not (username and password):
        return jsonify({"error": "Username and password required"}), 400

    # Fetch user from Supabase
    response = supabase.table("users").select("id, username, password").eq("username", username).execute()
    
    if not response.get("data"):
        return jsonify({"error": "Invalid username or password"}), 401

    user = response["data"][0]

    # Check password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify({"message": "Login successful!", "user_id": user["id"]}), 200

@app.route("/login-admin", methods=["POST"])
def admin_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not (username and password):
        return jsonify({"error": "Username and password required"}), 400

    # Fetch user from Supabase
    response = supabase.table("administrators").select("id, username, password").eq("username", username).execute()
    
    if not response.get("data"):
        return jsonify({"error": "Invalid username or password"}), 401

    user = response["data"][0]

    # Check password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify({"message": "Login successful!", "user_id": user["id"]}), 200

#Create User
@app.route("/create-user", methods=["POST"])
def create_user():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    mobile = data.get("mobile")
    location = data.get("location")

    response = supabase.table("users").insert(
        {"username": username, "password": password,"mobile": mobile, "location": location}
    ).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500

    return jsonify({"message": "User created successfully!"}), 201

#Create Admin
@app.route("/create-admin", methods=["POST"])
def create_admin():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    mobile = data.get("mobile")
    location = data.get("location")

    response = supabase.table("administrators").insert(
        {"username": username, "password": password,"mobile": mobile, "location": location}
    ).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500

    return jsonify({"message": "User created successfully!"}), 201

#Read Users. (Users List)
@app.route("/get-users", methods=["GET"])
def get_users():
    response = supabase.table("users").select("*").execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500

    return jsonify(response["data"]), 200

#Update User.
@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.json
    updates = {k: v for k, v in data.items() if v}  # Only update non-null values

    response = supabase.table("users").update(updates).eq("id", user_id).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500

    return jsonify({"message": "User updated successfully!"}), 200

#Delete User.
@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    response = supabase.table("users").delete().eq("id", user_id).execute()

    if response.get("error"):
        return jsonify({"error": response["error"]["message"]}), 500

    return jsonify({"message": "User deleted successfully!"}), 200


if __name__ == "__main__":
    app.run(debug=True)