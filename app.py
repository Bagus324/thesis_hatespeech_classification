import logging
from flask import (
    Flask,
    render_template,
    url_for,
    request,
    session,
    redirect,
    send_file,
    jsonify,
)
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime, timedelta
from transformers import AutoTokenizer
from firebase_admin import credentials, firestore, auth
from static.model.mbert_skripsi import Multilabelmbert as MBERT
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
import os
import re
import pandas as pd
import firebase_admin
import sys
from collections import defaultdict
from datetime import datetime
import torch
# INIT AND LOAD .ENV
app = Flask(__name__)
app.secret_key = "SECRETKEY"
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
load_dotenv()

# SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

labels_wps = [
    "HS",
    "Abusive",
    "HS_Individual",
    "HS_Group",
    "HS_Religion",
    "HS_Race",
    "HS_Physical",
    "HS_Gender",
    "HS_Other",
    "HS_Weak",
    "HS_Moderate",
    "HS_Strong",
    "PS",
]

trained_model = MBERT.load_from_checkpoint(
    checkpoint_path="MODEL_PATH.ckpt", labels=labels_wps
)
trained_model.eval()
trained_model.freeze()
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-multilingual-uncased"
)

# Initialize Firebase
cred = credentials.Certificate("FIREBASE_CREDENTIAL_PATH.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, email, nama_user):
        self.id = id
        self.email = email
        self.nama_user = nama_user

    @staticmethod
    def get(user_id):
        user_ref = db.collection("users")
        query = user_ref.where("uid", "==", user_id).limit(1)
        user_doc = query.get()
        if user_doc:
            user_data = user_doc[0].to_dict()
            return User(id=user_data["uid"], email=user_data["email"], nama_user=user_data["nama"])
        return None


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.route("/debug")
def debug():
    return render_template("test.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a comment was provided
        if "komen-single" in request.form:
            komen = request.form["komen-single"]
            if komen == "":
                return jsonify({"error": "Tidak dapat prediksi kalimat kosong."}), 400
            confident_prediction, shy_prediction, actual_prediction, confident_probability = predictor(
                komen, labels_wps
            )
            stringed_actual_prediction = ",".join(str(x) for x in actual_prediction)
            try:
                current_date = datetime.now().strftime("%Y-%m-%d")
                doc_ref = db.collection("data_komen").document()
                sample_date = '2024-04-17'
                doc_ref.set(
                    {
                        "komentar": komen,
                        "tanggal": current_date,
                        "labels": stringed_actual_prediction,
                    }
                )
                doc_ref = db.collection("visit").document('visit_count')
                doc_visit = doc_ref.get()
                if doc_visit.exists:
                    doc_visit_data = doc_visit.to_dict()
                    if current_date in doc_visit_data:
                        new_value = doc_visit_data[current_date] + 1
                    else:
                        new_value = 1
                    doc_ref.update({current_date: new_value})
                logger.info("Insert successful")
            except Exception as e:
                logger.error(f"Error inserting data: {e}")
            print(confident_probability)
            return render_template(
                "index.html",
                komen=komen,
                confident_prediction=confident_prediction,
                shy_prediction=shy_prediction,
                actual_prediction=actual_prediction,
                confident_probability=confident_probability,
                inference=True,
            )

        # Check if a file was uploaded
        if "file" in request.files:
            bytes_io = BytesIO()
            file = request.files["file"]
            if file.filename == '':
                return jsonify({"error": "Berkas belum diunggah."}), 400
            if file.filename.rsplit('.', 1)[1].lower() == 'csv':
                try:
                    df = pd.read_csv(file, header=None)
                except Exception as e:
                    return jsonify({"error": "Berkas yang diunggah adalah berkas kosong."}), 400
            else:
                xls_extension = file.filename.rsplit('.', 1)[1].lower()
                df = pd.read_excel(file, header=None)
            df = df.iloc[:, 0]
            df = df.dropna()
            df_list = df.values.tolist()
            predicted_labels = []
            confident_predictions, shy_predictions, actual_predictions = predictor_batched(
                df_list, labels_wps
            )
            for komentar, confident_prediction, shy_prediction, actual_prediction in zip(df_list, confident_predictions, shy_predictions, actual_predictions):

                stringed_actual_prediction = ",".join(str(x) for x in actual_prediction)
                try:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    sample_date = '2024-10-17'

                    doc_ref = db.collection("data_komen").document()
                    doc_ref.set(
                        {
                            "komentar": komentar,
                            "tanggal": current_date,
                            "labels": stringed_actual_prediction,
                        }
                    )
                    logger.info("Insert successful")
                except Exception as e:
                    logger.error(f"Error inserting data: {e}")
                # stringed_predicted_labels = ", ".join(str(x) for x in confident_prediction)
                stringed_predicted_labels = ""
                if "HS" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian, "
                if "Abusive" in confident_prediction:
                    stringed_predicted_labels += "Kalimat Kasar, "
                if "HS_Individual" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Individu, "
                if "HS_Group" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Suatu Kelompok, "
                if "HS_Religion" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Agama, "
                if "HS_Race" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Ras, "
                if "HS_Physical" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Fisik, "
                if "HS_Gender" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Terhadap Gender Tertentu, "
                if "HS_Other" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Lainnya"
                if "HS_Weak" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Lemah"
                if "HS_Moderate" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Sedang"
                if "HS_Strong" in confident_prediction:
                    stringed_predicted_labels += "Ujaran Kebencian Kuat"
                if "PS" in confident_prediction:
                    stringed_predicted_labels += "Positif/Netral"
                predicted_labels.append(stringed_predicted_labels)
            new_df = pd.DataFrame(columns=['Kalimat/Komentar', 'Prediksi Ujaran Kebencian'])
            new_df['Kalimat/Komentar'] = df_list
            new_df['Prediksi Ujaran Kebencian'] = predicted_labels
            

            doc_ref = db.collection("visit").document('visit_count')
            doc_visit = doc_ref.get()
            if doc_visit.exists:
                doc_visit_data = doc_visit.to_dict()
                if current_date in doc_visit_data:
                    new_value = doc_visit_data[current_date] + len(df_list)
                else:
                    new_value = len(df_list)
                doc_ref.update({current_date: new_value})
            bytes_io = BytesIO()


            if file.filename.rsplit('.', 1)[1].lower() == 'csv':
                new_df.to_csv(bytes_io, index=False)
                bytes_io.seek(0)
                return send_file(
                    bytes_io,
                    mimetype="text/csv",
                    as_attachment=True,
                    download_name="Processed_file.csv",
                )
            else:
                new_df.to_excel(bytes_io, index=False)
                bytes_io.seek(0)
                return send_file(
                    bytes_io,
                    mimetype="text/"+xls_extension,
                    as_attachment=True,
                    download_name="processed_file."+xls_extension,
                )

    return render_template("index.html")

@app.route("/d", methods=["GET", "POST"])
def download_example():
    return send_file("static/public/contoh_data.csv", as_attachment=True, download_name="Contoh.csv")




@app.route("/login", methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    elif request.method == "POST":
        try:
            username = request.form["username"]
            password = request.form["password"]
            user_fetch = auth.get_user_by_email(username)

            user_from_get = load_user(user_fetch.uid)

            login_user(user_from_get)
            session["status"] = True
            session['nama_user'] = user_from_get.nama_user

            return redirect(url_for("dashboard"))
        except Exception as e:
            return render_template("login.html", error=True)
    return render_template("login.html")


@app.route("/dashboard/charts", methods=["GET"])
@login_required
def dashboard():
    data = select_data_monthly()
    return render_template("dashboard.html", session=session, data=data)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard/datas")
@login_required
def datas():
    data_komen_ref = db.collection("data_komen")
    data = data_komen_ref.stream()
    data_list = [doc.to_dict() for doc in data]
    return render_template("tables.html", datas=data_list, session=session)


def select_data_monthly():
    current_year = datetime.now().year
    doc_ref = db.collection("visit").document('visit_count')
    doc_visit = doc_ref.get()
    doc_visit_data = doc_visit.to_dict()
    # Initialize a new dictionary with default value 0
    monthly_visits = defaultdict(int)

    # Get the current year
    current_year = datetime.now().year

    # Iterate over your dictionary
    for date, visits in doc_visit_data.items():
        # Parse the date string into a datetime object
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if the year of the date is the current year
        if date_obj.year == current_year:
            # Extract the year and month as a string
            year_month = date_obj.strftime('%Y-%m')
            
            # Add the visits to the corresponding month
            monthly_visits[year_month] += int(visits)

    # Create a list for all months of the current year
    all_months = [f"{current_year}-{str(month).zfill(2)}" for month in range(1, 13)]

    # Create a list of visit counts, defaulting to 0 for months without visits
    visit_counts = [monthly_visits.get(month, 0) for month in all_months]
    # data_komen_ref = db.collection("data_komen")
    # data = (
    #     data_komen_ref.where("tanggal", ">=", f"{current_year}-01-01")
    #     .where("tanggal", "<=", f"{current_year}-12-31")
    #     .stream()
    # )
    # monthly_data = [0] * 12
    # for doc in data:
    #     doc_data = doc.to_dict()
    #     month = int(doc_data["tanggal"].split("-")[1]) - 1
    #     monthly_data[month] += 1
    return visit_counts


def clean_str(string):
    string = re.sub(r"\bRT\b", " ", string)
    string = re.sub(r"\bUSER\b", " ", string)
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'t", " ", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = string.strip()
    return string


import torch

def predictor(comment, labels, threshold=0.5):
    test_comment = clean_str(comment)
    encoding = tokenizer.encode_plus(
        test_comment,
        add_special_tokens=True,
        max_length=100,
        return_token_type_ids=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Move tensors to CPU
    input_ids = encoding["input_ids"].cpu()
    token_type_ids = encoding["token_type_ids"].cpu()
    attention_mask = encoding["attention_mask"].cpu()

    # Move model to CPU
    trained_model.to("cpu")

    # Perform inference
    test_prediction = trained_model(input_ids, token_type_ids, attention_mask)
    test_prediction = test_prediction.cpu().flatten().numpy()
    
    confident_predict, shy_predict, actual_predict = [], [], []
    confident_probability = {}
    logger.info(f"Prediction for {test_comment}: {test_prediction}")
    
    for label, prediction in zip(labels, test_prediction):
        if prediction < threshold:
            shy_predict.append(label)
            actual_predict.append(0)
            continue
        confident_predict.append(label)
        actual_predict.append(1)
        confident_probability[label] = str(int(round(prediction * 100))) + "%"
        
    return confident_predict, shy_predict, actual_predict, confident_probability


def predictor_batched(comments, labels, threshold=0.5):
    inputs = tokenizer.batch_encode_plus(
        [clean_str(comment) for comment in comments],
        add_special_tokens=True,
        max_length=100,
        return_token_type_ids=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Move tensors to CPU
    input_ids = inputs["input_ids"].cpu()
    token_type_ids = inputs["token_type_ids"].cpu()
    attention_mask = inputs["attention_mask"].cpu()

    # Move model to CPU
    trained_model.to("cpu")

    # Perform inference
    with torch.no_grad():
        test_predictions = trained_model(input_ids, token_type_ids, attention_mask)
    test_predictions = test_predictions.cpu().detach().numpy()

    confident_predicts, shy_predicts, actual_predicts = [], [], []
    for test_prediction in test_predictions:
        confident_predict, shy_predict, actual_predict = [], [], []
        for label, prediction in zip(labels, test_prediction):
            if prediction < threshold:
                shy_predict.append(label)
                actual_predict.append(0)
            else:
                confident_predict.append(label)
                actual_predict.append(1)
        confident_predicts.append(confident_predict)
        shy_predicts.append(shy_predict)
        actual_predicts.append(actual_predict)

    return confident_predicts, shy_predicts, actual_predicts


@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413


if __name__ == "__main__":
    app.run(debug=True)
