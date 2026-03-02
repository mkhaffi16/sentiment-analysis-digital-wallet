from app import app
from app import server as ml
from flask import render_template, json, request, jsonify
import pandas as pd
import numpy as np
from pandas import DataFrame
from werkzeug.utils import secure_filename
import os
import csv

from sklearn.preprocessing import LabelEncoder

# Instansiasi Kelas
load_data = ml.LoadData()
prep = ml.Preprocess()
xgb = ml.XGBoostAlgorithm()
evl = ml.Evaluasi()


# Akhir Inisialisasi Kelas


# ini lokasi file baik untuk disimpan maupun untuk di buka
# seusaikan dengan direktori mu
DATASET_LOC = 'app/static/file/dataset/'
PREDICT_LOC = 'app/static/file/predict/'
PREPROSES_LOC = 'app/static/file/preproses/'

ALLOWED_EXTENSIONS = {'csv'}


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def training():
    if request.method == "POST":

        file = request.files["trainDataset"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(DATASET_LOC, filename))
        else:
            return jsonify(message='error'), 500

        #! Letak Proses ML
        loc_train = os.path.join(DATASET_LOC, filename)
        train_df = load_data.loadData(loc_train)

        train_df = prep.preproses_data(train_df)

        locHPrep = load_data.savedata(
            PREPROSES_LOC, train_df, "preprosestrain.csv")


        X_train,Y_train = prep.Word2Vec(train_df)
        model = xgb.train(X_train, Y_train)
        xgb.savemodel(model)
        #! END Letak Proses ML

        hasil = {"message": f"{file.filename} uploaded",
                 "hPrep": {locHPrep}
                 }

        res = json.dumps(hasil, default=set_default), 200
        return res
    return render_template("index.html")


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":

        file = request.files["testDataset"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(DATASET_LOC, filename))
        else:
            return jsonify(message='error'), 500

        # Get Input Form
        loc_test = os.path.join(DATASET_LOC, filename)
        test_df = load_data.loadData(loc_test)
        # END Get Input
        
        #! Letak Proses Test
        model = xgb.loadmodel("finalized_model.sav")
        test_df = prep.preproses_data(test_df)
        
        X_test,Y_test = prep.Word2Vec(test_df)

        predictions = xgb.test(X_test, model)

        result = pd.DataFrame(
            {"content": test_df['content'], 
             "Preprocess":test_df['Preprocess'], 
             "CustomerService_prediksi": predictions[:,0],
             "FiturAplikasi_prediksi": predictions[:,1],
             "UserExperience_prediksi": predictions[:,2],
             "Verifikasi_prediksi": predictions[:,3] })

        # def categorise(row):
        #     if sum([row['CustomerService_prediksi'],row['FiturAplikasi_prediksi'],row['UserExperience_prediksi'],row['Verifikasi_prediksi']]) == 0:
        #         return 'Bukan SARA'
        #     else:
        #         return 'SARA'
        
        # result['keterangan'] = result.apply(lambda row: categorise(row), axis=1)
        
        locpred = load_data.savedata(PREDICT_LOC, result, "hasilprediksi.csv")

        reportCustomerService_df = pd.DataFrame(evl.report(Y_test['CustomerService'],predictions[:,0])).transpose()
        reportFiturAplikasi_df = pd.DataFrame(evl.report(Y_test['FiturAplikasi'],predictions[:,1])).transpose()
        reportUserExperience_df = pd.DataFrame(evl.report(Y_test['UserExperience'],predictions[:,2])).transpose()
        reportVerifikasi_df = pd.DataFrame(evl.report(Y_test['Verifikasi'],predictions[:,3])).transpose()

        reportCustomerService_df = reportCustomerService_df.reset_index()
        reportCustomerService_df['precision'] = reportCustomerService_df['precision'].round(2)
        reportCustomerService_df['recall'] = reportCustomerService_df['recall'].round(2)
        reportCustomerService_df['f1-score'] = reportCustomerService_df['f1-score'].round(2)
        reportCustomerService_df['support'] = reportCustomerService_df['support'].round(2)

        reportFiturAplikasi_df = reportFiturAplikasi_df.reset_index()
        reportFiturAplikasi_df['precision'] = reportFiturAplikasi_df['precision'].round(2)
        reportFiturAplikasi_df['recall'] = reportFiturAplikasi_df['recall'].round(2)
        reportFiturAplikasi_df['f1-score'] = reportFiturAplikasi_df['f1-score'].round(2)
        reportFiturAplikasi_df['support'] = reportFiturAplikasi_df['support'].round(2)

        reportUserExperience_df = reportUserExperience_df.reset_index()
        reportUserExperience_df['precision'] = reportUserExperience_df['precision'].round(2)
        reportUserExperience_df['recall'] = reportUserExperience_df['recall'].round(2)
        reportUserExperience_df['f1-score'] = reportUserExperience_df['f1-score'].round(2)
        reportUserExperience_df['support'] = reportUserExperience_df['support'].round(2)

        reportVerifikasi_df = reportVerifikasi_df.reset_index()
        reportVerifikasi_df['precision'] = reportVerifikasi_df['precision'].round(2)
        reportVerifikasi_df['recall'] = reportVerifikasi_df['recall'].round(2)
        reportVerifikasi_df['f1-score'] = reportVerifikasi_df['f1-score'].round(2)
        reportVerifikasi_df['support'] = reportVerifikasi_df['support'].round(2)


        
        locreport_CustomerService = load_data.savedata(PREDICT_LOC,
                                  reportCustomerService_df, "report_CustomerService.csv")
        locreport_FiturAplikasi = load_data.savedata(PREDICT_LOC,
                                  reportFiturAplikasi_df, "report_FiturAplikasi.csv")
        locreport_UserExperience = load_data.savedata(PREDICT_LOC,
                                  reportUserExperience_df, "report_UserExperience.csv")  
        locreport_Verifikasi = load_data.savedata(PREDICT_LOC,
                                  reportVerifikasi_df, "report_Verifikasi.csv")                           

        heatmap_CustomerService = evl.cm(Y_test['CustomerService'],predictions[:,0],"CustomerService","Confussion Matrix Customer Service")
        heatmap_FiturAplikasi = evl.cm(Y_test['FiturAplikasi'],predictions[:,1],"FiturAplikasi","Confussion Matrix Fitur Aplikasi")
        heatmap_UserExperience = evl.cm(Y_test['UserExperience'],predictions[:,2],"UserExperience","Confussion Matrix User Experience")
        heatmap_Verifikasi = evl.cm(Y_test['Verifikasi'],predictions[:,3],"Verifikasi","Confussion Matrix Verifikasi")
        

        #! END Letak Proses Test

        hasil = {"message": f"{file.filename} uploaded",
                 "prediction": {locpred},
                 "report_CustomerService": {locreport_CustomerService},
                 "report_FiturAplikasi": {locreport_FiturAplikasi},
                 "report_UserExperience": {locreport_UserExperience},
                 "report_Verifikasi": {locreport_Verifikasi},
                 "hm_CustomerService": {heatmap_CustomerService},
                 "hm_FiturAplikasi": {heatmap_FiturAplikasi},
                 "hm_UserExperience": {heatmap_UserExperience},
                 "hm_Verifikasi": {heatmap_Verifikasi},
                 }

        res = json.dumps(hasil, default=set_default), 200
        return res
    return render_template("index.html")


@app.route("/hpreproses", methods=["GET"])
def tampil():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREPROSES_LOC + 'preprosestrain.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)


@app.route("/hpResult", methods=["GET"])
def tampilResult():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun

    with open(PREDICT_LOC + 'hasilprediksi.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)


@app.route("/hpReportCustomerService", methods=["GET"])
def tampilReport1():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREDICT_LOC + 'report_CustomerService.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)

@app.route("/hpReportFiturAplikasi", methods=["GET"])
def tampilReport2():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREDICT_LOC + 'report_FiturAplikasi.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)

@app.route("/hpReportUserExperience", methods=["GET"])
def tampilReport3():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREDICT_LOC + 'report_UserExperience.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)

@app.route("/hpReportVerifikasi", methods=["GET"])
def tampilReport4():
    data = []

    # ini mengambil file hasil prediksi
    # jadi file hasil prediksi mu di eksport dulu ke csv
    # tentukan nama filenya jangan yang berubah-ubah mis "hasilPrediksi" atau yang lain
    # tapi kalau mau bikin yang dinamis jg Gaskeun
    with open(PREDICT_LOC + 'report_Verifikasi.csv', encoding='utf-8') as csvfile:
        data_csv = csv.DictReader(csvfile, delimiter=',')
        for row in data_csv:
            data.append(dict(row))
    data = {"data": data}
    return jsonify(data)

@app.route("/realtimetext", methods=["GET", "POST"])
def realtimetext():
    if request.method == "POST":
        n_input = [str(request.form['realtime_input'])]

        kelas_ = [0,0,0,0]
        input_df = pd.DataFrame(
            {"content":n_input, "CustomerService":kelas_[0],"FiturAplikasi":kelas_[1],"UserExperience":kelas_[2],"Verifikasi":kelas_[3]})
        
        
        
        input_df = prep.preproses_data(input_df)
        X_test,Y_test = prep.Word2Vec(input_df)

        
        model = xgb.loadmodel("finalized_model.sav")
        text_pred = xgb.test(X_test, model)

        model = xgb.loadmodel("finalized_model.sav")

        text_pred = np.squeeze(text_pred)


        output_pred = [0,0,0,0]
        for i in range(4):
            if text_pred[i] == -1:
                output_pred[i] = "Negatif"
            elif text_pred[i] == 0:
                output_pred[i] = "Netral"
            else:
                output_pred[i] = "Positive"
        
        kalimat_pred = input_df.content[0]
        
        hasil = {
                 "kalimat_pred": {kalimat_pred},
                 "CustomerService_pred": {output_pred[0]},
                 "FiturAplikasi_pred": {output_pred[1]},
                 "UserExperience_pred": {output_pred[2]},
                 "Verifikasi_pred": {output_pred[3]}
                 }

        res = json.dumps(hasil, default=set_default), 200
        return res
    return render_template("index.html")





