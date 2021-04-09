import os
import pandas as pd
import copy
import logging
import datetime
import re
import json
import fasttext
from sklearn.model_selection import train_test_split
from collections import namedtuple
from flask import Flask, jsonify, request, url_for, render_template, json
from flask import render_template, Blueprint, make_response, request


app = Flask(__name__)

class PrefixMiddleware(object):
#class for URL sorting 
    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        #in this line I'm doing a replace of the word flaskredirect which is my app name in IIS to ensure proper URL redirect
        if environ['PATH_INFO'].lower().replace('/fasttext','').startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'].lower().replace('/fasttext','')[len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])            
            return ['This url does not belong to the app.'.encode()]

app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/api')

logging.basicConfig(filename='log.log',level=logging.DEBUG,format='%(asctime)s - %(process)d-%(levelname)s-%(message)s')

def preprocessing_text(text):
    text = text.lower()
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for i in punctuation:
        text = text.replace(i, ' ')
    # Eliminando tildes
    text = re.sub(r'[á]', 'a', text)
    text = re.sub(r'[é]', 'e', text)
    text = re.sub(r'[í]', 'i', text)
    text = re.sub(r'[ó]', 'o', text)
    text = re.sub(r'[ú]', 'u', text)
    text = re.sub(r'[à]', 'a', text)
    text = re.sub(r'[è]', 'e', text)
    text = re.sub(r'[ì]', 'i', text)
    text = re.sub(r'[ò]', 'o', text)
    text = re.sub(r'[ù]', 'u', text)
    text = re.sub(r'[â]', 'a', text)
    text = re.sub(r'[ê]', 'e', text)
    text = re.sub(r'[î]', 'i', text)
    text = re.sub(r'[ô]', 'o', text)
    text = re.sub(r'[û]', 'u', text)
    # Eliminando espacios dobles
    text = re.sub(r'\s+', ' ', text)
    return text 

def preprocessing_df(df, colx, coly):
    ### Preprocesamiento ###
    # Pasando a minúsculas el texto
    df[colx] = df[colx].str.lower()
    # Eliminando los signos de puntuacion
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for i in punctuation:
        df[colx] = df[colx].str.replace('\\' + i, ' ', regex=True)
    # Eliminando palabras irrelevantes    
    prepositions = ['a','ante','bajo','cabe','con','contra','de','desde','en','entre','hacia','hasta','para','por','según','sin','so','sobre','tras']
    prep_alike = ['durante','mediante','excepto','salvo','incluso','más','menos']
    adverbs = ['no','si','sí']
    articles = ['el','la','los','las','un','una','unos','unas','este','esta','estos','estas','aquel','aquella','aquellos','aquellas']
    aux_verbs = ['he','has','ha','hemos','habéis','han','había','habías','habíamos','habíais','habían']
    spanish_stop_words = set(prepositions+prep_alike+adverbs+articles+aux_verbs)
    # for i in spanish_stop_words:
    #   df[colx] = df[colx].replace(i, ' ')
    # Eliminando tildes
    df[colx] = df[colx].str.replace('[á]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[é]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[í]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ó]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[ú]', 'u', regex=True)
    df[colx] = df[colx].str.replace('[à]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[è]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[ì]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ò]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[ù]', 'u', regex=True)
    df[colx] = df[colx].str.replace('[â]', 'a', regex=True)
    df[colx] = df[colx].str.replace('[ê]', 'e', regex=True)
    df[colx] = df[colx].str.replace('[î]', 'i', regex=True)
    df[colx] = df[colx].str.replace('[ô]', 'o', regex=True)
    df[colx] = df[colx].str.replace('[û]', 'u', regex=True)
    # Eliminando espacios dobles
    df[colx] = df[colx].replace('\s+', ' ', regex=True)
    # Eliminando espacios iniciales y finales
    df[colx] = df[colx].str.strip()
    df[coly] = df[coly].str.replace('\"', '', regex=True)
    df[coly] = df[coly].str.replace('__label__', ' __label__')
    df[coly] = df[coly].str.strip()
    return df

def classifierMethod_fn(modelname, text, media='text'):
    model = fasttext.load_model('{}.bin'.format(modelname))
    texts = []
    if media == 'text':
        if type(text) == str:
            texts = [text]
        elif type(text) == list:
            texts = list(text)
    elif media == 'file':
        texts = text.readlines()
        texts = [t.decode('utf-8') for t in texts]
    result = []
    for line in texts:
        # line = preprocessing(line)
        # Quitando el enter del final de la linea
        line = str(line).replace('\r\n','')
        line = str(line).replace('\n','')
        # Ejecutando análisis
        analysis = model.predict(line.lower(), k=5)
        classes = []
        for i in range(5):
            classes.append({
                'confidence': analysis[1][i],
                'class_name': str(analysis[0][i]).replace('__label__','')
                })
        result.append({
                'text': line,
                'top_class': str(analysis[0][0]).replace('__label__',''),
                'confidence': analysis[1][0],
                'classes': classes
            })
    return result

def openFile(filename, fileextension, preprocess, formfile=None, split=None):
    if fileextension == 'csv':
        if formfile is None:
            df_data = pd.read_csv(os.path.join(os.getcwd(), '{}.{}'.format(filename, fileextension)))
        else:
            df_data = pd.read_csv(formfile)
    elif fileextension == 'txt':
        if formfile is None:
            df_data = pd.read_csv(os.path.join(os.getcwd(), '{}.{}'.format(filename, fileextension)))
        else:
            df_data = pd.read_csv(formfile)
            df_data.to_csv('{}IBM.csv'.format(filename), header=False, index=False)
            df_data.to_csv('{}.csv'.format(filename), sep='\t', header=False, index=False)
    elif fileextension == 'xlsx':
        if formfile is None:
            df_data = pd.read_excel(os.path.join(os.getcwd(), '{}.{}'.format(filename, fileextension)), engine='openpyxl')
        else:
            df_data = pd.read_excel(formfile, engine='openpyxl')
            df_data.to_csv('{}IBM.csv'.format(filename), header=False, index=False)
            df_data.to_csv('{}.csv'.format(filename), sep='\t', header=False, index=False)
    else:
        raise Exception('Archivo inválido.')
    df_data[df_data.keys()[1]] = df_data[df_data.keys()[1]].str.lower()
    df_data[df_data.keys()[1]] = df_data[df_data.keys()[1]].replace('\s+', '_', regex=True)
    if not str.__contains__(df_data[df_data.keys()[1]][0],'label'):
        df_data[df_data.keys()[1]] = '__label__' + df_data[df_data.keys()[1]]
    df_datawork = copy.copy(df_data)
    if preprocess == 'True':
        df_datawork = preprocessing_df(df_datawork, df_datawork.keys()[0], df_datawork.keys()[1])
        df_datawork.to_csv('{}IBM_preprocessed.csv'.format(filename), header=False, index=False)
        df_datawork.to_csv('{}_preprocessed.csv'.format(filename), sep='\t', header=False, index=False)
        result = '{}_preprocessed.csv'.format(filename)
    else:
        result = '{}.csv'.format(filename)
    if split == 'True':
        train, test, = train_test_split(df_datawork, test_size=0.2)
        train.to_csv('{}_trainset.csv'.format(filename), sep='\t', header=False, index=False)
        test.to_csv('{}_testset.csv'.format(filename), sep='\t', header=False, index=False)
    return result

@app.route('/models', methods=['GET','POST'])
def models_controller():
    try:
        result = [f for f in os.listdir() if f.endswith('.bin')]
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('ERROR: {}'.format(e))
        return jsonify(str(e)), 400

@app.route('/model', methods=['GET','POST'])
def model_controller():
    try:
        if (request.method == 'POST'):
            if (len(request.files) == 0):
                some_json = request.get_json() # request --> json
                preprocess = some_json['preprocess'] # boolean preprocess
                filename = some_json['filename'].rsplit(".",1)[0] # nombre del archivo local
                fileextension = some_json['filename'].rsplit(".",1)[1] # extensión del archivo local
                newfilename = openFile(filename, fileextension, preprocess)
            else:
                preprocess = request.form['preprocess'] # boolean preprocess
                formfile = request.files[''] # obtiene el archivo de formdata
                filename = formfile.filename.rsplit(".",1)[0] # nombre del archivo
                fileextension = formfile.filename.rsplit(".",1)[1] # extensión del archivo
                newfilename = openFile(filename, fileextension, preprocess, formfile)
        elif (request.method == 'GET'):
            preprocess = request.args.get('preprocess') # boolean preprocess
            filename = request.args.get('filename').rsplit(".",1)[0] # nombre del archivo local
            fileextension = request.args.get('filename').rsplit(".",1)[1] # extensión del archivo local
            newfilename = openFile(filename, fileextension, preprocess)
        # Entrenamiento del modelo
        ft = {
            'lr': 1.00,
            'dim': 100,
            'epoch': 250,
            'minCount': 1,
            'wordNgrams': 2
        }
        ft = namedtuple("FastTextParams", ft.keys())(*ft.values())
        model = fasttext.train_supervised(input='{}'.format(newfilename), lr=ft.lr, dim=ft.dim, epoch=ft.epoch, minCount=ft.minCount, wordNgrams=ft.wordNgrams)
        # Nombre del modelo
        modelname = '{}.bin'.format(newfilename)
        # Guardando modelo
        model.save_model(modelname)
        result = {}
        result['modelname'] = modelname
        result['labels'] = model.labels
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('ERROR: {}'.format(e))
        return jsonify(str(e)), 400

@app.route('/test', methods=['GET','POST'])
def test_controller():
    try:
        if (request.method == 'POST'):
            if (len(request.files) == 0):
                some_json = request.get_json() # request --> json
                modelname = some_json['modelname'] # nombre del archivo con el modelo
                preprocess = some_json['preprocess'] # boolean preprocess
                filename = some_json['filename'].rsplit(".",1)[0] # nombre del archivo local
                fileextension = some_json['filename'].rsplit(".",1)[1] # extensión del archivo local
                newfilename = openFile(filename, fileextension, preprocess)
            else:
                modelname = request.form['modelname'] # nombre del archivo con el modelo
                preprocess = request.form['preprocess'] # boolean preprocess
                formfile = request.files[''] # obtiene el archivo de formdata
                filename = formfile.filename.rsplit(".",1)[0] # nombre del archivo
                fileextension = formfile.filename.rsplit(".",1)[1] # extensión del archivo
                newfilename = openFile(filename, fileextension, preprocess, formfile)
        elif (request.method == 'GET'):
            modelname = request.args.get('modelname') # nombre del archivo con el modelo
            preprocess = request.args.get('preprocess') # boolean preprocess
            filename = request.args.get('filename').rsplit(".",1)[0] # nombre del archivo local
            fileextension = request.args.get('filename').rsplit(".",1)[1] # extensión del archivo local
            newfilename = openFile(filename, fileextension, preprocess)
        # Se carga el modelo
        model = fasttext.load_model('{}.bin'.format(modelname))
        result = {}
        result['score'] = model.test('{}'.format(newfilename))
        result['label'] = model.test_label('{}'.format(newfilename))
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('Exception occurred {}'.format(e))
        return jsonify(str(e)), 400

@app.route('/validate', methods=['GET','POST'])
def validate_controller():
    try:
        if (request.method == 'POST'):
            if (len(request.files) == 0):
                raise Exception('Invalid request, send formdata with trainfile and testfile')
            else:
                # trainfile = request.files['trainfile'] # obtiene el archivo de formdata
                # filename = trainfile.filename.rsplit(".",1)[0] # nombre del archivo
                # fileextension = trainfile.filename.rsplit(".",1)[1] # extensión del archivo
                # newtrainfilename = openFile(filename, fileextension, True, trainfile)
                # testfile = request.files['testfile'] # obtiene el archivo de formdata
                # filename = testfile.filename.rsplit(".",1)[0] # nombre del archivo
                # fileextension = testfile.filename.rsplit(".",1)[1] # extensión del archivo
                # newtestfilename = openFile(filename, fileextension, True, testfile)
                formfile = request.files[''] # obtiene el archivo de formdata
                filename = formfile.filename.rsplit(".",1)[0] # nombre del archivo
                fileextension = formfile.filename.rsplit(".",1)[1] # extensión del archivo
                newfilename = openFile(filename, fileextension, 'True', formfile, 'True')
        elif (request.method == 'GET'):
            raise Exception('Invalid request, send formdata with trainfile and testfile')
        # Entrenamiento del modelo
        ft = {
            'lr': 1.00,
            'dim': 100,
            'epoch': 250,
            'minCount': 1,
            'wordNgrams': 2
        }
        ft = namedtuple("FastTextParams", ft.keys())(*ft.values())
        model = fasttext.train_supervised(input='{}_trainset.csv'.format(filename), lr=ft.lr, dim=ft.dim, epoch=ft.epoch, minCount=ft.minCount, wordNgrams=ft.wordNgrams)
        result = {}
        result['score'] = model.test('{}_testset.csv'.format(filename))
        result['label'] = model.test_label('{}_testset.csv'.format(filename))
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('Exception occurred {}'.format(e))
        return jsonify(str(e)),401

@app.route('/classifier', methods=['GET','POST'])
def classifier_controller():
    try:
        if (request.method == 'POST'):
            if (len(request.files) == 0):
                some_json = request.get_json() # request --> json
                result = classifierMethod_fn(some_json['model'], some_json['text'])
            else:
                preprocess = request.form['preprocess'] # boolean preprocess
                formfile = request.files[''] # obtiene el archivo de formdata
                filename = formfile.filename.rsplit(".",1)[0] # nombre del archivo
                fileextension = formfile.filename.rsplit(".",1)[1] # extensión del archivo
                result = classifierMethod_fn(request.form['modelname'], formfile, 'file')
        elif (request.method == 'GET'):
            result = classifierMethod_fn(request.args.get('model'), request.args.get('text'))
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('ERROR: {}'.format(e))
        return jsonify(str(e)), 400

@app.route('/preprocess', methods=['GET','POST'])
def preprocess_controller():
    try:
        if (request.method == 'POST'):
            if (len(request.files) == 0):
                some_json = request.get_json() # request --> json
                filename = some_json['filename'].rsplit(".",1)[0] # nombre del archivo local
                fileextension = some_json['filename'].rsplit(".",1)[1] # extensión del archivo local
                newfilename = openFile(filename, fileextension, 'True')
            else:
                formfile = request.files[''] # obtiene el archivo de formdata
                filename = formfile.filename.rsplit(".",1)[0] # nombre del archivo
                fileextension = formfile.filename.rsplit(".",1)[1] # extensión del archivo
                newfilename = openFile(filename, fileextension, 'True', formfile)
            result = newfilename
        elif (request.method == 'GET'):
            result = 'Use POST method'
        return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200
    except Exception as e:
        logging.exception('ERROR: {}'.format(e))
        return jsonify(str(e)), 400

@app.route('/values')
def valuesController():
    result = {
        'about': 'FLASK APP RUNING',
        'version': 'FastText 0.9.2'
    }
    return app.response_class(json.dumps(result, sort_keys=False), mimetype=app.config['JSONIFY_MIMETYPE']), 200

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8082)
    #app.run(host='0.0.0.0',port=9010)