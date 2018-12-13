#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, abort
import base64
import face_reco
import file_reco
import flask_cors
from io import BytesIO

app = Flask(__name__)
flask_cors.CORS(app, supports_credentials=True)

def base64topath(request, name):
    image = request.json[name]
    image = base64.b64decode(image)
    filename = BytesIO(image)
    return filename, image

def locationstojson(locations):
    dataset = {}
    dataset['sum'] = len(locations)
    index = 0
    for i in locations:
        location = {}
        location['top'] = i[0]
        location['right'] = i[1]
        location['bottom'] = i[2]
        location['left'] = i[3]
        dataset['face-%d' % index] = location
        index += 1
    result = {
        "errno": 0,
        "face_locations":dataset
    }
    return result

def landmarkstojson(face_landmarks_list):
    dataset = {}
    dataset['sum'] = len(face_landmarks_list)
    index = 0
    for face_landmarks in face_landmarks_list:
        dataset['face-%d' % index] = face_landmarks
        index += 1
    result = {
        "errno": 0,
        "face_landmarks": dataset
    }
    return result

def filelocationtojson(image, box):
    data = base64.b64encode(image)
    data = data.decode('utf-8')
    box_pro = box.tolist()
    result = {
        "errno": 0,
        "image": data,
        "file_locations": box_pro
    }
    return result

def filefillintojson(origin_image, source_image, is_not, source_point):
    origin_data = base64.b64encode(origin_image)
    origin_data = origin_data.decode('utf-8')
    source_data = base64.b64encode(source_image)
    source_data = source_data.decode('utf-8')
    result = {
        "errno": 0,
        "image_1_trans": origin_data,
        "image_2_trans": source_data,
        "image_2_point": source_point,
        "fillin": is_not
    }
    return result

@app.route('/compare', methods=['POST'])
def face_compare():
    # 读取image参数
    if not request.json or not 'image_1' in request.json:
        print(request.json, request.get_data, request.headers)
        abort(400)

    image_1, _ = base64topath(request, 'image_1')
    image_2, _ = base64topath(request, 'image_2')

    score = face_reco.face_file_compare(image_1, image_2)
    result = {
        "errno": 0,
        "score": "%.2f"%score[0]
    }
    return jsonify(result)

@app.route('/locations', methods=['POST'])
def face_locations():
    # 读取image参数
    if not request.json or not 'image' in request.json:
        print(request.json, request.get_data, request.headers)
        abort(400)

    image, _ = base64topath(request, 'image')
    locations = face_reco.face_file_locations(image)
    result = locationstojson(locations)
    return jsonify(result)

@app.route('/landmarks', methods=['POST'])
def face_landmarks():
    # 读取image参数
    if not request.json or not 'image' in request.json:
        print(request.json, request.get_data, request.headers)
        abort(400)
    image, _ = base64topath(request, 'image')
    landmarks = face_reco.face_file_landmarks(image)
    result = landmarkstojson(landmarks)
    return jsonify(result)

@app.route('/filelocations', methods=['POST'])
def file_locations():
    # 读取image参数
    if not request.json or not 'image' in request.json:
        print(request.json, request.get_data, request.headers)
        abort(400)
    _, image = base64topath(request, 'image')
    img, box = file_reco.file_locations(image)
    result = filelocationtojson(img, box)
    return jsonify(result)

@app.route('/filefillin', methods=['POST'])
def file_fillin():
    # 读取image参数
    if not request.json or not 'image_1' in request.json:
        print(request.json, request.get_data, request.headers)
        abort(400)
    _, image_1 = base64topath(request, 'image_1')
    _, image_2 = base64topath(request, 'image_2')
    point = request.json["point"]
    print(point)
    is_not , sourcepoint, image_1, image_2= file_reco.file_iffillin(image_1, point,image_2)
    result = filefillintojson(image_1, image_2, is_not, sourcepoint)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12346)