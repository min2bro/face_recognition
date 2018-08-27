# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
from flask import Flask,render_template,request,make_response,jsonify
import face_recognition
import pickle
import cv2,io
import urllib.request
import numpy as np
import base64 
from PIL import Image
import requests
import pyttsx3
from skimage import io
engine = pyttsx3.init()

app = Flask(__name__)


def facedidentification(img):
# app = Flask(__name__)
	# print(img)


# @app.route('/facedidentification/',methods=['GET'])
# def facedidentification():
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-e", "--encodings", required=True,
	# 	help="path to serialized db of facial encodings")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	# 	help="face detection model to use: either `hog` or `cnn`")
	# args = vars(ap.parse_args())
# base64_string_url=request.args.get('base64_stringi')
	# base64_string=base64.b64encode(requests.get(img).content)
# base64_string=base64_string_url
# print("=========================================================================")
	# print(base64_string)
# print("=========================================================================")

# load the known faces and embeddings


	print("[INFO] loading encodings...")
	encode='encodings.pickle'
	data = pickle.loads(open(encode, "rb").read())

	base64_string='''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'''

	# load the input image and convert it from BGR to RGB
	# image='examples/aishwarya2.jpg'
	# print(base64.b64decode(base64_string))
	# img = np.fromstring(base64.b64decode(base64_string), np.uint8)
	# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# imgdata = base64.b64decode(base64_string)
	# image=Image.open(io.BytesIO(imgdata))
	# rgb=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	# cv2.imwrite("reconstructed.jpg", rgb)
	# rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# cv2.imwrite("reconstructed.jpg", rgb)
	# print(img)
	# image = cv2.imread(img)
	# print(image)
	# rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# url_response=urllib.request.urlopen(img)
	# img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
	# print(nparr)
	# img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	# image = cv2.imdecode(img_array, -1)
	image=cv2.imread(img)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model='cnn')
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		if name != 'Unknown':
			for k in counts.items():
				count_ante=k[1]
			print(count_ante)
			if(count_ante>1):
				names.append(name)
			else:
				names.append("Unknown")
		else:
			names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	print (names)
	engine.say("Hello "+names[0]+ " How are you")

	engine.runAndWait()
	
	return encodings,
	
# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

@app.route('/facedidentificationwithbase64',methods=['POST'])
def facedidentificationwithbase64():
	# base64_string=base64.b64encode(requests.get(img).content)
	# base64_string_url=request.args.get(imageString)

	if request.method == "POST":
		json_dict = request.get_json()
		base64_string_url = json_dict['imageString']


	print("[INFO] loading encodings...")
	encode='encodings.pickle'
	data = pickle.loads(open(encode, "rb").read())

	

	decoded_string = base64.b64decode(base64_string_url)
	nparr = np.fromstring(decoded_string, dtype=np.uint8)
	image = cv2.imdecode(nparr, -1)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model='cnn')
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		if name != 'Unknown':
			for k in counts.items():
				count_ante=k[1]
			print(count_ante)
			if(count_ante>=1):
				names.append(name)
			else:
				names.append("Unknown")
		else:
			names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	return jsonify(names)
	# engine.say("Hello "+names[0]+ " How are you")

	# engine.runAndWait()

if __name__ == "__main__":
	app.run(host='0.0.0.0',port=9763,debug=True)
	# facedidentification('https://i.imgur.com/vesXfIB.png')
	# facedidentificationwithbase64('https://i.imgur.com/vesXfIB.png')