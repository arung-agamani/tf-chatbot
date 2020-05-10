from flask import Flask
from flask import request, jsonify

from chatbot_server import ChatBot_Server

bot = ChatBot_Server()

server = Flask(__name__)

@server.rout('/', methods=['GET'])
def home():
    return "<h1>AAAAAAAAAAAAAAAAA</h1>"

@server.route('/chatbot', methods=['POST'])
def handleChat():
    print("Chat request inbound...")
    raw_message = request.json['rawMessage']
    previous_intent = request.json['previousIntent']
    print("raw_message:", raw_message)
    response_message = bot.chatbot_response(str(raw_message))
    print(response_message)
    response_body = {}
    response_body["status"] = 200
    response_body["message"] = response_message
    return response_body

# response_message = bot.chatbot_response("Can you help me?")
# print(response_message)
server.run(port=2000, threaded=False)

