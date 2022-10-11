from flask import Flask, request
from bot_service import Bot
from training import BotTrainer

app = Flask(__name__)


@app.route("/")
def home():
    return "success"


@app.route("/chatbot/api/get_response", methods=['POST'])
def get_bot_response():
    user_text = request.form.get('msg')
    bot = Bot(user_text)
    output = bot.predict()
    return output


@app.route('/chatbot/api/train')
def train_bot():
    bot_trainer = BotTrainer()
    bot_trainer.train()
    print("Bot Trained with data in DB and model saved successfully")
    return "Success"


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
