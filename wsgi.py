from flask_lifter_api import app, train_models

# Train models on startup
train_models()

if __name__ == '__main__':
    app.run()