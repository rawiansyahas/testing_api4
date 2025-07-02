FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download model during build (if using URL)
RUN python -c "import os; import requests; model_url = os.environ.get('MODEL_URL'); print('Downloading model...') if model_url else None; r = requests.get(model_url) if model_url else None; open('vggface_model.h5', 'wb').write(r.content) if model_url else None; print('Model downloaded') if model_url else print('No MODEL_URL provided')"

EXPOSE 8000

CMD ["python", "app.py"]