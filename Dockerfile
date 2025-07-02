FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download model during build (if using URL)
RUN python -c "
import os
import requests
model_url = os.environ.get('MODEL_URL')
if model_url:
    print('Downloading model...')
    r = requests.get(model_url)
    with open('vggface_model.h5', 'wb') as f:
        f.write(r.content)
    print('Model downloaded')
"

EXPOSE 8000

CMD ["python", "app.py"]