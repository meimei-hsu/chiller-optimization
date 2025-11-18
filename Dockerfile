FROM ugr-sail/sinergym:2.5.0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir stable-baselines3 matplotlib pandas numpy

CMD ["python", "env_demo.py"]
