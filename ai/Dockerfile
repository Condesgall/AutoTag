FROM python:3.10
WORKDIR /AutoTag
COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]