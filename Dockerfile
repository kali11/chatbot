FROM python:3.11-bookworm

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY chainlit.md .
COPY chat_service.py .

CMD ["chainlit", "run", "-h", "app.py"]