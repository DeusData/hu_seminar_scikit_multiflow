FROM python:3.8

# Set Working directory
WORKDIR app
ENV PYTHONPATH=/app

# Copy requirements file
COPY requirements.txt ./

# Install requirements
RUN pip3 install -r ./requirements.txt

CMD ["python3", "-u", "./main.py"]
