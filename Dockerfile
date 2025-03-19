FROM python:3.10.12-slim-buster

RUN apt update
RUN apt-get install python3-pip -y
# RUN apt-get install -y git

# Set the working directory to /usr/src/app.
WORKDIR /usr/src/app

# Copy the file from the local host to the filesystem of the container at the working directory.
COPY requirements.txt .

RUN pip install --upgrade pip

# Install specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# #Install gpt2 transformer for tokenizer
# RUN python3 -c "import transformers; transformers.AutoTokenizer.from_pretrained('gpt2')"

# #Install NLTK 
# RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy the project source code from the local host to the filesystem of the container at the working directory.
COPY . .

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]

