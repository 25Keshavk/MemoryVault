# Set base image (host OS)
FROM python:3.10-alpine

# By default, listen on port 8080
EXPOSE 8080/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY app.py .

ENV OPENAI_API_KEY sk-proj-dDBY2lCOVHZ3MaT4kIZcT3BlbkFJvs6HCORVKIUlcj3cSvcQ
ENV PINECONE_API_KEY 315cbdee-bc39-496c-846d-3bbce7475195


# Specify the command to run on container start
CMD [ "python", "./app.py" ]