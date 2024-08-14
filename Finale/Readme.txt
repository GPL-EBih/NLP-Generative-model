docker build -t nlp-lstm .

docker run -p 5000:5000 nlp-lstm
docker run -v /d/HCMUS/Github/NLP-Generative-model/Finale:/app -p 5000:5000 nlp-lstm

,,