# SMS Spam Detector

Streamlit machine-learning application that classifies a message as SPAM or HAM.

## Deployment
- Framework: Streamlit
- Model: Multinomial Naive Bayes
- Text features: CountVectorizer
- Hosting: Render Web Service

## Render settings
Build Command:
`pip install -r requirements.txt`

Start Command:
`streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
