FROM python:3.8-slim

RUN mkdir /streamlit

COPY streamlit_requriements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r streamlit_requriements.txt

COPY . /streamlit

EXPOSE 8502

CMD streamlit run app.py --server.port 8502