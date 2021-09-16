FROM python:3.8-slim

RUN mkdir /streamlit

COPY streamlit_requriements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r streamlit_requriements.txt

COPY . /streamlit

EXPOSE 8501

CMD streamlit run app.py