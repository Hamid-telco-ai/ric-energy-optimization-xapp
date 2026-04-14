# ===============================
# CCO-xAPP Multipage Streamlit Dashboard + LLM Assistant
# ===============================

FROM python:3.11-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy and install dependencies ----
COPY services/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Extra libraries (explicit versions for stability) ----
RUN pip install --no-cache-dir \
    streamlit \
    "openai>=2.7.1" \
    streamlit-chat \
    streamlit-autorefresh \
    matplotlib \
    pandas

# ---- Copy app source code ----
#COPY services/dashboard/ ./
COPY services/dashboard/ /app/

# ---- Expose default Streamlit port ----
EXPOSE 8501

# ---- Launch multipage Streamlit app ----
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]





############################## My Old Docker File######################3

#FROM python:3.11-slim

#WORKDIR /app

# Install Python dependencies
#COPY services/dashboard/requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Install additional libraries for the RIC Assistant Chat
#RUN pip install --no-cache-dir streamlit openai>=2.7.1 streamlit-chat streamlit-autorefresh

# Copy dashboard code
#COPY services/dashboard/ ./

# Expose Streamlit port
#EXPOSE 8501

# Run the dashboard
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
