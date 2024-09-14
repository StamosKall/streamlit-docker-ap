# Χρήση μιας ελαφριάς εικόνας Python ως βάση
FROM python:3.9-slim

# Ορισμός του working directory στο container
WORKDIR /app

# Αντιγραφή του αρχείου requirements.txt και εγκατάσταση των εξαρτήσεων
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή του υπόλοιπου κώδικα της εφαρμογής
COPY . .

# Έκθεση της θύρας στην οποία θα τρέχει η εφαρμογή
EXPOSE 8501

# Εντολή για την εκκίνηση της εφαρμογής
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
