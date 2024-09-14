# Web-Based Data Mining Application

## Περιγραφή

Αυτή η εφαρμογή επιτρέπει την εξόρυξη και ανάλυση δεδομένων μέσω μιας **web-based** διεπαφής, αναπτύχθηκε με χρήση του **Streamlit** και πακετάρεται με **Docker** για εύκολη διανομή και εκτέλεση σε οποιοδήποτε περιβάλλον.

## Οδηγίες Εκτέλεσης της Εφαρμογής μέσω Docker

### Προαπαιτούμενα:
- **Docker** πρέπει να είναι εγκατεστημένο στον υπολογιστή σας.

### Βήματα Εκτέλεσης:

1. **Κλωνοποιήστε το αποθετήριο του έργου, δημιουργήστε το Docker image και εκκινήστε το container**:
   ```bash
   git clone https://github.com/StamosKall/streamlit-docker-ap.git && \
   cd streamlit-docker-ap && \
   docker build -t my-streamlit-app . && \
   docker run -p 8501:8501 my-streamlit-app

Ανοίξτε τον browser:

Αφού εκκινήσετε το container, ανοίξτε έναν browser και επισκεφθείτε τη διεύθυνση:

http://localhost:8501

Η εφαρμογή θα εμφανιστεί εκεί και θα είναι έτοιμη προς χρήση.

