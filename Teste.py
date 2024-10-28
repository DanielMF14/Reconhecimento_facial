import cv2

# Carregar o classificador Haar Cascade para detecção de rosto
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Ler o frame da webcam
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Converter o frame para escala de cinza (melhora a eficiência da detecção)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar um retângulo em torno de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibir o frame com as detecções
    cv2.imshow('Detecção Facial', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
