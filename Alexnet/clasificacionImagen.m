camera = webcam; %Reconoce la camara
nnet = alexnet; %Nombre de la red neuronal
picture = camera.snapshot; %Objeto que almancena la foto

picture = imreize(picture, [227, 227]); %Redimensionar la imagen a formato compatible con la red neuronal
label = classify(nnet, picture) %Brinde una clasificación.
image(picture);
