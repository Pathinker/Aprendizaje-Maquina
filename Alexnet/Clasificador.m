nnet = alexnet; %Nombre de la red neuronal
layers = nnet.Layers; %Acceder a las capaz de la red neuronal

%La red de alexnet contiene más de 100 capaz, las capaz 23 y 25 se encargan
%de abastraer las caracteristicas.

layers(23)=fullyConnectedLayer(5); % 5 = Número de carpetas.
layers(25)=classificationLayer;

allImages = imageDatastore("Imagenes", "IncludeSubfolders",true, "LabelSource", "foldernames"); %Extraer todas las imagenes con sus subcarpetas
allImages.ReadFcn = @(loc)adjustImageChannels(imresize(imread(loc), [227, 227])); %Redimensionar y ajustar los canales de las imágenes; %Redimensionar las imagenes al formato tolerado por Alexnet
[trainingImages, testImages] = splitEachLabel(allImages, 0.8,"randomized");

opts = trainingOptions("sgdm", "InitialLearnRate", 0.001, "MaxEpochs", 20, "MiniBatchSize",64);
myNet = trainNetwork(trainingImages, layers, opts);

predictedLabels = classify(myNet, testImages);
accuracy = mean(predictedLabels == testImages.Labels);

allImages = imageDatastore("Evaluar", "IncludeSubfolders",true);
allImages.ReadFcn =  @(loc)adjustImageChannels(imresize(imread(loc), [227, 227]));

for image_index=1:30
    picture= readimage(allImages, image_index);
    label = classify(myNet, picture);
    image(picture);
    title(char(label));
    pause(3);
end

function adjustedImage = adjustImageChannels(image) %Las imagenes de Alexnet necesitan tener 3 canales en RGB y mismo tamaño en caso de no tenerlos duplicados.
    % Verificar si la imagen tiene tres canales
    if size(image, 3) == 3
        % Si tiene tres canales, no es necesario ajustarla
        adjustedImage = image;
    else
        % Si tiene menos de tres canales, convertirla a una imagen RGB agregando canales duplicados
        adjustedImage = cat(3, image, image, image);
    end
end
