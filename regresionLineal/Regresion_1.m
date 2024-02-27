%Regresión Lineal

load accidents

x = hwydata(:, 14); % : Seleccionar todas las filas de la columna 14, población de automovilistas por estados.
y = hwydata(:, 4); % [b0, b1], table2array
format long;

scatter(x, y) % Obtener el diagrama de dispersión

b1 = x\y; %Retornar solamente un coeficiente (pendiente).

yCalc1 = b1*x; %Recta sin b0
hold on; %Mantener graficas,evita crear otras
plot(x, yCalc1);
xlabel('Automovilistas')
ylabel('Accidentes')
title('Accidentes automovilisticos') %emg, avg, csv excel
grid on

X = [(ones (length(x),1)), x]; %Calcular el origen, necesito incorporar de longitud x por una columna.
b = X\y; %Incluye el origen y la pendiente.

yCalc2 = b(1) + b(2)*x;
plot(x, yCalc2, '--')
legend('Data', 'Regresión sin Intercepción', 'Regresión Lineal Estandar', 'Location','best') %Location best es para evitar que la leyenda oculte información importante.

%R1 = 1 - sum((y - yCalc1))
