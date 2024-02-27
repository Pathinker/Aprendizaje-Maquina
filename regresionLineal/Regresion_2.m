%Importar mi set de datos.

format long
format long g
nombreArchivo = "Excel/pib_jalisco_actividades_secundarias_precios_corrientes_2003-2020.xlsx";
Datos = readtable(nombreArchivo, "Range", "A:B");

Independiente = obtenerDatos(Datos, 1);
Dependiente = obtenerDatos(Datos, 2);

x = table2array(Independiente);
y = table2array(Dependiente);

scatter(x,y);
hold on

xlabel("Año")
ylabel("Cantidad")
title("Jalisco Actividades Secundarias")

X = [(ones (length(x),1)), x]; %Calcular el origen, necesito incorporar de longitud x por una columna.
b = X\y; %Incluye el origen y la pendiente.

yCalc2 = b(1) + b(2)*x;
plot(x, yCalc2, '--');
xlabel("Año")
ylabel("Cantidad")
title("Jalisco Actividades Secundarias")
legend('Datos', 'Regresión Lineal Estandar', 'Location','best')
fprintf("Ecuación Regresión linea %f + %f x", b(1), b(2));

function Temporal = obtenerDatos(Variable, Columna) %Erradicar valores que no sean numeros.
    Temporal = [];

    for i = 1:size(Variable, 1)
       
        if ~isnan(Variable{i, Columna}) %En una tabla debo acceder a los elementos por llaves, no parentesis
     
            Temporal = [Temporal; Variable(i, Columna)];
            
        end
    end
end
