# Build
Dentro del directorio `docker/` ejecutar:
```
docker build -t weather_pred:latest . 
```

# Uso
El script de inferencia lee un archivo `input.csv` y produce un archivo `predictions.csv` ambos dentro del directorio `/app/data/`.

Dentro de un directorio que contenga un archivo `input.csv` ejecutar:
```
docker run --rm -u $(id -u):$(id -g) -v ".:/app/data" weather_pred:latest
```

* `--rm` para eliminar el contenedor una vez realizada la predicción.
* `-u $(id -u):$(id -g)` para generar el archivo `predictions.csv` con el uid y gid del usuario que lo ejecuta.
* `-v ".:/app/data"` para montar el directorio actual como /app/data .

