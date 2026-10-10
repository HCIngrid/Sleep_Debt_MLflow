# Experimentación de deuda de sueño con MLflow

Este directorio contiene el modelo base de deuda de sueño, su refactorización intermedia y un script de experimentación reutilizable e instrumentado con MLflow.

## Archivos actuales

- `notebooks/TC5061_Notebook_Baseline_HernandezIngrid.ipynb`: notebook original con el análisis exploratorio y el modelo base.
- `notebooks/TC5061_Experimentacion_MLFlow_HernandezIngrid.ipynb`: notebook intermedio organizado en funciones.
- `sleep_debt_mlflow/modeling/train.py`: script de entrenamiento y evaluación ejecutable desde la línea de comandos.
- `data/raw/bedtime_screentime_sleep_debt.csv`: dataset utilizado por el proyecto.
- `requirements.txt`: dependencias directas y versiones del entorno de trabajo.

Los notebooks se conservan únicamente como antecedentes históricos y exploratorios del desarrollo. El punto de entrada oficial para el entrenamiento y la experimentación es `sleep_debt_mlflow/modeling/train.py`.

## Entorno utilizado

El proyecto utiliza un entorno virtual creado con `venv`.

Desde la raíz del proyecto, crear el entorno con:

```powershell
python -m venv .venvmlflow
```

Activar el entorno desde PowerShell:

```powershell
.\.venvmlflow\Scripts\Activate.ps1
```

Actualizar `pip` e instalar las dependencias declaradas:

```powershell
python -m pip install --upgrade pip
python -m pip install -r "requirements.txt"
```

Para ejecutar los notebooks con este entorno, seleccione `Python (.venvmlflow)` como kernel del notebook en Visual Studio Code.

Antes de ejecutar el script de entrenamiento, primero se debe iniciar la interfaz de MLflow. Con la interfaz activa, ejecutar el script desde otra terminal donde `.venvmlflow` también esté activado.

## Servidor de MLflow

Después de crear y activar `.venvmlflow` e instalar las dependencias, iniciar la interfaz de MLflow en el mismo entorno activo:

```powershell
mlflow ui
```

De manera predeterminada, la interfaz estará disponible en <http://127.0.0.1:5000>. El notebook y el script utilizan esa misma dirección de seguimiento.

El experimento se registra con el nombre `Logistic Regression Sleep Debt`.

## Ejecución del script

Con el servidor de MLflow activo, la configuración predeterminada se ejecuta desde la raíz del proyecto con:

```powershell
python -m sleep_debt_mlflow.modeling.train
```

El script localiza automáticamente el dataset en `data/raw/bedtime_screentime_sleep_debt.csv`. También acepta una ruta diferente:

```powershell
python -m sleep_debt_mlflow.modeling.train --data-path ".\ruta\datos.csv"
```

Los tres hiperparámetros y la semilla se pueden cambiar sin editar el código:

```powershell
python -m sleep_debt_mlflow.modeling.train --C 0.5 --solver lbfgs --max-iter 1500 --seed 777 --run-name "lr-c05"
```

Para consultar todos los argumentos y sus valores predeterminados:

```powershell
python -m sleep_debt_mlflow.modeling.train --help
```

Cada ejecución crea una sola run. Los archivos de predicciones y matrices de confusión se generan temporalmente, se registran como artefactos y se eliminan del sistema local al finalizar.

## Flujo actual

1. Carga el dataset.
2. Elimina `user_id` y `next_day_fatigue_score`.
3. Define los tipos categóricos y el orden de la variable objetivo.
4. Conserva las gráficas del análisis exploratorio en los notebooks, pero no las ejecuta desde `train.py`.
5. Divide los datos en 70% entrenamiento, 15% validación y 15% prueba mediante una partición estratificada.
6. Ajusta el preprocesamiento únicamente con los datos de entrenamiento.
7. Entrena una regresión logística con el conjunto de entrenamiento.
8. Registra las métricas de entrenamiento y validación.
9. Reentrena el modelo final con entrenamiento y validación.
10. Evalúa una sola vez sobre prueba y registra el modelo final.

Todo el flujo de entrenamiento, validación y prueba se agrupa en una sola run de MLflow.

## Configuraciones sugeridas para las cinco corridas

Las siguientes ejecuciones mantienen la misma semilla y modifican los hiperparámetros del modelo:

```powershell
python -m sleep_debt_mlflow.modeling.train --C 1.0 --solver lbfgs --max-iter 1000 --seed 777 --run-name "baseline"
python -m sleep_debt_mlflow.modeling.train --C 0.1 --solver lbfgs --max-iter 1000 --seed 777 --run-name "c-01"
python -m sleep_debt_mlflow.modeling.train --C 10.0 --solver lbfgs --max-iter 1000 --seed 777 --run-name "c-10"
python -m sleep_debt_mlflow.modeling.train --C 1.0 --solver saga --max-iter 1000 --seed 777 --run-name "solver-saga"
python -m sleep_debt_mlflow.modeling.train --C 1.0 --solver lbfgs --max-iter 5 --seed 777 --run-name "max-iter-5"
```

## Configuración predeterminada

Los hiperparámetros actuales de la regresión logística son:

| Parámetro | Valor |
|---|---:|
| `C` | `1.0` |
| `solver` | `lbfgs` |
| `max_iter` | `1000` |

La semilla predeterminada es `random_state=777`.

`lbfgs` utiliza regularización L2 de forma predeterminada.

## Información registrada en MLflow

La run registra explícitamente:

- Los tres hiperparámetros del modelo.
- La semilla aleatoria y los porcentajes de validación y prueba.
- `accuracy`, `recall_macro`, `f1_macro`, `balanced_accuracy` y `gmean` para entrenamiento, validación y prueba.
- Los archivos de predicciones de cada partición.
- Las matrices de confusión de cada partición.
- El pipeline final de preprocesamiento y regresión logística mediante `mlflow.sklearn.log_model`.
- La firma del modelo y un ejemplo de entrada.

## Reproducibilidad

La semilla controla tanto la partición estratificada como el modelo. Dos ejecuciones con los mismos datos, dependencias y parámetros deben producir la misma partición y los mismos resultados.
