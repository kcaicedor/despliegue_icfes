FROM python:3.12-slim

# Crear usuario sin privilegios de administrador
RUN adduser --disabled-password --gecos '' dash-user

# Definir directorio de trabajo
WORKDIR /opt/app

# Copiar los archivos del tablero al contenedor
COPY . /opt/app/

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/app/requirements.txt

# Asignar permisos al usuario restringido
RUN chown -R dash-user:dash-user /opt/app

USER dash-user

# Puerto expuesto por Dash
EXPOSE 8050

# Iniciar el tablero
CMD ["python", "app.py"]