# Ubicación de las carpetas originales.
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path

$clientDir = Join-Path $projectRoot "dockerfiles/federated-client"
$serverDir = Join-Path $projectRoot "dockerfiles/federated-server"

# Directorio del paquete Flower.
$packageDir = Join-Path $PSScriptRoot "water_federated"

# Crea el paquete Python.
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

New-Item -ItemType File -Force `
    -Path (Join-Path $packageDir "__init__.py") | Out-Null

# Copia los módulos del cliente.
Copy-Item "$clientDir/client.py" $packageDir -Force
Copy-Item "$clientDir/model.py" $packageDir -Force
Copy-Item "$clientDir/data.py" $packageDir -Force

# Copia los módulos del servidor.
Copy-Item "$serverDir/server.py" $packageDir -Force
Copy-Item "$serverDir/evaluate.py" $packageDir -Force
Copy-Item "$serverDir/plots.py" $packageDir -Force

Write-Host "Paquete Flower preparado correctamente."