<#
  Coqui TTS server helpers for PowerShell
  - Start-TTSServer    : lanza el contenedor server (detached)
  - Stop-TTSServer     : para y borra el contenedor
  - Restart-TTSServer  : reinicia el contenedor
  - Get-TTSServerStatus: muestra si está corriendo
  - List-TTSModels     : ejecuta --list_models dentro del contenedor (si está corriendo)
#>

function Start-TTSServer {
    param(
        [string] $ContainerName = "coqui_tts_server",
        [string] $HostOutDir = "C:\Users\soyko\Documents\tts-output",
        [string] $HostDataset = "C:\Users\soyko\Documents\tts-dataset",
        [string] $HostModels = "C:\Users\soyko\Documents\tts-models",
        [string] $ModelName = "tts_models/multilingual/multi-dataset/xtts_v2",
        [string] $HostTTS = "C:\Users\soyko\Documents\tts",
        [string] $ModelRecipes = "C:\Users\soyko\Documents\tts-recipes",
        [switch] $UseCuda, 
        [switch] $RemoveIfExists
    )

    $exists = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}" 2>$null
    if ($exists -and $RemoveIfExists) {
        docker stop $ContainerName 2>$null | Out-Null
        docker rm $ContainerName 2>$null | Out-Null
        $exists = $null
    }

    if ($exists) {
        Write-Output "Contenedor $ContainerName ya existe. Usa Restart-TTSServer o Stop-TTSServer."
        return
    }

    New-Item -ItemType Directory -Path $HostOutDir -Force | Out-Null
    New-Item -ItemType Directory -Path $HostDataset -Force | Out-Null
    New-Item -ItemType Directory -Path $HostModels -Force | Out-Null

    $use_cuda_arg = if ($UseCuda) { "--use_cuda true" } else { "--use_cuda false" }

    docker run --rm -it `
        --name $ContainerName `
        --gpus all `
        -p 5002:5002 `
        -v "${HostOutDir}:/root/tts-output" `
        -v "${HostDataset}:/root/dataset" `
        -v "${HostModels}:/root/.local/share/tts" `
        -v "${HostTTS}\server:/root/TTS/server" `
        -v "${HostTTS}\utils:/root/TTS/utils" `
        -v "${ModelRecipes}:/root/recipes/" `
        --entrypoint /bin/bash `
        ghcr.io/coqui-ai/tts

}

function Stop-TTSServer {
    param([string] $ContainerName = "coqui_tts_server")
    Write-Output "Parando y eliminando contenedor $ContainerName..."
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
    Write-Output "Hecho."
}

function Restart-TTSServer {
    param([string] $ContainerName = "coqui_tts_server")
    Stop-TTSServer -ContainerName $ContainerName
    Start-TTSServer -ContainerName $ContainerName
}

function Get-TTSServerStatus {
    param([string] $ContainerName = "coqui_tts_server")
    $ps = docker ps --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    if (-not $ps) {
        Write-Output "No hay contenedor '$ContainerName' en ejecución."
    } else {
        Write-Output $ps
    }
}

function List-TTSModels {
    param([string] $ContainerName = "coqui_tts_server")
    $running = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
    if (-not $running) {
        Write-Error "Contenedor $ContainerName no está corriendo. Inicia con Start-TTSServer."
        return
    }
    docker exec -it $ContainerName python3 TTS/server/server.py --list_models
}


#python3 TTS/server/server.py --model_path /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 --config_path  /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json --use_cuda true
