#requires -Version 3.0

<#
.SYNOPSIS
    Run FL server with Docker Compose

.DESCRIPTION
    This script builds and runs a federated learning server using Docker Compose.

.EXAMPLE
    .\run_server.ps1"
    
#>

# Set error action preference
$ErrorActionPreference = "Stop"

# Always run from the script's directory so Compose finds docker-compose.yml
Set-Location -Path $PSScriptRoot
$env:CURRENT_DIR = $PSScriptRoot

function Write-SuccessMessage { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[SUCCESS]" -Color "Green" }
function Write-ErrorMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[ERROR]"   -Color "Red"   }
function Write-InfoMessage    { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[INFO]"    -Color "Cyan"  }
function Write-BuildMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[BUILD]"   -Color "Yellow"}
function Write-LaunchMessage  { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[LAUNCH]"  -Color "Magenta"}
# ---- Messaging helpers ----
function Write-StatusMessage {
    param(
        [string]$Message,
        [string]$Icon = "[INFO]",
        [string]$Color = "White"
    )
    Write-Host "$Icon $Message" -ForegroundColor $Color
}

# Ensure UID/GID environment variables exist for Compose interpolation
try {
    if (-not $env:UID) {
        # Windows users don’t have a numeric UID, so use 1000 (Linux default)
        $env:UID = "1000"
    }
    if (-not $env:GID) {
        $env:GID = "1000"
    }
    Write-InfoMessage "Environment UID=$($env:UID), GID=$($env:GID)"
}
catch {
    Write-ErrorMessage "Failed to set UID/GID environment variables: $($_.Exception.Message)"
    exit 1
}

# Main script logic
try {

    # === Display summary ===
    Write-Host ""
    Write-Host "[CONFIG] Federated Learning Server Configuration" -ForegroundColor Blue
    Write-Host ("=" * 50) -ForegroundColor Blue
    Write-InfoMessage "Script directory: $env:CURRENT_DIR"
    Write-Host ""

    # === Check Docker CLI present and daemon running ===
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker CLI not found" }

        docker info | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }

        Write-InfoMessage "Docker Desktop and engine detected."
    }
    catch {
        Write-ErrorMessage "Docker Desktop is not installed or not running."
        exit 1
    }

    # === Check Docker Compose (new or legacy) ===
    $composeCommand = $null
    try {
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $composeCommand = "docker compose"
            Write-InfoMessage "Docker Compose: $composeVersion"
        } else {
            $composeVersion = docker-compose --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $composeCommand = "docker-compose"
                Write-InfoMessage "Docker Compose: $composeVersion"
            } else {
                throw "Docker Compose not found"
            }
        }
    }
    catch {
        Write-ErrorMessage "Docker Compose is not available."
        Write-Host "Please ensure Docker Compose is installed."
        exit 1
    }

    Write-Host ""

    # === Build Docker image ===
    Write-BuildMessage "Building Docker image..."
    Write-Host ""
    try {
        if ($composeCommand -eq "docker compose") {
            & docker compose build
        } else {
            & docker-compose build
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed with exit code $LASTEXITCODE"
        }
        Write-SuccessMessage "Docker image built successfully"
    }
    catch {
        Write-ErrorMessage "Failed to build Docker image: $($_.Exception.Message)"
        exit 1
    }

    Write-Host ""

    # === Run container ===
    Write-LaunchMessage "Starting FL server container..."
    Write-Host ""
    try {
        if ($composeCommand -eq "docker compose") {
            & docker compose up
        } else {
            & docker-compose up
        }
        & $composeCommand up; $containerExitCode = $LASTEXITCODE

        Write-Host ""
        if ($containerExitCode -eq 0) {
            Write-SuccessMessage "FL server completed successfully"
        } else {
            Write-ErrorMessage "FL server exited with code $containerExitCode"
        }
    }
    catch {
        Write-ErrorMessage "Failed to run container: $($_.Exception.Message)"
        exit 1
    }
}
catch {
    Write-ErrorMessage "Script failed: $($_.Exception.Message)"
    exit 1
}
finally {
    # === Cleanup ===
    Write-Host ""
    Write-InfoMessage "Cleaning up environment variables..."
    Remove-Item Env:CURRENT_DIR  -ErrorAction SilentlyContinue
}

Write-Host ""
Write-SuccessMessage "Script completed"