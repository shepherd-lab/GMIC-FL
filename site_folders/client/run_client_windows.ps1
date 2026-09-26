#requires -Version 5.1
<#
.SYNOPSIS
    Build and start the FL client container using Podman Compose, then open an interactive shell inside it.

.DESCRIPTION
    This script uses Podman (NOT Docker) to:
      1) Build the container image using Podman Compose
      2) Start the container detached
      3) Drop you into a bash shell inside the running container

.PARAMETER DataDir
    Path to the data directory to mount in the container

.PARAMETER GPUs
    GPU devices to use. Options: "all", "none", or specific GPU IDs like "0,1"
    Default: "all"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data" -GPUs "0,1"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data" -GPUs "none"
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DataDir,

    [Parameter(Position=1)]
    [string]$GPUs = "all"
)

$ErrorActionPreference = "Stop"

# Always run from the script's directory so Compose finds docker-compose.yml
Set-Location -Path $PSScriptRoot
$env:CURRENT_DIR = $PSScriptRoot

# ---- Messaging helpers ----
function Write-StatusMessage {
    param(
        [string]$Message,
        [string]$Icon = "[INFO]",
        [string]$Color = "White"
    )
    Write-Host "$Icon $Message" -ForegroundColor $Color
}
function Write-SuccessMessage { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[SUCCESS]" -Color "Green" }
function Write-ErrorMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[ERROR]"   -Color "Red"   }
function Write-InfoMessage    { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[INFO]"    -Color "Cyan"  }
function Write-BuildMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[BUILD]"   -Color "Yellow" }
function Write-LaunchMessage  { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[LAUNCH]"  -Color "Magenta" }

# Validate -GPUs format: 'all', 'none', or '0,1,2'
if ($GPUs -notmatch '^(all|none|\d+(,\d+)*)$') {
    Write-ErrorMessage "Invalid -GPUs value '$GPUs' (use 'all', 'none', or e.g. '0,1')"
    exit 1
}

# Ensure UID/GID environment variables exist for Compose interpolation (harmless if unused)
try {
    if (-not $env:UID) { $env:UID = "1000" }
    if (-not $env:GID) { $env:GID = "1000" }
    Write-InfoMessage "Environment UID=$($env:UID), GID=$($env:GID)"
}
catch {
    Write-ErrorMessage "Failed to set UID/GID environment variables: $($_.Exception.Message)"
    exit 1
}

# --- PODMAN CHECKS / MACHINE BOOTSTRAP ---
function Test-PodmanAvailable {
    try {
        & podman --version | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Ensure-PodmanEngine {
    if (-not (Test-PodmanAvailable)) {
        Write-ErrorMessage "Podman CLI not found. Install Podman Desktop (recommended) or Podman CLI and ensure 'podman' is on PATH."
        exit 1
    }

    # First try podman info (works only if machine/engine is running)
    try {
        & podman info | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-InfoMessage "Podman engine detected (podman info OK)."
            return
        }
    } catch { }

    # Try starting default machine (common on Windows/macOS)
    Write-InfoMessage "Podman engine not reachable yet. Attempting to start Podman machine..."
    try {
        & podman machine start | Out-Null
    } catch { }

    # Re-try
    try {
        & podman info | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-InfoMessage "Podman machine started and engine is reachable."
            return
        }
    } catch { }

    Write-ErrorMessage "Podman engine still not reachable. Try running:  podman machine start"
    exit 1
}

# --- COMPOSE DETECTION ---
# We prefer:  podman compose ...
# Fallback:   podman-compose ...
$composeMode = $null  # "plugin" or "legacy"

function Detect-PodmanCompose {
    # Try podman compose (plugin)
    try {
        $v = & podman compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $script:composeMode = "plugin"
            Write-InfoMessage "Podman Compose (plugin): $v"
            return
        }
    } catch { }

    # Try podman-compose (python tool)
    try {
        $v = & podman-compose --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $script:composeMode = "legacy"
            Write-InfoMessage "podman-compose (legacy): $v"
            return
        }
    } catch { }

    Write-ErrorMessage "Podman Compose is not available. Install Podman Desktop (recommended) or install 'podman-compose'."
    exit 1
}

function Invoke-Compose {
    param([Parameter(Mandatory=$true)][string[]]$Args)

    # CRITICAL:
    # PowerShell captures anything written to the success output stream as the function's "return value".
    # External command output can be captured and pollute $rc (e.g., $rc becomes an array of log lines + exit code).
    # Pipe to Out-Host to force streaming to the console (not capture), then read $LASTEXITCODE.
    if ($script:composeMode -eq "plugin") {
        & podman compose @Args | Out-Host
        return [int]$LASTEXITCODE
    } elseif ($script:composeMode -eq "legacy") {
        & podman-compose @Args | Out-Host
        return [int]$LASTEXITCODE
    } else {
        throw "Compose mode not set"
    }
}

try {
    # === Validate data directory ===
    if (-not (Test-Path $DataDir -PathType Container)) {
        Write-ErrorMessage "Data directory '$DataDir' does not exist."
        exit 1
    }

    # === Normalize paths ===
    $DataDirResolved = Resolve-Path $DataDir

    # === Set environment variables for Compose interpolation ===
    $env:DATA_DIR    = $DataDirResolved.Path
    $env:GPU_DEVICES = $GPUs

    # === Display summary ===
    Write-Host ""
    Write-Host "[CONFIG] FL Container Configuration" -ForegroundColor Blue
    Write-Host ("=" * 50) -ForegroundColor Blue
    Write-InfoMessage "Script directory: $env:CURRENT_DIR"
    Write-InfoMessage "Data directory:   $($env:DATA_DIR)"
    Write-InfoMessage "GPU devices:      $($env:GPU_DEVICES)"
    Write-Host ""

    # === Ensure Podman engine is available ===
    Ensure-PodmanEngine

    # === Detect Podman Compose ===
    Detect-PodmanCompose

    Write-Host ""

    # === Build image ===
    Write-BuildMessage "Building image (Podman Compose)..."
    Write-Host ""
    $rc = Invoke-Compose -Args @("build","gmic_fl")
    if ($rc -ne 0) {
        Write-ErrorMessage "Build failed with exit code $rc"
        exit $rc
    }
    Write-SuccessMessage "Image built successfully"
    Write-Host ""

    # === Start container detached ===
    Write-LaunchMessage "Starting container (detached)..."
    Write-Host ""
    $rc = Invoke-Compose -Args @("up","-d","gmic_fl")
    if ($rc -ne 0) {
        Write-ErrorMessage "Failed to start container (compose up -d) with exit code $rc"
        exit $rc
    }
    Write-SuccessMessage "Container started: gmic_fl_container"
    Write-Host ""

    # === Drop user into container shell ===
    Write-InfoMessage "Opening interactive shell in container (podman exec)..."
    Write-Host ""
    & podman exec -it gmic_fl_container bash -lc @'
cd /workspace
echo "Inside container."
echo ""
echo "  1) Next steps:"
echo "       cd /workspace/startup"
echo "       chmod +x start_fl.sh"
echo "       ./start_fl.sh"
echo ""
echo "  2) Attach later:"
echo "       tmux attach -t fl_client"
echo ""
exec bash
'@

    $rc = $LASTEXITCODE
    Write-Host ""
    if ($rc -eq 0) {
        Write-SuccessMessage "Exited container shell."
    } else {
        Write-ErrorMessage "podman exec returned exit code $rc"
        exit $rc
    }
}
catch {
    Write-ErrorMessage "Script failed: $($_.Exception.Message)"
    exit 1
}
finally {
    Write-Host ""
    Write-InfoMessage "Cleaning up environment variables..."
    Remove-Item Env:DATA_DIR     -ErrorAction SilentlyContinue
    Remove-Item Env:GPU_DEVICES  -ErrorAction SilentlyContinue
    Remove-Item Env:CURRENT_DIR  -ErrorAction SilentlyContinue
}

Write-Host ""
Write-SuccessMessage "Done."