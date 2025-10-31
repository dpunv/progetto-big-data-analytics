param(
    [Parameter(Mandatory=$true)]
    [int]$NumNodes
)

if ($NumNodes -le 0) {
    Write-Error "Number of nodes must be positive"
    exit 1
}

Write-Host "Creating YAML configuration for $NumNodes nodes..." -ForegroundColor Cyan

python generate_config.py $NumNodes

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error generating configuration"
    exit 1
}

Write-Host "Starting main.py..." -ForegroundColor Green
python main.py
