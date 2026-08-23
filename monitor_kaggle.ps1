while ($true) {
    $status = .venv\Scripts\kaggle kernels status kkm121121/bharatsrm-net-v4-pretraining
    if ($status -match "complete") {
        Write-Host "Kernel complete! Downloading artifacts..."
        mkdir -Force kaggle_outputs
        .venv\Scripts\kaggle kernels output kkm121121/bharatsrm-net-v4-pretraining -p kaggle_outputs
        break
    } elseif ($status -match "error" -or $status -match "cancel") {
        Write-Host "Kernel stopped with status: $status"
        break
    }
    Start-Sleep -Seconds 300
}
