## Set-up entorno y librería base
#### Teniendo conda ya instalado
```powershell
# Actualizar conda y usar el solver libmamba (solo necesario si es conda --version <= 23.10)
conda update -n base -c conda-forge conda python
# conda config --set solver libmamba       

# Inicializar conda en PowerShell
conda init powershell
# Reiniciar PowerShell
```
#### Crear el entorno
```powershell
conda create -n plasticc_vae python=3.10 cudatoolkit=11.8 pytorch=2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge

conda activate plasticc_vae

## Set-up entorno y librería base
#### Teniendo conda ya instalado
```powershell
# Actualizar conda y usar el solver libmamba (solo necesario si es conda --version <= 23.10)
conda update -n base -c conda-forge conda python
# conda config --set solver libmamba       

# Inicializar conda en PowerShell
conda init powershell
# Reiniciar PowerShell
```
#### Crear el entorno
```powershell
conda create -n plasticc python=3.10 cudatoolkit=11.8 pytorch=2.2 pytorch-cuda=11.8 pytorch-lightning=2.2 numpy=1.26.4 pandas scikit-learn matplotlib astropy umap-learn onnx -c pytorch -c nvidia -c conda-forge

conda activate plasticc

# Solo estos dos realmente necesitan pip
pip install wandb onnxruntime
```
### Guardar el entorno
```powershell
conda env export --from-history > env\environment.yml

pip freeze > env\requirements.txt
```

### Vericar con python
```python
import torch, platform

print("Python", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version reported by torch:", torch.version.cuda)
```

```
### Guardar el entorno
```powershell
conda env export --from-history > env\environment.yml

pip freeze > env\requirements.txt
```

### Vericar con python
```python
import torch, platform

print("Python", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version reported by torch:", torch.version.cuda)
```
