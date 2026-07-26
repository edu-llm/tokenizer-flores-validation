$ErrorActionPreference = "Stop"

$SuperBpeCommit = "bbd09768fc28a875cef48e6bdd66e3a17454628e"
$TokenizersCommit = "757f2a55c0820ed47064e1fe473deea39b7b611b"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required (https://docs.astral.sh/uv/)"
}

New-Item -ItemType Directory -Force ".cache" | Out-Null
if (-not (Test-Path ".cache/superbpe")) {
    git clone https://github.com/PythonNut/superbpe.git ".cache/superbpe"
}
git -C ".cache/superbpe" checkout $SuperBpeCommit

if (-not (Test-Path ".cache/tokenizers-superbpe")) {
    git clone https://github.com/alisawuffles/tokenizers-superbpe.git ".cache/tokenizers-superbpe"
}
git -C ".cache/tokenizers-superbpe" checkout $TokenizersCommit

uv python install 3.11
uv venv --clear --python 3.11 ".venv-benchmark"
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = $null
uv pip install --python ".venv-benchmark/Scripts/python.exe" `
    ".cache/tokenizers-superbpe/bindings/python" `
    click filelock psutil pysimdjson regex tiktoken "transformers==4.45.2"

& ".venv-benchmark/Scripts/python.exe" -c @"
import tokenizers
assert tokenizers.__version__ == "0.20.1", tokenizers.__version__
print("Official benchmark environment ready: tokenizers", tokenizers.__version__)
"@

Write-Host "SuperBPE commit: $SuperBpeCommit"
Write-Host "Patched tokenizers commit: $TokenizersCommit"

