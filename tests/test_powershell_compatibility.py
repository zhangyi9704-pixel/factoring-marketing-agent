"""Exercise the native stderr regression in the real Windows PowerShell 5.1 engine."""
import base64
import os
from pathlib import Path
import subprocess

import pytest

PS51 = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
pytestmark = pytest.mark.skipif(os.name != "nt" or not PS51.exists(), reason="Windows PowerShell 5.1 required")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("native_exit", [0, 7])
def test_compose_native_stderr_is_captured_and_exit_code_decides(native_exit):
    native = base64.b64encode(("[Console]::Out.WriteLine('offline stdout'); "
                              "[Console]::Error.WriteLine('offline diagnostic'); "
                              f"exit {native_exit}").encode("utf-16-le")).decode()
    script = r"""$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path (Get-Location).Path 'deploy/docker/verify.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'parse failed' }
$definition = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-Compose'}, $true)
Invoke-Expression $definition.Extent.Text
$composeArgs = @()
$nativePowerShell = Join-Path $PSHOME 'powershell.exe'
$nativeCommand = '__NATIVE__'
function docker {
    & $nativePowerShell -NoProfile -NonInteractive -EncodedCommand $nativeCommand
    $global:LASTEXITCODE = $LASTEXITCODE
}
$caught = $null
try { $captured = Invoke-Compose -Arguments @('offline') } catch { $caught = $_.Exception.Message }
if ($ErrorActionPreference -ne 'Stop') { throw 'error preference leaked' }
if (__EXIT__ -eq 0) {
    if ($caught -or $captured -notmatch 'offline stdout' -or $captured -notmatch 'offline diagnostic') { throw 'successful native stderr was rejected or lost' }
} else {
    if ($caught -ne 'Compose operation failed (exit __EXIT__); inspect the service safely.') { throw 'native failure was not redacted' }
}
Write-Output 'PASS'
""".replace("__NATIVE__", native).replace("__EXIT__", str(native_exit))
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    result = subprocess.run([str(PS51), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"
