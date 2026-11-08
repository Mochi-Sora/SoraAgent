<#
.SYNOPSIS
    SoraAgent one-line installer for Windows.

.DESCRIPTION
    Installs SoraAgent into an isolated virtual environment and puts the
    sora, sora-gateway, and sora-memory commands on your PATH. Run
    `sora update` to upgrade. Requires Python 3.9 or newer.

.EXAMPLE
    irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1 | iex

.EXAMPLE
    $env:SORA_EXTRAS = "none"; irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1 | iex

.EXAMPLE
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1))) -Uninstall
#>
& {
    param([object[]]$argList)

    $ErrorActionPreference = "Stop"
    $Repo = if ($env:SORA_REPO) { $env:SORA_REPO } else { "Mochi-Sora/SoraAgent" }
    $Package = "sora-agent"
    $DefaultExtras = "openai,memory,cli,files"
    $sourceState = @{ Temp = $null }

    # Arguments are parsed manually because `param()` is not valid when a
    # script is executed through `irm ... | iex`. This also keeps direct
    # .ps1 execution and invoked script blocks working.
    $Version = if ($env:SORA_VERSION) { $env:SORA_VERSION } else { "latest" }
    $Extras = if ($null -ne $env:SORA_EXTRAS) { $env:SORA_EXTRAS } else { $DefaultExtras }
    $Source = $env:SORA_SOURCE
    $InstallDir = $env:SORA_INSTALL_DIR
    $BinDir = $env:SORA_BIN_DIR
    $Uninstall = $false
    $Purge = $false
    $NoPath = $false
    $Help = $false

    function Get-OptionValue {
        param(
            [string]$Option,
            [string]$InlineValue,
            [int]$Index,
            [object[]]$Arguments
        )
        if ($InlineValue) { return $InlineValue }
        if ($Index + 1 -ge $Arguments.Count) { throw "$Option requires a value" }
        return [string]$Arguments[$Index + 1]
    }

    for ($i = 0; $i -lt $argList.Count; $i++) {
        $arg = [string]$argList[$i]
        $option = $arg
        $inline = ""
        if ($arg -match '^([^=]+)=(.+)$') {
            $option = $Matches[1]
            $inline = $Matches[2]
        }

        switch -Regex ($option) {
            '^(--?version|-v)$' {
                $Version = Get-OptionValue -Option $option -InlineValue $inline -Index $i -Arguments $argList
                if (-not $inline) { $i++ }
                break
            }
            '^(--?extras|-e)$' {
                $Extras = Get-OptionValue -Option $option -InlineValue $inline -Index $i -Arguments $argList
                if (-not $inline) { $i++ }
                break
            }
            '^(--?source)$' {
                $Source = Get-OptionValue -Option $option -InlineValue $inline -Index $i -Arguments $argList
                if (-not $inline) { $i++ }
                break
            }
            '^(--?installdir|--install-dir)$' {
                $InstallDir = Get-OptionValue -Option $option -InlineValue $inline -Index $i -Arguments $argList
                if (-not $inline) { $i++ }
                break
            }
            '^(--?bindir|--bin-dir)$' {
                $BinDir = Get-OptionValue -Option $option -InlineValue $inline -Index $i -Arguments $argList
                if (-not $inline) { $i++ }
                break
            }
            '^(--?nopath|--no-path)$' { $NoPath = $true; break }
            '^(--?uninstall)$' { $Uninstall = $true; break }
            '^(--?purge)$' { $Purge = $true; break }
            '^(--?help|-h)$' { $Help = $true; break }
            default { throw "unknown option: $arg (try -Help)" }
        }
    }

    if (-not $InstallDir) {
        if ($env:LOCALAPPDATA) {
            $InstallDir = Join-Path $env:LOCALAPPDATA "SoraAgent"
        } else {
            $InstallDir = Join-Path $HOME ".local\share\sora-agent"
        }
    }
    if (-not $BinDir) { $BinDir = Join-Path $InstallDir "bin" }

    function Write-Info([string]$Message) {
        Write-Host "  $Message"
    }

    function Write-Warn([string]$Message) {
        Write-Host "warning: $Message" -ForegroundColor Yellow
    }

    function Assert-SafeDirectory([string]$Directory, [string]$Label) {
        if (-not $Directory) { throw "$Label is empty" }
        $full = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
        $root = [IO.Path]::GetPathRoot($full).TrimEnd('\')
        if (-not $full -or $full -eq $root) {
            throw "$Label must not be a drive root"
        }
        if ($HOME) {
            $homeDir = [IO.Path]::GetFullPath($HOME).TrimEnd('\')
            if ($full -eq $homeDir) { throw "$Label must not be HOME" }
            if ($homeDir.StartsWith($full + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "$Label must not be an ancestor of HOME"
            }
        }
    }

    function Show-Usage {
        @"
SoraAgent installer

Usage:
  install.ps1 [options]

Options:
  -Version VERSION   Release tag to install, or "main" (default: latest)
  -Extras EXTRAS     Comma-separated extras, e.g. "openai,memory,cli,files",
                     or "none" for the dependency-free core
                     (default: $DefaultExtras)
  -Source PATH       Install from a local checkout instead of GitHub
  -InstallDir DIR    Install root (default: $InstallDir)
  -BinDir DIR        Directory for command wrappers (default: $BinDir)
  -NoPath            Do not add the bin directory to your user PATH
  -Uninstall         Remove SoraAgent
  -Purge             With -Uninstall, also delete ~\.sora_agent data
  -Help              Show this help

Environment variables:
  SORA_VERSION, SORA_EXTRAS, SORA_SOURCE, SORA_INSTALL_DIR, SORA_BIN_DIR,
  SORA_PYTHON, SORA_REPO

Examples:
  irm https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.ps1 | iex
  $env:SORA_EXTRAS = "none"; irm .../install.ps1 | iex
  & ([scriptblock]::Create((irm .../install.ps1))) -Version main
  & ([scriptblock]::Create((irm .../install.ps1))) -Uninstall
"@ | Write-Host
    }

    function Remove-FromUserPath([string]$Directory) {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not $userPath) { return }
        $entries = @($userPath -split ";" | Where-Object { $_ -and $_.TrimEnd("\") -ne $Directory.TrimEnd("\") })
        if ($entries.Count -eq 0) {
            [Environment]::SetEnvironmentVariable("Path", $null, "User")
        } else {
            [Environment]::SetEnvironmentVariable("Path", ($entries -join ";"), "User")
        }
    }

    function Uninstall-SoraAgent {
        foreach ($name in @("sora", "sora-gateway", "sora-memory")) {
            foreach ($suffix in @(".cmd", ".ps1", ".bat")) {
                $path = Join-Path $BinDir ($name + $suffix)
                if (-not (Test-Path $path)) { continue }
                $content = Get-Content -Raw $path -ErrorAction SilentlyContinue
                $managed = $content -and (
                    $content -match 'managed by sora-agent installer' -or
                    $content -match [regex]::Escape($InstallDir) -or
                    $content -match [regex]::Escape("venv\Scripts\$name.exe")
                )
                if ($managed) {
                    Remove-Item -Force $path
                } else {
                    Write-Warn "leaving $path; it was not created by this installer"
                }
            }
        }

        if (Test-Path $InstallDir) {
            $venvConfig = Join-Path $InstallDir "venv\pyvenv.cfg"
            if (Test-Path $venvConfig) {
                Write-Info "Removing $InstallDir"
                Remove-Item -Recurse -Force $InstallDir
            } else {
                Write-Warn "refusing to remove $InstallDir: it does not look like a SoraAgent install"
            }
        }

        Remove-FromUserPath $BinDir

        if ($Purge) {
            if (-not $HOME) { throw "HOME is not set; refusing to purge user data" }
            $dataDir = Join-Path $HOME ".sora_agent"
            if (Test-Path $dataDir) {
                Write-Info "Removing $dataDir (memory, identity, sessions, calendar)"
                Remove-Item -Recurse -Force $dataDir
            }
        } else {
            Write-Info "Keeping user data in $(Join-Path $HOME '.sora_agent') (use -Purge to delete it)"
        }

        Write-Host ""
        Write-Host "SoraAgent uninstalled."
    }

    function Find-Python {
        $commands = @()
        if ($env:SORA_PYTHON) {
            $file = $env:SORA_PYTHON
            if (-not (Get-Command $file -ErrorAction SilentlyContinue) -and -not (Test-Path $file)) {
                throw "SORA_PYTHON=$file was not found"
            }
            $commands += @{ File = $file; Args = @() }
        }
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $commands += @{ File = (Get-Command py).Source; Args = @("-3") }
        }
        if (Get-Command python -ErrorAction SilentlyContinue) {
            $commands += @{ File = (Get-Command python).Source; Args = @() }
        }
        if (Get-Command python3 -ErrorAction SilentlyContinue) {
            $commands += @{ File = (Get-Command python3).Source; Args = @() }
        }

        foreach ($candidate in $commands) {
            $check = @($candidate.Args) + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)")
            & $candidate.File @check 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }

        throw "Python 3.9 or newer is required, but it was not found. Install Python from https://www.python.org/downloads/ and re-run this installer."
    }

    function Resolve-Source {
        if ($Source) {
            if (-not (Test-Path $Source)) { throw "source directory not found: $Source" }
            $resolved = (Resolve-Path $Source).Path
            if (-not (Test-Path (Join-Path $resolved "pyproject.toml"))) {
                throw "$resolved does not look like a SoraAgent checkout (pyproject.toml missing)"
            }
            return $resolved
        }

        if ($Version -eq "latest") {
            Write-Info "Looking up the latest SoraAgent release..."
            try {
                $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -TimeoutSec 20 -Headers @{ Accept = "application/vnd.github+json" }
                $tag = $release.tag_name
            } catch {
                $tag = ""
            }
            if ($tag) {
                $ref = "refs/tags/$tag"
                $label = $tag
            } else {
                $ref = "refs/heads/main"
                $label = "main (no release found)"
            }
        } elseif ($Version -eq "main" -or $Version -eq "master") {
            $ref = "refs/heads/$Version"
            $label = $Version
        } else {
            $tag = $Version
            if (-not $tag.StartsWith("v")) { $tag = "v$tag" }
            $ref = "refs/tags/$tag"
            $label = $tag
        }

        $tmp = Join-Path ([IO.Path]::GetTempPath()) ("sora-agent-" + [Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $tmp | Out-Null
        try {
            $url = "https://github.com/$Repo/archive/$ref.zip"
            $zip = Join-Path $tmp "sora-agent.zip"
            Write-Info "Downloading SoraAgent $label..."
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
            } catch {
                throw "could not download $url"
            }
            Expand-Archive -Path $zip -DestinationPath $tmp -Force

            $dirs = @(Get-ChildItem -Path $tmp -Directory)
            if ($dirs.Count -eq 0) { throw "the downloaded archive did not contain a source directory" }
            $resolved = $dirs[0].FullName
            if (-not (Test-Path (Join-Path $resolved "pyproject.toml"))) {
                throw "$resolved does not look like a SoraAgent checkout (pyproject.toml missing)"
            }
            $sourceState.Temp = $tmp
            return $resolved
        } catch {
            Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
            throw
        }
    }

    if ($Help) {
        Show-Usage
        return
    }

    Assert-SafeDirectory $InstallDir "InstallDir"
    Assert-SafeDirectory $BinDir "BinDir"

    if ($Uninstall) {
        Uninstall-SoraAgent
        return
    }

    if ($Purge) {
        throw "-Purge is only valid with -Uninstall"
    }

    if ((Test-Path $InstallDir) -and -not (Test-Path (Join-Path $InstallDir "venv\pyvenv.cfg"))) {
        $existing = Get-ChildItem -Force $InstallDir -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existing) {
            throw "$InstallDir is not empty and does not look like a SoraAgent install"
        }
    }

    # GitHub requires TLS 1.2; Windows PowerShell 5.1 does not always enable it.
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {
        # ignored; newer PowerShell versions do not need this
    }

    $python = Find-Python
    $pyArgs = @($python.Args)
    $pythonVersion = (& $python.File @pyArgs -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))' | Out-String).Trim()
    Write-Info "Using Python $pythonVersion"

    $venvDir = Join-Path $InstallDir "venv"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

    if (-not (Test-Path $venvPython)) {
        Write-Info "Creating virtual environment in $venvDir..."
        & $python.File @pyArgs -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -Recurse -Force $venvDir -ErrorAction SilentlyContinue
            throw "failed to create the virtual environment (ensure the Python venv module is available)"
        }
    }

    Write-Info "Installing $Package..."
    $pipLog = Join-Path ([IO.Path]::GetTempPath()) ("sora-agent-pip-" + [Guid]::NewGuid().ToString("N") + ".log")
    try {
        & $venvPython -m pip install --quiet --disable-pip-version-check --upgrade pip setuptools wheel *> $pipLog
        if ($LASTEXITCODE -ne 0) {
            Get-Content $pipLog | Write-Host
            throw "failed to update pip, setuptools, and wheel"
        }

        $sourceDir = Resolve-Source
        $spec = "."
        if ($Extras -and $Extras -ne "none") {
            $spec = ".[$Extras]"
        }

        Push-Location $sourceDir
        try {
            & $venvPython -m pip install --quiet --disable-pip-version-check --no-build-isolation --upgrade --force-reinstall --no-cache-dir $spec *> $pipLog
            if ($LASTEXITCODE -ne 0) {
                Get-Content $pipLog | Write-Host
                throw "pip install failed"
            }
        } finally {
            Pop-Location
        }
    } finally {
        Remove-Item -Force $pipLog -ErrorAction SilentlyContinue
        if ($sourceState.Temp) {
            Remove-Item -Recurse -Force $sourceState.Temp -ErrorAction SilentlyContinue
            $sourceState.Temp = $null
        }
    }

    $defaultBin = Join-Path $InstallDir "bin"
    $useRelativePaths = $BinDir.TrimEnd("\") -eq $defaultBin.TrimEnd("\")
    foreach ($name in @("sora", "sora-gateway", "sora-memory")) {
        $exe = Join-Path $venvDir "Scripts\$name.exe"
        if (-not (Test-Path $exe)) { throw "installation completed but $name.exe is missing" }
        $cmd = Join-Path $BinDir "$name.cmd"
        if (Test-Path $cmd) { Remove-Item -Force $cmd }
        if ($useRelativePaths) {
            $target = "%~dp0..\venv\Scripts\$name.exe"
            Set-Content -Path $cmd -Value @("@echo off", "rem managed by sora-agent installer", "`"$target`" %*") -Encoding ASCII
        } else {
            Set-Content -Path $cmd -Value @("@echo off", "rem managed by sora-agent installer", "`"$exe`" %*") -Encoding Default
        }
    }

    if (-not $NoPath) {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not $userPath) { $userPath = "" }
        $entries = @($userPath -split ";" | Where-Object { $_ })
        if (-not ($entries | Where-Object { $_.TrimEnd("\") -eq $BinDir.TrimEnd("\") })) {
            $newPath = if ($userPath) { "$BinDir;$userPath" } else { $BinDir }
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
            $env:Path = "$BinDir;$env:Path"
            Write-Host "Added $BinDir to your user PATH (open a new terminal to use it)."
        }
    }

    & $venvPython -m sora_agent.cli --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn "sora did not start correctly" }
    & $venvPython -m sora_agent.channels --help 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn "sora-gateway did not start correctly" }
    & $venvPython -m sora_agent.memory --help 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn "sora-memory did not start correctly" }

    Write-Host ""
    Write-Host "SoraAgent installed."
    Write-Host ""
    Write-Host "  sora setup"
    Write-Host "  sora --help"
    Write-Host "  sora-gateway --help"
    Write-Host "  sora-memory doctor"
    Write-Host ""
    Write-Host "Install root: $InstallDir"
    Write-Host "Commands:     $BinDir"
    Write-Host ""
    if ($Extras -and $Extras -ne "none" -and (($Extras -split ",") -contains "cli")) {
        Write-Host "To upgrade later, run: sora update"
    } else {
        Write-Host "To upgrade later, re-run this installer (the CLI extra is not installed)."
    }
    Write-Host "To remove SoraAgent:"
    Write-Host "  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/$Repo/main/install.ps1))) -Uninstall"
} -argList $args
