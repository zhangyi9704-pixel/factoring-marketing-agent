param(
    [ValidateSet('Health', 'Seed', 'Verify')]
    [string]$Mode = 'Health',
    [ValidatePattern('^[a-f0-9]{12}$')]
    [string]$RunId,
    [int]$WaitSeconds = 180
)

# Run from the project root. No Python, credentials printed, or business initialization.
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ((Get-Location).Path -ne $projectRoot) { throw 'Run this script from the project root.' }
$composeArgs = @('compose', '--env-file', '.env', '-f', 'deploy/docker/compose.yaml')
function Invoke-Compose {
    param([string[]]$Arguments, [string]$InputText)
    # Windows PowerShell 5.1 turns native stderr into ErrorRecord even on exit 0.
    # Capture diagnostics privately and decide success using the native exit code.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ($PSBoundParameters.ContainsKey('InputText')) {
            $output = @($InputText | & docker @composeArgs @Arguments 2>&1)
        } else {
            $output = @(& docker @composeArgs @Arguments 2>&1)
        }
        $composeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($composeExitCode -ne 0) {
        # Do not dump command arguments, environment, or unfiltered diagnostic output.
        throw "Compose operation failed (exit $composeExitCode); inspect the service safely."
    }
    return (($output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
}
function Assert-Equal([string]$Actual, [string]$Expected, [string]$Label) {
    if ($Actual -ne $Expected) { throw "$Label did not return the expected result." }
    Write-Output "PASS: $Label"
}
function Invoke-Milvus([string]$Operation, [hashtable]$Body) {
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:19530/v2/vectordb/$Operation" `
        -Method Post -ContentType 'application/json' -Body ($Body | ConvertTo-Json -Depth 12 -Compress) `
        -TimeoutSec 60
    if ($result.code -ne 0) { throw "Milvus $Operation failed, code $($result.code)." }
    return $result
}

# Only read-only health probes repeat; seed writes and restarts are never retried here.
$deadline = (Get-Date).AddSeconds($WaitSeconds)
do {
    $states = Invoke-Compose -Arguments @('ps', '--all', '--format', '{{.Service}}|{{.Health}}|{{.State}}')
    $lines = @($states -split '\r?\n' | Where-Object { $_ })
    $ready = $lines.Count -eq 6 -and @($lines | Where-Object { $_ -notmatch '\|healthy\|running$' }).Count -eq 0
    if (-not $ready) {
        if ((Get-Date) -ge $deadline) { throw 'Six healthy services were not observed before timeout.' }
        Start-Sleep -Seconds 5
    }
} until ($ready)
Write-Output $states

$sqlCommand = 'export MYSQL_PWD="$MYSQL_PASSWORD"; exec mysql --protocol=TCP --host=127.0.0.1 --user="$MYSQL_USER" --database="$MYSQL_DATABASE" --batch --skip-column-names --execute="SELECT 1"'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'mysql', 'sh', '-ec', $sqlCommand)) '1' 'MySQL application-user SELECT 1'
$mysqlSettings = 'export MYSQL_PWD="$MYSQL_PASSWORD"; exec mysql --protocol=TCP --host=127.0.0.1 --user="$MYSQL_USER" --database="$MYSQL_DATABASE" --batch --skip-column-names --execute="SELECT DATABASE(); SELECT @@default_authentication_plugin; SELECT @@character_set_server;"'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'mysql', 'sh', '-ec', $mysqlSettings)) "factoring_coach`ncaching_sha2_password`nutf8mb4" 'MySQL database, default authentication and character set'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis', 'sh', '-ec', 'export REDISCLI_AUTH="$REDIS_PASSWORD"; exec redis-cli ping')) 'PONG' 'Redis authenticated PING'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', 'export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"; exec redis-cli ping')) 'PONG' 'Redis Stack authenticated PING'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis', 'redis-cli', 'ping')) 'NOAUTH Authentication required.' 'Redis rejects anonymous PING'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'redis-cli', 'ping')) 'NOAUTH Authentication required.' 'Redis Stack rejects anonymous PING'
$modules = Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', 'export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"; exec redis-cli INFO modules')
if ($modules -notmatch 'module:name=search,' -or $modules -notmatch 'module:name=ReJSON,') { throw 'RedisJSON or RediSearch missing.' }
Write-Output 'PASS: RedisJSON + RediSearch modules'
foreach ($service in 'redis', 'redis-stack') {
    $passwordName = if ($service -eq 'redis') { 'REDIS_PASSWORD' } else { 'CHECKPOINT_REDIS_PASSWORD' }
    $command = 'export REDISCLI_AUTH="${' + $passwordName + '}"; exec redis-cli --raw CONFIG GET appendonly'
    Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', $service, 'sh', '-ec', $command)) "appendonly`nyes" "$service AOF enabled"
}
$etcdHealth = Invoke-Compose -Arguments @('exec', '-T', 'etcd', 'etcdctl', 'endpoint', 'health')
if ($etcdHealth -notmatch 'is healthy') { throw 'etcd endpoint health failed.' }
Write-Output 'PASS: etcd endpoint health'
$null = Invoke-Compose -Arguments @('exec', '-T', 'minio', 'curl', '--fail', '--silent', 'http://localhost:9000/minio/health/live')
Write-Output 'PASS: MinIO HTTP health'
Assert-Equal ([string](Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:9091/healthz' -TimeoutSec 15).StatusCode) '200' 'Milvus host HTTP healthz'

# Inspect only bindings and mount names. Never dump docker inspect/config with Env.
foreach ($service in 'etcd', 'minio', 'milvus', 'mysql', 'redis', 'redis-stack') {
    $containerId = Invoke-Compose -Arguments @('ps', '-q', $service)
    $bindingsText = & docker inspect --format '{{json .NetworkSettings.Ports}}' $containerId
    if ($LASTEXITCODE -ne 0) { throw 'Port inspection failed.' }
    $bindings = $bindingsText | ConvertFrom-Json
    $expected = switch ($service) {
        'mysql' { @{ '3306/tcp' = '3307' } }
        'redis' { @{ '6379/tcp' = '6379' } }
        'redis-stack' { @{ '6379/tcp' = '6380' } }
        'milvus' { @{ '19530/tcp' = '19530'; '9091/tcp' = '9091' } }
        default { @{} }
    }
    $observed = @{}
    if ($null -ne $bindings) {
        foreach ($property in $bindings.PSObject.Properties) {
            foreach ($binding in @($property.Value)) {
                if ($null -eq $binding) { continue }
                if ($binding.HostIp -ne '127.0.0.1') { throw "$service has a non-loopback binding." }
                $observed[$property.Name] = $binding.HostPort
            }
        }
    }
    if ($observed.Count -ne $expected.Count) { throw "$service has unexpected published ports." }
    foreach ($port in $expected.Keys) {
        if ($observed[$port] -ne $expected[$port]) { throw "$service port mapping differs." }
    }
    $mountNames = & docker inspect --format '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}} {{end}}{{end}}' $containerId
    $expectedVolume = 'factoring_' + $(if ($service -eq 'redis-stack') { 'redis_stack' } else { $service }) + '_data'
    if ($LASTEXITCODE -ne 0 -or $mountNames.Trim() -ne $expectedVolume) { throw "$service data volume differs or an anonymous volume is attached." }
    Write-Output "PASS: $service bindings and volume $mountNames"
}
if ($Mode -eq 'Health') { return }
if (-not $RunId) { throw 'Seed/Verify requires a 12-character lowercase hex RunId.' }
$marker = "envcheck_$RunId"
$database = "factoring_envcheck_$RunId"
$collection = "envcheck_$RunId"
$key = "envcheck:$RunId"
$index = "envcheck_idx_$RunId"
$bucket = "factoring-envcheck-$RunId"
$rootSql = 'export MYSQL_PWD="$MYSQL_ROOT_PASSWORD"; exec mysql --user=root --batch --skip-column-names'
$minioAlias = 'mc alias set envcheck http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null; '

if ($Mode -eq 'Seed') {
    # Unique infrastructure-only names; never use business tables or factoring_knowledge.
    $sql = "CREATE DATABASE $database; CREATE TABLE $database.persistence_probe (id INT PRIMARY KEY, marker VARCHAR(64)); INSERT INTO $database.persistence_probe VALUES (1, '$marker');"
    $null = Invoke-Compose -Arguments @('exec', '-T', 'mysql', 'sh', '-ec', $rootSql) -InputText $sql
    Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'etcd', 'etcdctl', 'put', "/factoring/envcheck/$RunId", $marker)) 'OK' 'etcd probe write'
    $redisWrite = 'export REDISCLI_AUTH="$REDIS_PASSWORD"; exec redis-cli SET "$1" "$2"'
    Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis', 'sh', '-ec', $redisWrite, 'probe', $key, $marker)) 'OK' 'Redis probe write'
    $jsonWrite = 'export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"; exec redis-cli -x JSON.SET "$1" "$"'
    Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', $jsonWrite, 'probe', $key) -InputText ('{"marker":"' + $marker + '"}')) 'OK' 'Stack JSON.SET'
    $searchCreate = @'
export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"
exec redis-cli FT.CREATE "$1" ON JSON PREFIX 1 "$2" SCHEMA '$.marker' AS marker TAG
'@
    Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', $searchCreate, 'probe', $index, $key)) 'OK' 'Stack FT.CREATE JSON index'
    $minioWrite = $minioAlias + 'mc mb "envcheck/$1" >/dev/null; printf "%s" "$2" | mc pipe "envcheck/$1/probe.txt" >/dev/null'
    $null = Invoke-Compose -Arguments @('exec', '-T', 'minio', 'sh', '-ec', $minioWrite, 'probe', $bucket, $marker)
    Write-Output 'PASS: MinIO authenticated object write'
    $null = Invoke-Milvus 'collections/create' @{collectionName=$collection; dimension=2; metricType='COSINE'; params=@{consistencyLevel='Strong'}}
    $insert = Invoke-Milvus 'entities/insert' @{collectionName=$collection; data=@(@{id=1; vector=@(1.0, 0.0); marker=$marker})}
    if ($insert.data.insertCount -ne 1) { throw 'Milvus probe insert count differs.' }
    $null = Invoke-Milvus 'collections/flush' @{collectionName=$collection}
    Write-Output 'PASS: Milvus isolated probe collection and vector write/flush'
    Write-Output "Seed complete, RunId=$RunId. Restart only this Compose project, then run -Mode Verify with the same RunId."
    return
}

Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'mysql', 'sh', '-ec', $rootSql) -InputText "SELECT marker FROM $database.persistence_probe WHERE id=1;") $marker 'MySQL persisted marker'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'etcd', 'etcdctl', 'get', "/factoring/envcheck/$RunId", '--print-value-only')) $marker 'etcd persisted marker'
$redisRead = 'export REDISCLI_AUTH="$REDIS_PASSWORD"; exec redis-cli --raw GET "$1"'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis', 'sh', '-ec', $redisRead, 'probe', $key)) $marker 'Redis persisted marker'
$jsonRead = 'export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"; exec redis-cli --raw JSON.GET "$1"'
$json = Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', $jsonRead, 'probe', $key) | ConvertFrom-Json
Assert-Equal $json.marker $marker 'Stack persisted JSON marker'
$searchRead = 'export REDISCLI_AUTH="$CHECKPOINT_REDIS_PASSWORD"; exec redis-cli --raw FT.SEARCH "$1" "$2" NOCONTENT'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'redis-stack', 'sh', '-ec', $searchRead, 'probe', $index, "@marker:{$marker}")) "1`n$key" 'Stack persisted search index and result'
$minioRead = $minioAlias + 'exec mc cat "envcheck/$1/probe.txt"'
Assert-Equal (Invoke-Compose -Arguments @('exec', '-T', 'minio', 'sh', '-ec', $minioRead, 'probe', $bucket)) $marker 'MinIO persisted object'
$null = Invoke-Milvus 'collections/load' @{collectionName=$collection}
$query = Invoke-Milvus 'entities/query' @{collectionName=$collection; filter='id == 1'; outputFields=@('id', 'marker'); consistencyLevel='Strong'}
if (@($query.data).Count -ne 1 -or $query.data[0].marker -ne $marker) { throw 'Milvus persisted vector marker differs.' }
Write-Output 'PASS: Milvus persisted vector marker'
Write-Output "Verify complete, RunId=$RunId. Probe data retained under isolated envcheck names."
