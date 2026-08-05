# Launch Plan A scale tokenizer EC2 host (r7i.2xlarge, host-built Docker).
# Mirrors networking/instance-profile used by the live plan-b-pools-pull instance.
#
# Prerequisites: stage_plan_a_ec2_bundle.py already ran.
# Usage:
#   pwsh scripts/launch_plan_a_ec2.ps1
#   pwsh scripts/launch_plan_a_ec2.ps1 -DryRun

param(
    [string]$Profile = $(if ($env:AWS_PROFILE) { $env:AWS_PROFILE } else { "sbsandbox" }),
    [string]$Region = $(if ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "us-east-1" }),
    [string]$InstanceType = "r7i.2xlarge",
    [string]$AmiId = "ami-08bc385c9fc5afc94",  # AL2023 x86_64 (same as plan-b pull)
    [string]$SubnetId = "subnet-0bbe2b7870da13713",
    [string]$SecurityGroupId = "sg-087218d8c87aa8576",
    [string]$IamInstanceProfile = "edullm-downloader",
    [string]$CorpusS3Root = "s3://edullm-datasets/_scratch/plan-a-fineweb",
    [int]$VolumeGb = 100,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$UserDataPath = Join-Path $Root "artifacts\plan_a_ec2_userdata.sh"
if (-not (Test-Path $UserDataPath)) {
    throw "Missing userdata: $UserDataPath"
}

# Inject CORPUS_S3_ROOT into userdata without rewriting the committed file.
# AWS CLI base64-encodes file:// user-data; do not pre-encode.
$UserDataBody = Get-Content -Raw -Path $UserDataPath
$Header = @"
#!/bin/bash
export CORPUS_S3_ROOT='$CorpusS3Root'
export AWS_REGION='$Region'
"@
$UserDataBody = $UserDataBody -replace '^#!.*\r?\n', ''
$Combined = ($Header + "`n" + $UserDataBody) -replace "`r`n", "`n"
$TempUserData = Join-Path $env:TEMP ("plan-a-userdata-{0}.sh" -f [guid]::NewGuid().ToString("n"))
# Write UTF-8 without BOM; Unix newlines.
[System.IO.File]::WriteAllText($TempUserData, $Combined, [System.Text.UTF8Encoding]::new($false))

Write-Host "EC2 value reminder: $InstanceType ~`$0.53/h on-demand us-east-1 + ${VolumeGb}GB gp3."
Write-Host "Setup builds Docker (can take 30+ min). Scale trains are NOT started by userdata."
Write-Host "Profile=$Profile Region=$Region Ami=$AmiId Subnet=$SubnetId SG=$SecurityGroupId ProfileName=$IamInstanceProfile"

$blockDevice = "DeviceName=/dev/xvda,Ebs={VolumeSize=${VolumeGb},VolumeType=gp3,DeleteOnTermination=true}"
# file:// needs a forward-slash URI on Windows for awscli.
$UserDataUri = "file://" + ($TempUserData -replace '\\', '/')

$awsArgs = @(
    "ec2", "run-instances",
    "--profile", $Profile,
    "--region", $Region,
    "--image-id", $AmiId,
    "--instance-type", $InstanceType,
    "--subnet-id", $SubnetId,
    "--security-group-ids", $SecurityGroupId,
    "--iam-instance-profile", "Name=$IamInstanceProfile",
    "--block-device-mappings", $blockDevice,
    "--associate-public-ip-address",
    "--user-data", $UserDataUri,
    "--metadata-options", "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled",
    "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=plan-a-tokenizer-scale},{Key=Project,Value=plan-a},{Key=Owner,Value=aryan.verma}]",
    "--query", "Instances[0].{InstanceId:InstanceId,State:State.Name,Type:InstanceType,Az:Placement.AvailabilityZone}",
    "--output", "json"
)

if ($DryRun) {
    $awsArgs += "--dry-run"
    Write-Host "Dry-run only."
}

$out = & aws @awsArgs 2>&1
Write-Host $out
if ($LASTEXITCODE -ne 0 -and -not ($DryRun -and ("$out" -match "DryRunOperation"))) {
    throw "run-instances failed: $out"
}

if (-not $DryRun) {
    $json = $out | Out-String | ConvertFrom-Json
    $iid = $json.InstanceId
    Write-Host ""
    Write-Host "Launched $iid. Poll setup:"
    Write-Host "  aws s3 cp $CorpusS3Root/_STATUS.txt - --profile $Profile --region $Region"
    Write-Host "  aws ssm start-session --target $iid --profile $Profile --region $Region"
    Write-Host "  aws ec2 describe-instances --instance-ids $iid --profile $Profile --region $Region --query `"Reservations[0].Instances[0].State.Name`""
    Write-Host ""
    Write-Host "When _STATUS is SETUP_OK, start trains on the host:"
    Write-Host "  sudo /usr/local/bin/run_plan_a_scale_docker.sh bpe"
    Write-Host "  sudo /usr/local/bin/run_plan_a_scale_docker.sh superbpe"
}
